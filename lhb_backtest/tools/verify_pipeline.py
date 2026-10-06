# -*- coding: utf-8 -*-
"""Deep semantic verification of every pipeline module.

Unlike unit tests (synthetic fixtures) this script validates the REAL data
artifacts end-to-end: every check recomputes the expected value through an
independent code path and compares against what the pipeline produced.

Run:  python tools/verify_pipeline.py
Exit code 0 = all checks passed, 1 = at least one FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

RESULTS: list[tuple[str, bool, str]] = []
RNG = np.random.default_rng(42)
EPS = 1e-6


def check(section: str, name: str, ok: bool, detail: str = "", warn_only: bool = False) -> None:
    """warn_only=True marks environment/data-source issues that should be
    surfaced but are not module bugs (do not fail the exit code)."""
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append((f"{section} | {name}", bool(ok) or warn_only, detail if ok else f"[{status}] {detail}"))
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ───────────────────────── 1. calendar ─────────────────────────

def verify_calendar() -> None:
    section("1. Trading calendar")
    from src.utils.calendar import get_trading_dates, calendar_max_date, is_trading_date

    days = get_trading_dates("2019-01-01", "2026-05-26")
    check("calendar", "dates sorted & unique", days == sorted(set(days)))
    check("calendar", "plausible count 2019-2026.05 (~1780)", 1700 <= len(days) <= 1850,
          f"n={len(days)}")
    check("calendar", "weekend never a trading day",
          all(pd.Timestamp(d).dayofweek < 5 for d in days))
    check("calendar", "2025-10-01 (National Day) not trading", not is_trading_date("2025-10-01"))

    cmax = calendar_max_date()
    extension_end = (pd.Timestamp(cmax) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    ext = get_trading_dates("2019-01-01", extension_end)
    check("calendar", "extends beyond snapshot end for missing-date detection",
          len(ext) > len(days) and ext[-1] > cmax,
          f"snapshot ends {cmax}, extended to {ext[-1]}")


# ───────────────────────── 2. raw ingestion ─────────────────────────

def verify_raw() -> None:
    section("2. Raw ingestion artifacts")
    from src.utils.calendar import get_trading_dates

    kline = pd.read_parquet(PROJECT_ROOT / "data/raw/daily_kline.parquet",
                            columns=["stock_code", "trade_date", "open", "high", "low", "close"])
    dup = kline.duplicated(subset=["stock_code", "trade_date"]).sum()
    check("raw.kline", "no duplicate (stock, date)", dup == 0, f"dups={dup}")

    cal = set(get_trading_dates("2019-01-01", "2026-12-31"))
    kdates = set(kline["trade_date"].astype(str).unique())
    off_cal = kdates - cal
    check("raw.kline", "all dates are trading days", len(off_cal) == 0,
          f"{len(off_cal)} off-calendar dates" if off_cal else "")

    # OHLC integrity on a sample
    samp = kline.sample(min(len(kline), 200_000), random_state=42).dropna()
    bad = ((samp["high"] < samp["low"] - EPS)
           | (samp["high"] < samp["close"] - EPS) | (samp["high"] < samp["open"] - EPS)
           | (samp["low"] > samp["close"] + EPS) | (samp["low"] > samp["open"] + EPS)).sum()
    check("raw.kline", "OHLC integrity (high>=o/c>=low)", bad == 0, f"violations={bad}")

    per_day = kline.groupby("trade_date")["stock_code"].nunique()
    partial_days = per_day[per_day < per_day.median() * 0.5]
    check("raw.kline", "no partial days (stock count < 50% of median)",
          len(partial_days) == 0,
          f"{len(partial_days)} partial days "
          f"({partial_days.index.min()} ~ {partial_days.index.max()}, "
          f"min={per_day.min()}) — SimTradeData source snapshot incomplete; "
          "update the SimTradeData repo then rerun step 2 (loader now re-fetches "
          "partial days automatically)",
          warn_only=True)

    lhb = pd.read_parquet(PROJECT_ROOT / "data/raw/lhb_summary.parquet")
    dup = lhb.duplicated(subset=["trade_date", "stock_code"]).sum()
    check("raw.lhb", "no duplicate (date, stock) after multi-reason merge", dup == 0,
          f"dups={dup}")
    ldates = set(lhb["trade_date"].astype(str).unique())
    check("raw.lhb", "all LHB dates are trading days", len(ldates - cal) == 0,
          f"{len(ldates - cal)} off-calendar")

    # Every trading day should have SOME lhb events (spot known-good range)
    rng_days = [d for d in sorted(cal) if "2024-01-01" <= d <= "2026-05-26"]
    missing_days = [d for d in rng_days if d not in ldates]
    check("raw.lhb", "no missing trading days 2024-01~2026-05", len(missing_days) == 0,
          f"missing {len(missing_days)}: {missing_days[:5]}")

    det = pd.read_parquet(PROJECT_ROOT / "data/raw/lhb_broker_detail.parquet",
                          columns=["trade_date", "stock_code", "flag", "broker_name"])
    det_pairs = set(map(tuple, det[["trade_date", "stock_code"]].drop_duplicates().values))
    lhb_pairs = set(map(tuple, lhb[["trade_date", "stock_code"]].values))
    # Source-only investor-classification disclosures are retained without
    # fabricating a summary. Every ordinary seat event must still match.
    orphan_pairs = det_pairs - lhb_pairs
    full_detail = pd.read_parquet(PROJECT_ROOT / "data/raw/lhb_broker_detail.parquet",
                                  columns=["stock_code", "trade_date", "disclosure_kind"])
    kinds = full_detail.groupby(["trade_date", "stock_code"]).disclosure_kind.agg(set)
    unexplained = [pair for pair in orphan_pairs if kinds.loc[pair] != {"investor_category"}]
    check("raw.detail", "source-only detail events are classified", not unexplained,
          f"source-only={len(orphan_pairs)}, unexplained={len(unexplained)}")


# ───────────────────────── 3. cleaning ─────────────────────────

def verify_clean() -> None:
    section("3. Cleaning layer")
    raw = pd.read_parquet(PROJECT_ROOT / "data/raw/lhb_summary.parquet")
    clean = pd.read_parquet(PROJECT_ROOT / "data/clean/lhb_summary.parquet")

    check("clean.lhb", "row conservation raw -> clean",
          abs(len(clean) - raw.drop_duplicates(subset=["trade_date", "stock_code"]).shape[0]) == 0,
          f"raw={len(raw)}, clean={len(clean)}")

    code_ok = clean["stock_code"].astype(str).str.match(r"^\d{6}\.(SH|SZ|BJ)$").all()
    check("clean.lhb", "stock codes normalized XXXXXX.EX", code_ok)
    date_ok = clean["trade_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    check("clean.lhb", "dates normalized YYYY-MM-DD", date_ok)

    # Unit contract: clean ratio = raw percent / 100 (compare via join)
    j = clean.merge(
        raw[["trade_date", "stock_code", "net_buy_ratio"]].rename(
            columns={"net_buy_ratio": "raw_nbr"}),
        on=["trade_date", "stock_code"], how="inner",
    ).dropna(subset=["net_buy_ratio", "raw_nbr"])
    j["raw_nbr"] = pd.to_numeric(j["raw_nbr"], errors="coerce")
    samp = j.sample(min(len(j), 50_000), random_state=42).dropna()
    ok = np.allclose(samp["net_buy_ratio"], samp["raw_nbr"] / 100.0, atol=1e-9)
    check("clean.lhb", "unit contract: net_buy_ratio = raw% / 100", ok, f"checked {len(samp)}")

    kraw = pd.read_parquet(PROJECT_ROOT / "data/raw/daily_kline.parquet",
                           columns=["stock_code", "trade_date"])
    kclean = pd.read_parquet(PROJECT_ROOT / "data/clean/daily_kline.parquet",
                             columns=["stock_code", "trade_date", "pct_chg"])
    check("clean.kline", "row conservation raw -> clean",
          len(kclean) == kraw.drop_duplicates(subset=["stock_code", "trade_date"]).shape[0],
          f"raw={len(kraw):,}, clean={len(kclean):,}")
    pc = kclean["pct_chg"].dropna()
    check("clean.kline", "pct_chg stays PERCENT scale (p99 in 5~35)",
          5 <= pc.abs().quantile(0.99) <= 35, f"p99={pc.abs().quantile(0.99):.2f}")

    det = pd.read_parquet(PROJECT_ROOT / "data/clean/lhb_broker_detail.parquet")
    unk = (det["direction"] == "unknown").sum()
    check("clean.detail", "flag -> direction mapping complete", unk == 0, f"unknown={unk}")
    d = det.dropna(subset=["buy_amount", "net_amount", "sell_amount"])
    samp = d.sample(min(len(d), 50_000), random_state=42)
    ok = np.allclose(samp["sell_amount"], samp["buy_amount"] - samp["net_amount"], atol=1.0)
    check("clean.detail", "sell_amount = buy_amount - net_amount", ok, f"checked {len(samp)}")
    rank_ok = det["rank"].between(1, 10).all()
    check("clean.detail", "rank within 1..10", rank_ok)


# ───────────────────────── 4. features / factors ─────────────────────────

def verify_factors() -> None:
    section("4. Feature & factor layer")
    ev = pd.read_parquet(PROJECT_ROOT / "data/factor/lhb_event_factors.parquet")
    clean = pd.read_parquet(PROJECT_ROOT / "data/clean/lhb_summary.parquet",
                            columns=["trade_date", "stock_code"])
    check("factors", "event rows == clean summary rows", len(ev) == len(clean),
          f"events={len(ev):,}, clean={len(clean):,}")

    kline_cov = ev["close"].notna().mean()
    check("factors", "kline join coverage > 97%", kline_cov > 0.97, f"{kline_cov:.2%}")

    d = ev.dropna(subset=["lhb_buy_total", "lhb_sell_total", "buy_sell_ratio"])
    d = d[d["lhb_sell_total"] > 0]
    samp = d.sample(min(len(d), 20_000), random_state=42)
    ok = np.allclose(samp["buy_sell_ratio"],
                     samp["lhb_buy_total"] / samp["lhb_sell_total"], rtol=1e-6)
    check("factors", "buy_sell_ratio recompute", ok, f"checked {len(samp)}")

    buy_cols = [f"buy{i}_amount" for i in range(1, 6)]
    d = ev.dropna(subset=["buy1_amount", "buy1_concentration"])
    samp = d.sample(min(len(d), 10_000), random_state=42)
    expected = samp["buy1_amount"] / samp[buy_cols].sum(axis=1, skipna=True)
    ok = np.allclose(samp["buy1_concentration"], expected, rtol=1e-6)
    check("factors", "buy1_concentration (seat-internal) recompute", ok, f"checked {len(samp)}")
    check("factors", "buy1_concentration within (0, 1]",
          samp["buy1_concentration"].between(0, 1 + EPS).all())

    # lhb_window_days: reasons mentioning 三个交易日 must be 3
    has3 = ev["summary_selected_reason"].astype(str).str.contains("连续三个交易日", na=False)
    w = ev.loc[has3, "lhb_window_days"]
    check("factors", "3-day-board reasons -> window=3", (w == 3).all(),
          f"{(w != 3).sum()} mismatches of {len(w)}")
    # and their cross-window factors must be masked
    masked = ev.loc[has3, "buy1_to_daily_amount_ratio"].notna().sum()
    check("factors", "cross-window factors masked for 3-day boards", masked == 0,
          f"{masked} not masked")

    # inst_buy_count recompute from broker names
    broker_cols = [f"buy{i}_broker" for i in range(1, 6) if f"buy{i}_broker" in ev.columns]
    d = ev[ev["inst_buy_count"].notna()]
    samp = d.sample(min(len(d), 5_000), random_state=42)
    expected = sum(
        samp[c].astype(str).str.contains("机构专用", na=False).astype(float)
        for c in broker_cols
    )
    ok = np.allclose(samp["inst_buy_count"], expected)
    check("factors", "inst_buy_count recompute from names", ok, f"checked {len(samp)}")
    no_detail = ev[ev["buy1_broker"].isna() | (ev["buy1_broker"].astype(str) == "")]
    check("factors", "inst_buy_count NaN when no broker detail",
          no_detail["inst_buy_count"].isna().all(),
          f"{no_detail['inst_buy_count'].notna().sum()} false zeros")


# ───────────────────────── 5. labels ─────────────────────────

def verify_labels() -> None:
    section("5. Label layer (independent recompute on random events)")
    lab = pd.read_parquet(PROJECT_ROOT / "data/factor/lhb_event_labeled.parquet")
    kline = pd.read_parquet(
        PROJECT_ROOT / "data/clean/daily_kline.parquet",
        columns=["stock_code", "trade_date", "open", "close", "high", "low",
                 "high_limit", "pct_chg"],
    )

    # Build independent per-stock date-indexed lookup for sampled stocks
    samp_events = lab.dropna(subset=["future_return_1d"]).sample(300, random_state=42)
    stocks = set(samp_events["stock_code"])
    k = kline[kline["stock_code"].isin(stocks)].sort_values(["stock_code", "trade_date"])
    k_by_stock = {c: g.reset_index(drop=True) for c, g in k.groupby("stock_code")}

    mism = {"fr1": 0, "fr5": 0, "entry1": 0, "oo1": 0, "gap": 0, "ndlu": 0, "n": 0}
    for _, e in samp_events.iterrows():
        g = k_by_stock.get(e["stock_code"])
        if g is None:
            continue
        idx = g.index[g["trade_date"] == e["trade_date"]]
        if len(idx) == 0:
            continue
        i = idx[0]
        if i + 2 >= len(g):
            continue
        mism["n"] += 1
        c0, c1, c5 = g.loc[i, "close"], g.loc[i + 1, "close"], None
        o1, o2 = g.loc[i + 1, "open"], g.loc[i + 2, "open"]
        if i + 5 < len(g):
            c5 = g.loc[i + 5, "close"]

        if abs(e["future_return_1d"] - (c1 / c0 - 1)) > 1e-6:
            mism["fr1"] += 1
        if c5 is not None and pd.notna(e.get("future_return_5d")):
            if abs(e["future_return_5d"] - (c5 / c0 - 1)) > 1e-6:
                mism["fr5"] += 1
        if pd.notna(e.get("entry_open_return_1d")):
            if abs(e["entry_open_return_1d"] - (c1 / o1 - 1)) > 1e-6:
                mism["entry1"] += 1
        if pd.notna(e.get("entry_open_exit_open_1d")):
            if abs(e["entry_open_exit_open_1d"] - (o2 / o1 - 1)) > 1e-6:
                mism["oo1"] += 1
        if pd.notna(e.get("entry_open_gap")):
            if abs(e["entry_open_gap"] - (o1 / c0 - 1)) > 1e-6:
                mism["gap"] += 1
        hl1 = g.loc[i + 1, "high_limit"]
        if pd.notna(hl1) and pd.notna(e.get("next_day_limit_up")):
            expect = float(c1 >= hl1 - 1e-4)
            if e["next_day_limit_up"] != expect:
                mism["ndlu"] += 1

    n = mism.pop("n")
    check("labels", f"independent recompute on {n} events", n >= 250, f"n={n}")
    for key, label in (("fr1", "future_return_1d"), ("fr5", "future_return_5d"),
                       ("entry1", "entry_open_return_1d"),
                       ("oo1", "entry_open_exit_open_1d (legal T+1)"),
                       ("gap", "entry_open_gap"), ("ndlu", "next_day_limit_up (exact price)")):
        check("labels", f"{label} matches", mism[key] == 0, f"mismatches={mism[key]}/{n}")

    # No look-ahead at the data edge: events on the last kline date must have NaN labels
    kmax = kline["trade_date"].max()
    edge = lab[lab["trade_date"] == kmax]
    if len(edge):
        check("labels", f"edge events ({kmax}) have NaN future_return_1d",
              edge["future_return_1d"].isna().all(),
              f"{edge['future_return_1d'].notna().sum()} non-NaN")

    # alpha = future_return - benchmark_return (recompute via benchmark table)
    from src.data_sources.simtradedata_metadata import load_benchmark
    bench = load_benchmark()
    if bench is not None and not bench.empty and "alpha_5d" in lab.columns:
        b = bench.sort_values("trade_date").reset_index(drop=True)
        b["bret5"] = b["close"].shift(-5) / b["close"] - 1
        m = lab.dropna(subset=["alpha_5d", "future_return_5d"]).sample(2000, random_state=42)
        m = m.merge(b[["trade_date", "bret5"]], on="trade_date", how="left").dropna(subset=["bret5"])
        ok = np.allclose(m["alpha_5d"], m["future_return_5d"] - m["bret5"], atol=1e-8)
        check("labels", "alpha_5d = future_return_5d - benchmark_5d", ok, f"checked {len(m)}")

    # untradable semantics: next_day_open_at_limit=1 implies open_1 >= high_limit_1
    st = lab[(lab["next_day_open_at_limit"] == 1.0)].sample(
        min(500, int((lab["next_day_open_at_limit"] == 1.0).sum())), random_state=42)
    viol = 0
    for _, e in st.iterrows():
        g = k_by_stock.get(e["stock_code"])
        if g is None:
            continue
        idx = g.index[g["trade_date"] == e["trade_date"]]
        if len(idx) == 0 or idx[0] + 1 >= len(g):
            continue
        o1, hl1 = g.loc[idx[0] + 1, "open"], g.loc[idx[0] + 1, "high_limit"]
        if pd.notna(o1) and pd.notna(hl1) and o1 < hl1 - 1e-4:
            viol += 1
    check("labels", "next_day_open_at_limit implies open>=limit", viol == 0,
          f"violations={viol} (sampled)")


# ───────────────────────── 6. filter engine ─────────────────────────

def verify_filters() -> None:
    section("6. Filter engine (full-set semantic checks)")
    from src.backtest.filter_engine import apply_filters

    df = pd.read_parquet(PROJECT_ROOT / "data/factor/lhb_event_labeled.parquet")

    f = apply_filters(df, {"exclude_ST": True})
    check("filters", "exclude_ST keeps no is_st==1", (f["is_st"] != 1.0).all(),
          f"kept {len(f):,}/{len(df):,}")

    f = apply_filters(df, {"include_only_limit_up_event": True})
    check("filters", "include_only_limit_up keeps only event_day_limit_up==1",
          (f["event_day_limit_up"] == 1.0).all(), f"kept {len(f):,}")

    f = apply_filters(df, {"only_single_day_board": True})
    check("filters", "only_single_day_board keeps window==1",
          (f["lhb_window_days"] == 1).all(), f"kept {len(f):,}")

    f = apply_filters(df, {"exclude_untradable_next_day": True})
    check("filters", "exclude_untradable keeps no untradable==1",
          (f["next_day_untradable"] != 1.0).all(), f"kept {len(f):,}")

    f = apply_filters(df, {"exclude_lhb_reason": "无价格涨跌幅限制"})
    has = f["lhb_reason"].astype(str).str.contains("无价格涨跌幅限制", na=False).sum()
    check("filters", "exclude_lhb_reason removes all matches", has == 0,
          f"kept {len(f):,}, still-matching={has}")

    # exclude_consecutive_lhb_day: no kept event may have (stock, prev trading
    # day) present in the ORIGINAL event universe
    from src.utils.calendar import get_trading_dates
    f = apply_filters(df, {"exclude_consecutive_lhb_day": True})
    all_days = get_trading_dates("2018-12-01", str(df["trade_date"].max()))
    prev_map = {d: (all_days[i - 1] if i > 0 else "") for i, d in enumerate(all_days)}
    universe = set(df["stock_code"].astype(str) + "|" + df["trade_date"].astype(str))
    prev_keys = f["stock_code"].astype(str) + "|" + f["trade_date"].astype(str).map(prev_map).fillna("")
    consec = prev_keys.isin(universe).sum()
    check("filters", "exclude_consecutive: kept events never follow a listed day",
          consec == 0, f"violations={consec} of {len(f):,} kept")
    # regression guard for the old positional-index bug: late dates must survive
    check("filters", "exclude_consecutive keeps recent events (index-bug guard)",
          str(f["trade_date"].max()) >= "2026-05-01", f"max kept date={f['trade_date'].max()}")

    f = apply_filters(df, {"include_only_first_board": True})
    check("filters", "first_board keeps is_first_limit_up_board==1",
          (f["is_first_limit_up_board"] == 1.0).all(), f"kept {len(f):,}")


# ───────────────────────── 7. statistics / grid search ─────────────────────────

def verify_statistics() -> None:
    section("7. Statistics & grid search")
    from src.backtest.filter_engine import apply_filters
    from src.backtest.statistics import compute_backtest_statistics

    df = pd.read_parquet(PROJECT_ROOT / "data/factor/lhb_event_labeled.parquet")
    sub = apply_filters(df, {"exclude_ST": True, "include_only_limit_up_event": True,
                             "exclude_untradable_next_day": True})
    stats = compute_backtest_statistics(sub, horizons=[1, 5], round_trip_cost=0.002).iloc[0]

    r1 = sub["future_return_1d"].dropna()
    check("stats", "sample_count matches", stats["sample_count"] == len(sub),
          f"{stats['sample_count']} vs {len(sub)}")
    check("stats", "avg_return_1d recompute",
          abs(stats["avg_return_1d"] - r1.mean()) < 1e-9)
    check("stats", "win_rate_1d recompute",
          abs(stats["win_rate_1d"] - (r1 > 0).mean()) < 1e-9)

    e1 = sub["entry_open_return_1d"].dropna()
    check("stats", "research_intraday_return_1d = mean(entry), not executable",
          abs(stats["research_intraday_return_1d"] - e1.mean()) < 1e-9)

    oo = sub["entry_open_exit_open_1d"].dropna()
    check("stats", "net_oo_return_1d (legal T+1 exit) recompute",
          abs(stats["net_oo_return_1d"] - (oo - 0.002).mean()) < 1e-9)

    gains = r1[r1 > 0].mean()
    losses = r1[r1 < 0].mean()
    if pd.notna(gains) and pd.notna(losses) and losses != 0:
        expected_pl = abs(gains / losses)
        check("stats", "profit_loss_ratio_1d recompute",
              abs(stats.get("profit_loss_ratio_1d", np.nan) - expected_pl) < 1e-6)

    # grid search: tiny grid, verify combo count + IS/OOS sample split
    from src.backtest.grid_search import grid_search
    res = grid_search(
        param_grid={"min_net_buy_ratio": [0.0, 0.05]},
        event_df=sub,
        horizons=[1],
        oos_split_date="2024-07-01",
        min_sample_count=30,
        round_trip_cost=0.002,
        generate_plots=False,
        save_path=None,
    )
    check("grid", "combo count == grid product", len(res) == 2, f"rows={len(res)}")
    row = res[res["min_net_buy_ratio"] == 0.05].iloc[0]
    manual = sub[(sub["net_buy_ratio"] >= 0.05)
                 & (sub["trade_date"].astype(str) < "2024-07-01")
                 & (sub["label_end_date_2d"] < "2024-07-01")]
    check("grid", "IS sample_count matches manual filter",
          row["sample_count"] == len(manual),
          f"{row['sample_count']} vs {len(manual)}")
    has_oos = any(c.startswith("oos_") for c in res.columns)
    check("grid", "OOS columns present", has_oos)


# ───────────────────────── main ─────────────────────────

def main() -> int:
    verify_calendar()
    verify_raw()
    verify_clean()
    verify_factors()
    verify_labels()
    verify_filters()
    verify_statistics()

    fails = [(n, d) for n, ok, d in RESULTS if not ok]
    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(RESULTS)} checks, {len(RESULTS) - len(fails)} passed, {len(fails)} failed")
    for name, detail in fails:
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
