"""Named economic hypotheses with explicit raw fields, timing and comparison roles."""
import hashlib
import numpy as np
import pandas as pd

from .foundation_data import attach_reports
from .signals import KEYS, decision_dates

SOURCES = {
    "size": "https://www.nber.org/papers/w24458",
    "large_size": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_port_form_sz.html",
    "earnings_yield": "https://www.nber.org/papers/w24458",
    "book_to_price": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    "profitability": "https://www.aqr.com/Insights/Research/Working-Paper/Size-Matters-If-You-Control-Your-Junk",
    "low_volatility": "https://www.nber.org/papers/w10852",
    "market_trend": "https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum",
    "price": "https://www.aqr.com/Insights/Datasets/Momentum-Indices-Monthly",
}


def definition(spec):
    s = spec.strategy
    descriptions = {
        "size": ("小市值股票是否有可交易的规模溢价？", "原始收盘价 × 最近已公布财报的总股本", "估算总市值越小，排名越前", "score = −log(收盘价 × 已披露总股本)"),
        "large_size": ("大市值对照：规模另一端的收益与风险如何？", "原始收盘价 × 最近已公布财报的总股本", "估算总市值越大，排名越前", "score = log(收盘价 × 已披露总股本)"),
        "earnings_yield": ("盈利相对价格更便宜的股票，是否表现更好？", "最新已公布年度归母净利润、估算总市值", "正盈利收益率越高，排名越前", "score = 已公布年度归母净利润 / 估算总市值；只取正值"),
        "book_to_price": ("账面权益相对价格更便宜的股票，是否表现更好？", "最近已公布股东权益合计、估算总市值", "账面权益／市值越高，排名越前", "score = 已公布股东权益合计 / 估算总市值；权益须为正"),
        "profitability": ("盈利相对账面资本更高的股票，是否具有持续优势？", "最新已公布年度归母净利润、同年股东权益合计", "正年度利润／权益越高，排名越前", "score = 年度归母净利润 / 同年股东权益合计；分子分母须为正"),
        "low_volatility": ("低波动选股能否改善含费风险收益？", f"最近 {s.lookback} 日 log(收盘价／参考前收价)", "收益波动越低，排名越前", f"score = −std(日对数收益, {s.lookback}, ddof=1) × sqrt(252)"),
        "random": ("没有选股信息时，同一约束下会发生什么？", f"股票代码、事前固定种子 {s.seed}", "稳定哈希排序；不看历史或未来收益", "score = SHA256(seed + ':' + 股票代码)前13位整数 / 16^13"),
        "market_trend": ("在固定随机底仓上，市场趋势择时是否值得？", f"同种子股票底仓、市场价格参考的 {s.trend_window} 日均线", "参考收盘高于均线时持有，否则目标全部为现金", f"gate = 市场参考收盘 > SMA{ s.trend_window }；底仓按固定种子哈希排序"),
        "cash": ("不承担股票风险的零利息现金对照", "初始现金", "所有日期股票目标为零", "所有股票 target_weight = 0；现金利息 = 0"),
        "price": ("相对强势延续还是短期下跌反弹？", f"最近 {s.lookback} 日、跳过 {s.skip} 日的参考价收益", "按动量或反转分数排名", "M = product(C / P_ref, j=skip..lookback−1) − 1；score = M（动量）或 −M（反转）"),
    }
    idea, raw, feature, formula = descriptions[s.family]
    bridge = ""
    if s.share_basis == "known_bonus":
        bridge = " 股本桥接：报告期末股数×连乘(1+报告期后且截至决策日已发生的送转比例)。未被新季报覆盖的配股记录排除；仍不覆盖全部增发、回购或转股。"
        raw += "；股本桥接已发生送转"
        formula += "；估算股本 = 季报股数 × 已发生送转乘数，市值改用此股本"
    return {"idea": idea, "raw_data": raw, "feature_description": feature, "formula": formula,
            "signal": formula, "score": feature,
            "filters": f"共同资格：非 ST、已上市、有效限价、至少 {max(20,s.lookback,s.trend_window,s.min_history)} 个连续市场日；20 日均成交额≥{s.min_avg_amount:g}。估算市值和财报有效。规模分位须>{s.size_floor_quantile:g}。" + ("另要求已公布年度归母净利润为正。" if s.positive_profit else ""),
            "availability": "仅使用公告日期严格早于决策日的财报；日线假设收盘后可用，下一交易日开始执行。财报源无历史修订档案。" + bridge,
            "ranking": f"同一资格池内将 score 保留12位小数，按分数降序、代码升序取前 {s.top_k}；均分 {s.allocation:.0%} 目标资金，不足数量留现金。",
            "parameter_origin": "文献启发的 A 股主板多头原型，不是论文多空因子的原样复现。总市值是已披露股本估计；年度盈利不是 TTM；利润／权益不是完整 QMJ。",
            "source": "https://www.nber.org/papers/w30917" if s.family == "price" and s.direction == "reversal" else SOURCES.get(s.family, ""), "family": s.family,
            "evaluation": "按公开来源与预登记假设探索；已查看的历史不称未见样本外。随机与大市值是对照，不是待推荐的冠军。"}


def decide(bars, calendar, metadata, spec, filings, benchmark, actions=None):
    s = spec.strategy
    schedule = decision_dates(calendar, spec.experiment.start_date, spec.experiment.end_date, s.rebalance)
    rows = bars.loc[bars.trade_date.isin(schedule)].merge(metadata[["stock_code", "listed_date", "de_listed_date"]], on="stock_code", validate="many_to_one")
    rows = attach_reports(rows, filings, actions, s.share_basis == "known_bonus")
    annual_ok = rows.a_age_days.between(0, 550) & np.isfinite(rows.a_np_parent_company_owners)
    cap_ok = rows.q_age_days.between(0, 240) & np.isfinite(rows.estimated_market_cap) & rows.estimated_market_cap.gt(0)
    rows["reason"] = np.select([
        ~(rows.listed_date.le(rows.trade_date) & rows.de_listed_date.gt(rows.trade_date)),
        ~rows.is_st.eq(0), ~rows.limit_price_valid.eq(1), ~rows.history_valid,
        ~np.isfinite(rows.avg_amount_20) | rows.avg_amount_20.lt(s.min_avg_amount) | ~rows.avg_volume_20.gt(0),
        ~cap_ok, ~annual_ok if s.require_fundamentals else np.zeros(len(rows), bool)],
        ["not_listed_or_unknown", "st_or_unknown", "unknown_limits", "insufficient_history", "liquidity", "missing_or_stale_share_report", "missing_or_stale_annual_report"], default="eligible")
    eligible = rows.reason.eq("eligible")
    # Percentiles use the common eligible pool BEFORE family-specific filters.
    ordered = rows.loc[eligible].sort_values(["trade_date", "estimated_market_cap", "stock_code"])
    pct = (ordered.groupby("trade_date").cumcount() + 1) / ordered.groupby("trade_date").stock_code.transform("size")
    rows["size_percentile"] = pct.reindex(rows.index)
    rows.loc[eligible & rows.size_percentile.le(s.size_floor_quantile), "reason"] = "size_band"
    if s.positive_profit:
        rows.loc[rows.reason.eq("eligible") & ~(annual_ok & rows.a_np_parent_company_owners.gt(0)), "reason"] = "nonpositive_annual_profit"
    family = s.family
    if family in {"size", "large_size"}:
        rows["score"] = np.log(rows.estimated_market_cap.where(rows.estimated_market_cap.gt(0))) * (-1 if family == "size" else 1)
    elif family in {"earnings_yield", "book_to_price", "profitability"}:
        rows["score"] = rows[family]
        valid = rows.score.gt(0)
        if family in {"earnings_yield", "profitability"}:
            valid &= annual_ok & rows.a_np_parent_company_owners.gt(0)
        if family == "profitability":
            valid &= rows.a_total_shareholder_equity.gt(0)
        if family == "book_to_price":
            valid &= rows.q_total_shareholder_equity.gt(0)
        rows.loc[rows.reason.eq("eligible") & ~valid, "reason"] = "invalid_family_fundamental"
    elif family == "low_volatility":
        rows["score"] = -rows.volatility
    elif family in {"random", "market_trend", "cash"}:
        scores = {c: int(hashlib.sha256(f"{s.seed}:{c}".encode()).hexdigest()[:13], 16)/16**13 for c in rows.stock_code.unique()}
        rows["score"] = rows.stock_code.map(scores)
    if family == "market_trend":
        if benchmark is None:
            raise ValueError("市场趋势策略缺少参考指数历史")
        b = benchmark.sort_values("trade_date").copy()
        if b.trade_date.duplicated().any():
            raise ValueError("市场参考日期重复")
        b = b.set_index("trade_date").reindex(calendar)
        ma = b.close.rolling(s.trend_window, min_periods=s.trend_window).mean()
        gate = b.close.gt(ma) & ma.notna() & b.close.gt(0)
        rows["market_gate"] = rows.trade_date.map(gate).fillna(False)
        rows["market_reference_close"] = rows.trade_date.map(b.close)
        rows["market_reference_ma"] = rows.trade_date.map(ma)
        rows.loc[rows.reason.eq("eligible") & ~rows.market_gate, "reason"] = "market_trend_cash"
    elif s.trend_window and family == "price":
        rows.loc[rows.reason.eq("eligible") & rows.trend.le(0), "reason"] = "trend_filter"
    if family == "cash":
        rows["reason"] = "cash_control"
    rows.loc[rows.reason.eq("eligible") & ~np.isfinite(rows.score), "reason"] = "invalid_score"
    if s.min_score is not None:
        rows.loc[rows.reason.eq("eligible") & rows.score.le(s.min_score), "reason"] = "score_threshold"
    # Numerical precision is explicitly part of the new family's tie policy.
    rows["score"] = rows.score.round(12)
    rows = rows.sort_values(["trade_date", "score", "stock_code"], ascending=[True, False, True])
    eligible = rows.reason.eq("eligible")
    rows["rank"] = np.nan
    rows.loc[eligible, "rank"] = rows.loc[eligible].groupby("trade_date").cumcount() + 1
    rows["selected"] = rows["rank"].le(s.top_k)
    rows["target_weight"] = np.where(rows.selected, s.allocation / s.top_k, 0.)
    rows["execution_date"] = rows.trade_date.map(schedule)
    rows["known_at_assumed"] = rows.trade_date + "T15:30:00+08:00"
    cols = KEYS + ["known_at_assumed", "execution_date", "close", "momentum", "score", "trend", "avg_amount_20", "avg_volume_20", "reason", "rank", "selected", "target_weight", "estimated_market_cap", "size_percentile", "earnings_yield", "book_to_price", "profitability", "q_report_date", "q_publication_date", "q_total_shares", "q_age_days", "a_report_date", "a_publication_date", "a_np_parent_company_owners", "a_total_shareholder_equity", "a_age_days"]
    cols += [c for c in ("volatility", "market_gate", "market_reference_close", "market_reference_ma", "estimated_shares", "share_action_multiplier", "unresolved_rights_since_report") if c in rows]
    return rows[cols].reset_index(drop=True), schedule
