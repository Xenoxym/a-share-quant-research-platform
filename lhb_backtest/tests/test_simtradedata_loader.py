"""Unit tests for SimTradeData loader and related normalisation code.

All tests are offline (no network, no real parquet files).
Mock data is written to a temporary directory via pytest's tmp_path fixture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_stock_parquet(directory: Path, filename: str, rows: list[dict]) -> Path:
    """Write a minimal stocks/ parquet file for testing."""
    df = pd.DataFrame(rows)
    path = directory / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def _write_valuation_parquet(directory: Path, filename: str, rows: list[dict]) -> Path:
    """Write a minimal valuation/ parquet file for testing."""
    df = pd.DataFrame(rows)
    path = directory / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def export_cn_dir(tmp_path: Path) -> Path:
    """Create a minimal SimTradeData export/cn/ layout for testing."""
    stocks_dir = tmp_path / "stocks"
    stocks_dir.mkdir()
    val_dir = tmp_path / "valuation"
    val_dir.mkdir()

    # SZ stock (000001.SZ) — 3 trading days
    _write_stock_parquet(stocks_dir, "000001.SZ.parquet", [
        {"date": "2024-01-02", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2,
         "preclose": 10.0, "volume": 1_000_000, "money": 10_200_000,
         "high_limit": 11.0, "low_limit": 9.0},
        {"date": "2024-01-03", "open": 10.2, "high": 10.8, "low": 10.0, "close": 10.6,
         "preclose": 10.2, "volume": 1_200_000, "money": 12_600_000,
         "high_limit": 11.22, "low_limit": 9.18},
        {"date": "2024-01-04", "open": 10.6, "high": 11.0, "low": 10.4, "close": 10.9,
         "preclose": 10.6, "volume": 900_000, "money": 9_810_000,
         "high_limit": 11.66, "low_limit": 9.54},
    ])

    # SS stock (Shanghai) — should become .SH after normalisation
    _write_stock_parquet(stocks_dir, "600519.SS.parquet", [
        {"date": "2024-01-02", "open": 1500.0, "high": 1520.0, "low": 1490.0, "close": 1510.0,
         "preclose": 1500.0, "volume": 50_000, "money": 75_500_000,
         "high_limit": 1650.0, "low_limit": 1350.0},
    ])

    # Valuation for 000001.SZ
    _write_valuation_parquet(val_dir, "000001.SZ.parquet", [
        {"date": "2024-01-02", "turnover_rate": 0.82},
        {"date": "2024-01-03", "turnover_rate": 0.95},
        {"date": "2024-01-04", "turnover_rate": 0.71},
    ])

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: load_stocks_parquet
# ---------------------------------------------------------------------------

class TestLoadStocksParquet:
    def test_basic_load_returns_expected_columns(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-01", "2024-12-31", include_valuation=False)

        required = {"trade_date", "stock_code", "open", "high", "low", "close",
                    "pre_close", "pct_chg", "volume", "amount", "data_source"}
        assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    def test_field_rename_applied(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-01", "2024-12-31", include_valuation=False)

        # SimTradeData names must NOT appear in output
        assert "date" not in df.columns
        assert "preclose" not in df.columns
        assert "money" not in df.columns

        # Renamed names must appear
        assert "trade_date" in df.columns
        assert "pre_close" in df.columns
        assert "amount" in df.columns

    def test_data_source_tag(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-01", "2024-12-31", include_valuation=False)
        assert (df["data_source"] == "simtradedata").all()

    def test_date_filter_applied(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-03", "2024-01-03", include_valuation=False)
        assert set(df["trade_date"].unique()) == {"2024-01-03"}

    def test_pct_chg_computed_correctly(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-02", "2024-01-02", include_valuation=False)
        row = df[df["stock_code"] == "000001.SZ"].iloc[0]
        # (10.2 / 10.0 - 1) * 100 = 2.0
        assert abs(row["pct_chg"] - 2.0) < 1e-6

    def test_high_low_limit_present(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-01", "2024-12-31", include_valuation=False)
        assert "high_limit" in df.columns
        assert "low_limit" in df.columns

    def test_stock_count(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-01", "2024-12-31", include_valuation=False)
        assert df["stock_code"].nunique() == 2

    def test_empty_result_for_out_of_range_date(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2025-01-01", "2025-12-31", include_valuation=False)
        assert df.empty

    def test_missing_stocks_dir_raises(self, tmp_path: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        with pytest.raises(FileNotFoundError, match="stocks"):
            load_stocks_parquet(tmp_path / "nonexistent", "2024-01-01", "2024-12-31")

    def test_valuation_join_adds_turnover_rate(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-02", "2024-01-02", include_valuation=True)
        row = df[df["stock_code"] == "000001.SZ"].iloc[0]
        assert "turnover_rate" in df.columns
        assert abs(row["turnover_rate"] - 0.82) < 1e-6

    def test_valuation_join_null_for_missing_stock(self, export_cn_dir: Path):
        """600519.SS has no valuation file — turnover_rate should be NaN."""
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-02", "2024-01-02", include_valuation=True)
        # 600519.SS has no valuation parquet → NaN
        row = df[df["stock_code"] == "600519.SS"].iloc[0]
        assert pd.isna(row["turnover_rate"])

    def test_valuation_skipped_when_disabled(self, export_cn_dir: Path):
        from src.data_sources.simtradedata_loader import load_stocks_parquet

        df = load_stocks_parquet(export_cn_dir, "2024-01-02", "2024-01-02", include_valuation=False)
        # turnover_rate may be null/NA but column must exist
        assert "turnover_rate" in df.columns


# ---------------------------------------------------------------------------
# Tests: normalize_stock_code (.SS → .SH)
# ---------------------------------------------------------------------------

class TestNormalizeStockCode:
    def test_ss_suffix_becomes_sh(self):
        from src.cleaning.normalize_codes import normalize_stock_code
        assert normalize_stock_code("600519.SS") == "600519.SH"

    def test_sz_unchanged(self):
        from src.cleaning.normalize_codes import normalize_stock_code
        assert normalize_stock_code("000001.SZ") == "000001.SZ"

    def test_sh_unchanged(self):
        from src.cleaning.normalize_codes import normalize_stock_code
        assert normalize_stock_code("600000.SH") == "600000.SH"

    def test_bj_unchanged(self):
        from src.cleaning.normalize_codes import normalize_stock_code
        assert normalize_stock_code("833XXX.BJ") == "833XXX.BJ"

    def test_plain_code_inferred(self):
        from src.cleaning.normalize_codes import normalize_stock_code
        assert normalize_stock_code("000001") == "000001.SZ"
        assert normalize_stock_code("600000") == "600000.SH"

    @pytest.mark.parametrize("raw,expected", [
        ("600519.SS", "600519.SH"),
        ("688001.SS", "688001.SH"),
        ("000001.SZ", "000001.SZ"),
        ("600000.SH", "600000.SH"),
        ("sh.600000", "600000.SH"),
        ("sz.000001", "000001.SZ"),
    ])
    def test_parametrize_normalization(self, raw, expected):
        from src.cleaning.normalize_codes import normalize_stock_code
        assert normalize_stock_code(raw) == expected


# ---------------------------------------------------------------------------
# Tests: stock-only universe filtering
# ---------------------------------------------------------------------------

class TestAshareStockFilter:
    @pytest.mark.parametrize(
        "code",
        ["600519.SS", "688001.SS", "000001.SZ", "300750.SZ", "920001.BJ"],
    )
    def test_accepts_a_share_stock_codes(self, code):
        from src.data_sources.simtradedata_loader import _is_a_share_stock_code

        assert _is_a_share_stock_code(code)

    @pytest.mark.parametrize(
        "code",
        ["510300.SS", "159919.SZ", "160105.SZ", "000300.SS", "600519.SH"],
    )
    def test_rejects_funds_indices_and_wrong_suffix(self, code):
        from src.data_sources.simtradedata_loader import _is_a_share_stock_code

        assert not _is_a_share_stock_code(code)


# ---------------------------------------------------------------------------
# Tests: _compute_pct_chg edge cases
# ---------------------------------------------------------------------------

def test_mixed_timestamp_precisions_are_imported(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.data_sources.simtradedata_loader import load_stocks_parquet
    (tmp_path/'stocks').mkdir()
    (tmp_path/'valuation').mkdir()
    for code,unit in [('000001.SZ','us'),('600000.SS','ns')]:
        dates=pa.array([pd.Timestamp('2026-09-22')],type=pa.timestamp(unit))
        pq.write_table(pa.table({'date':dates,'close':[11.0],'preclose':[10.0]}),tmp_path/'stocks'/f'{code}.parquet')
        pq.write_table(pa.table({'date':dates,'turnover_rate':[2.5]}),tmp_path/'valuation'/f'{code}.parquet')
    result=load_stocks_parquet(tmp_path,'2026-09-22','2026-09-22')
    assert len(result)==2 and result.turnover_rate.eq(2.5).all()
    assert result.trade_date.eq('2026-09-22').all()


def test_unreadable_vendor_file_aborts_import(tmp_path):
    from src.data_sources.simtradedata_loader import load_stocks_parquet
    (tmp_path/'stocks').mkdir()
    (tmp_path/'stocks/000001.SZ.parquet').write_bytes(b'broken')
    with pytest.raises(RuntimeError,match='Cannot safely import stock'):
        load_stocks_parquet(tmp_path,'2026-09-22','2026-09-22')


class TestComputePctChg:
    def test_zero_preclose_yields_nan(self):
        from src.data_sources.simtradedata_loader import _compute_pct_chg

        df = pd.DataFrame({"close": [10.0, 5.0], "pre_close": [0.0, 10.0]})
        result = _compute_pct_chg(df)
        assert pd.isna(result.iloc[0])
        assert abs(result.iloc[1] - (-50.0)) < 1e-6

    def test_normal_case(self):
        from src.data_sources.simtradedata_loader import _compute_pct_chg

        df = pd.DataFrame({"close": [11.0], "pre_close": [10.0]})
        result = _compute_pct_chg(df)
        assert abs(result.iloc[0] - 10.0) < 1e-6

    def test_nan_propagation(self):
        from src.data_sources.simtradedata_loader import _compute_pct_chg

        df = pd.DataFrame({"close": [float("nan")], "pre_close": [10.0]})
        result = _compute_pct_chg(df)
        assert pd.isna(result.iloc[0])
