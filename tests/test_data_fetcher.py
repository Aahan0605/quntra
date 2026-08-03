"""
Tests for UnifiedDataFetcher (Task 1-2).

Unit tests run offline (validator + routing logic). Integration tests
that hit NSE endpoints auto-skip when the network is unavailable
(e.g. inside the build sandbox) but run on a normal machine.
"""
import socket

import numpy as np
import pandas as pd
import pytest

from src.utils.data_fetcher import UnifiedDataFetcher, DataQualityReport


def _network_available(host="www.nseindia.com", port=443, timeout=3) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


NETWORK = _network_available()
needs_network = pytest.mark.skipif(not NETWORK, reason="no market-data network access")


def _clear_broker_caches():
    # class-level cache — real credentials in config/secrets.env would
    # otherwise leak a live client across unrelated tests
    UnifiedDataFetcher._breeze = None
    UnifiedDataFetcher._breeze_tried = False
    UnifiedDataFetcher._breeze_symbol_map = None


@pytest.fixture(autouse=True)
def _reset_broker_caches():
    _clear_broker_caches()
    yield
    _clear_broker_caches()


@pytest.fixture
def fetcher():
    return UnifiedDataFetcher()


@pytest.fixture
def clean_df():
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=30)
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, 30))
    return pd.DataFrame({
        "open": close * 0.999, "high": close * 1.01,
        "low": close * 0.99, "close": close,
        "volume": 1_000_000,
    }, index=idx)


# ----------------------------------------------------------------------- #
# validate_data (offline)

def test_validate_clean_dataframe_passes(fetcher, clean_df):
    report = fetcher.validate_data(clean_df)
    assert isinstance(report, DataQualityReport)
    assert report.passed, report.issues
    assert report.n_rows == 30


def test_validate_nan_close_fails(fetcher, clean_df):
    df = clean_df.copy()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    report = fetcher.validate_data(df)
    assert not report.passed
    assert any("NaN" in i for i in report.issues)


def test_validate_empty_fails(fetcher):
    report = fetcher.validate_data(pd.DataFrame())
    assert not report.passed
    assert report.n_rows == 0


def test_validate_negative_price_fails(fetcher, clean_df):
    df = clean_df.copy()
    df.iloc[3, df.columns.get_loc("close")] = -5.0
    report = fetcher.validate_data(df)
    assert not report.passed
    assert any("non-positive" in i for i in report.issues)


def test_validate_stale_data_fails(fetcher):
    idx = pd.bdate_range(end=pd.Timestamp.now() - pd.Timedelta(days=40), periods=30)
    df = pd.DataFrame({"close": np.linspace(100, 110, 30)}, index=idx)
    report = fetcher.validate_data(df)
    assert not report.passed
    assert any("stale" in i for i in report.issues)


def test_validate_bad_tick_flagged(fetcher, clean_df):
    df = clean_df.copy()
    df.iloc[10, df.columns.get_loc("close")] *= 2.0  # +100% jump
    report = fetcher.validate_data(df)
    assert not report.passed
    assert any("suspicious" in i for i in report.issues)


def test_validate_gap_detection(fetcher):
    idx = pd.DatetimeIndex(
        list(pd.bdate_range("2026-05-01", periods=10))
        + list(pd.bdate_range("2026-06-15", periods=10))
    )
    df = pd.DataFrame({"close": np.linspace(100, 105, 20)}, index=idx)
    report = fetcher.validate_data(df, max_stale_days=10_000)
    assert not report.passed
    assert any("gaps" in i for i in report.issues)


# ----------------------------------------------------------------------- #
# Integration (require market-data network)

@needs_network
def test_fetch_30d_reliance_ohlc(fetcher):
    from datetime import date, timedelta
    end = date.today()
    df = fetcher.get_historical_ohlc("RELIANCE.NS", end - timedelta(days=45), end)
    assert len(df) >= 25
    assert not df["close"].isna().any()
    report = fetcher.validate_data(df)
    assert report.passed, report.issues


@needs_network
def test_fetch_nifty_options_chain(fetcher):
    df = fetcher.get_options_chain("NIFTY")
    if len(df) == 0:
        pytest.skip("NSE blocking options-chain endpoint (403/503) — "
                    "expected outside market hours")
    assert len(df) > 0


@needs_network
def test_live_quote(fetcher):
    df = fetcher.get_live_quote(["RELIANCE.NS"])
    assert len(df) == 1
    assert df.iloc[0]["last_price"] is not None


# ----------------------------------------------------------------------- #
# Breeze real-time quotes: symbol-map join + routing priority (all offline)

def test_breeze_symbol_map_joins_on_last_column(fetcher, monkeypatch, tmp_path):
    """The security master's last column is the plain NSE symbol — proven
    by inspection (RELIANCE/TCS/INFY/HDFCBANK all matched their known
    Breeze stock_code), not derivable by string transforms on the ticker."""
    cache = tmp_path / "icici_nse_scripmaster.csv"
    pd.DataFrame({"symbol": ["RELIANCE", "TCS"],
                 "stock_code": ["RELIND", "TCS"]}).to_csv(cache, index=False)
    monkeypatch.setattr("src.utils.data_fetcher._ROOT", tmp_path.parent)
    (tmp_path.parent / "data" / "cache").mkdir(parents=True, exist_ok=True)
    cache.rename(tmp_path.parent / "data" / "cache" / "icici_nse_scripmaster.csv")

    UnifiedDataFetcher._breeze_symbol_map = None
    mapping = fetcher._load_breeze_symbol_map()
    assert mapping["RELIANCE"] == "RELIND"
    assert mapping["TCS"] == "TCS"
    UnifiedDataFetcher._breeze_symbol_map = None


def test_breeze_quotes_use_the_symbol_map(fetcher, monkeypatch):
    class FakeBreeze:
        def get_quotes(self, stock_code, exchange_code, product_type):
            assert stock_code == "RELIND"
            return {"Success": [{"exchange_code": "NSE", "ltp": 1234.5,
                                 "previous_close": 1200.0, "open": 1210.0,
                                 "high": 1240.0, "low": 1205.0,
                                 "ltp_percent_change": 2.87,
                                 "ltt": "29-Jul-2026 10:00:00"}]}

    monkeypatch.setattr(fetcher, "_get_breeze", lambda: FakeBreeze())
    monkeypatch.setattr(fetcher, "_load_breeze_symbol_map",
                       lambda: {"RELIANCE": "RELIND"})
    out = fetcher._breeze_quotes(["RELIANCE.NS"])
    assert out["RELIANCE.NS"]["last_price"] == 1234.5
    assert out["RELIANCE.NS"]["source"] == "breeze_realtime"


def test_breeze_quotes_skip_unmapped_tickers(fetcher, monkeypatch):
    monkeypatch.setattr(fetcher, "_get_breeze", lambda: object())
    monkeypatch.setattr(fetcher, "_load_breeze_symbol_map", lambda: {})
    assert fetcher._breeze_quotes(["UNKNOWN.NS"]) == {}


def test_breeze_quotes_empty_when_breeze_unavailable(fetcher, monkeypatch):
    monkeypatch.setattr(fetcher, "_get_breeze", lambda: None)
    assert fetcher._breeze_quotes(["RELIANCE.NS"]) == {}


def test_live_quote_prefers_breeze_over_yfinance(fetcher, monkeypatch):
    monkeypatch.setattr(fetcher, "_breeze_quotes",
                       lambda tickers: {"RELIANCE.NS": {
                           "ticker": "RELIANCE.NS", "last_price": 111.0,
                           "close": 111.0, "open": None, "day_high": None,
                           "day_low": None, "prev_close": None,
                           "change_pct": None, "timestamp": None,
                           "source": "breeze_realtime"}})
    monkeypatch.setattr(fetcher, "_yf_quote",
                       lambda t: pytest.fail(
                           "yfinance should not be queried for a ticker "
                           "Breeze already served"))
    df = fetcher.get_live_quote(["RELIANCE.NS"])
    assert df.iloc[0]["source"] == "breeze_realtime"


def test_live_quote_falls_through_to_yfinance_when_breeze_has_nothing(
        fetcher, monkeypatch):
    monkeypatch.setattr(fetcher, "_breeze_quotes", lambda tickers: {})
    monkeypatch.setattr(fetcher, "_yf_quote",
                       lambda t: {"ticker": t, "last_price": 222.0,
                                 "close": 222.0, "open": None,
                                 "day_high": None, "day_low": None,
                                 "prev_close": None, "change_pct": None,
                                 "timestamp": None, "source": "yfinance_delayed"})
    df = fetcher.get_live_quote(["RELIANCE.NS"])
    assert df.iloc[0]["source"] == "yfinance_delayed"
