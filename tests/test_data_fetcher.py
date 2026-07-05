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
    assert len(df) > 0


@needs_network
def test_live_quote(fetcher):
    df = fetcher.get_live_quote(["RELIANCE.NS"])
    assert len(df) == 1
    assert df.iloc[0]["last_price"] is not None
