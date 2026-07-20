"""Kite real-time quote path in UnifiedDataFetcher — mocked, offline."""

from unittest.mock import MagicMock

import pytest

from src.utils.data_fetcher import UnifiedDataFetcher


@pytest.fixture(autouse=True)
def _reset_kite_cache():
    # class-level cache must not leak between tests
    UnifiedDataFetcher._kite = None
    UnifiedDataFetcher._kite_tried = False
    yield
    UnifiedDataFetcher._kite = None
    UnifiedDataFetcher._kite_tried = False


def _fake_kite(quote_payload):
    k = MagicMock()
    k.quote.return_value = quote_payload
    return k


def test_kite_symbol_mapping():
    assert UnifiedDataFetcher._kite_symbol("RELIANCE.NS") == "NSE:RELIANCE"
    assert UnifiedDataFetcher._kite_symbol("M&M.NS") == "NSE:M&M"


def test_kite_quotes_preferred(monkeypatch):
    payload = {
        "NSE:RELIANCE": {
            "last_price": 1330.0,
            "timestamp": "2026-07-20 13:05:00",
            "ohlc": {"open": 1320.0, "high": 1335.0, "low": 1318.0,
                     "close": 1325.0},
        }
    }
    f = UnifiedDataFetcher()
    monkeypatch.setattr(f, "_get_kite", lambda: _fake_kite(payload))
    rows = f._kite_quotes(["RELIANCE.NS"])
    assert "RELIANCE.NS" in rows
    r = rows["RELIANCE.NS"]
    assert r["source"] == "kite_realtime"
    assert r["last_price"] == 1330.0
    assert r["prev_close"] == 1325.0
    assert round(r["change_pct"], 4) == round((1330 / 1325 - 1) * 100, 4)


def test_get_live_quote_uses_kite_when_available(monkeypatch):
    payload = {"NSE:TCS": {"last_price": 2260.0,
                           "ohlc": {"close": 2250.0}}}
    f = UnifiedDataFetcher()
    monkeypatch.setattr(f, "_get_kite", lambda: _fake_kite(payload))
    df = f.get_live_quote(["TCS.NS"])
    assert df.iloc[0]["source"] == "kite_realtime"
    assert df.iloc[0]["last_price"] == 2260.0


def test_no_kite_returns_empty_and_falls_through(monkeypatch):
    f = UnifiedDataFetcher()
    monkeypatch.setattr(f, "_get_kite", lambda: None)
    assert f._kite_quotes(["RELIANCE.NS"]) == {}


def test_kite_batch_failure_degrades(monkeypatch):
    k = MagicMock()
    k.quote.side_effect = RuntimeError("network")
    f = UnifiedDataFetcher()
    monkeypatch.setattr(f, "_get_kite", lambda: k)
    assert f._kite_quotes(["RELIANCE.NS"]) == {}  # no crash, empty
