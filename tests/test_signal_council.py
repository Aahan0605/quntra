"""Tests for src/governor/council.py + PaperTrader exits — offline."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.db.session as db_session
from src.db import init_db
from src.governor.council import SignalCouncil


@pytest.fixture(autouse=True)
def _db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(f"sqlite:///{tmp_path}/council.db")
    yield
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def _trending_panel(n=120):
    """Panel where TCS trends up strongly and SBIN trends down."""
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    up = 100 * np.cumprod(1 + np.full(n, 0.004))
    down = 100 * np.cumprod(1 - np.full(n, 0.003))
    flat = np.full(n, 100.0) + np.random.default_rng(1).normal(0, 0.2, n)
    return pd.DataFrame({"TCS.NS": up, "SBIN.NS": down,
                         "ITC.NS": flat}, index=idx)


def test_score_premarket_ranks_trend(monkeypatch):
    council = SignalCouncil()
    panel = _trending_panel()
    bench = panel["ITC.NS"]
    with patch("src.utils.cache_loader.load_close_panel",
               return_value=panel), \
         patch("src.utils.cache_loader.load_benchmark", return_value=bench):
        scores = council.score_premarket(list(panel.columns))
    assert scores["TCS.NS"] > scores["SBIN.NS"]
    assert 0 <= min(scores.values()) and max(scores.values()) <= 12


def test_ml_vote_neutral_without_model():
    council = SignalCouncil()
    council._deployed_loaded = True  # no models loaded
    assert council._ml_vote("NOSUCH.NS") == 1


def test_live_signals_respect_daily_cap():
    council = SignalCouncil(capital=25_000)
    prices = pd.DataFrame({"close": [500.0] * 10})
    with patch("src.utils.cache_loader.load_ticker", return_value=prices):
        s1 = council.live_signals(["A.NS", "B.NS", "C.NS", "D.NS"])
        assert len(s1) == 3  # MAX_TRADES_PER_DAY
        s2 = council.live_signals(["E.NS"])
        assert s2 == []  # budget exhausted for the day


def test_live_signals_skip_unaffordable():
    council = SignalCouncil(capital=25_000)  # per-trade budget ₹8,333
    pricey = pd.DataFrame({"close": [30_000.0] * 10})
    with patch("src.utils.cache_loader.load_ticker", return_value=pricey):
        assert council.live_signals(["MRF.NS"]) == []


def test_signal_hash_is_one_per_ticker_per_day():
    council = SignalCouncil()
    prices = pd.DataFrame({"close": [100.0] * 10})
    with patch("src.utils.cache_loader.load_ticker", return_value=prices):
        sigs = council.live_signals(["TCS.NS"])
    today = datetime.now(timezone.utc).date().isoformat()
    assert sigs[0]["signal_hash"] == f"TCS.NS-{today}"


def test_live_signals_rejects_earnings_blacklist():
    from src.db import SystemState, get_session
    with get_session() as s:
        s.merge(SystemState(key="earnings_blacklist",
                            value={"tickers": ["TCS.NS"]}))
    council = SignalCouncil()
    prices = pd.DataFrame({"close": [100.0] * 10})
    with patch("src.utils.cache_loader.load_ticker", return_value=prices):
        sigs = council.live_signals(["TCS.NS", "INFY.NS"])
    tickers = [s["ticker"] for s in sigs]
    assert "TCS.NS" not in tickers      # in earnings blackout
    assert "INFY.NS" in tickers


# --------------------------------------------------------------------- #
# PaperTrader exit engine

class _QuoteFetcher:
    def __init__(self, price):
        self.price = price

    def get_live_quote(self, tickers):
        return pd.DataFrame([{"ticker": tickers[0],
                              "last_price": self.price}])


class _NullBrain:
    def remember_trade(self, *_a, **_k):
        return "x"


def _trader_with_position(entry=100.0, days_old=0):
    from src.execution.paper_trader import PaperTrader
    trader = PaperTrader(brain=_NullBrain(), fetcher=_QuoteFetcher(entry))
    trade = trader.place_order("TCS.NS", "LONG", qty=1, price=entry,
                               signal_hash="h1")
    assert trade["status"] == "FILLED"
    if days_old:
        pos = trader._positions["h1"]
        pos["entry_time"] = datetime.now(timezone.utc) - timedelta(
            days=days_old)
    return trader


def test_stop_loss_exit():
    trader = _trader_with_position(entry=100.0)
    trader.fetcher.price = 97.0  # -3% < -2% stop
    closed = trader.manage_positions()
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "STOP_LOSS"
    assert trader.get_positions() == []


def test_take_profit_exit():
    trader = _trader_with_position(entry=100.0)
    trader.fetcher.price = 105.0  # +5% > +4% target
    closed = trader.manage_positions()
    assert closed[0]["exit_reason"] == "TAKE_PROFIT"


def test_time_stop_exit():
    trader = _trader_with_position(entry=100.0, days_old=8)
    trader.fetcher.price = 100.5  # inside the band, but stale
    closed = trader.manage_positions()
    assert closed[0]["exit_reason"] == "TIME_STOP"


def test_position_held_inside_band():
    trader = _trader_with_position(entry=100.0)
    trader.fetcher.price = 101.0
    assert trader.manage_positions() == []
    assert len(trader.get_positions()) == 1
