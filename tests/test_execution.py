"""Tests for PaperTrader + KiteOMS (Tasks 2-1, 2-2)."""
import pandas as pd
import pytest

import src.db.session as db_session
from src.db import init_db
from src.governor.brain import QuNtraBrain
from src.execution.paper_trader import PaperTrader
from src.execution.kite_oms import KiteOMS, OrderState


class StubFetcher:
    def __init__(self, price=2450.0):
        self.price = price

    def get_live_quote(self, tickers):
        return pd.DataFrame([{"ticker": t, "last_price": self.price}
                             for t in tickers])


@pytest.fixture
def brain(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/exec.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    yield QuNtraBrain()
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


@pytest.fixture
def paper(brain):
    return PaperTrader(brain=brain, fetcher=StubFetcher(), starting_cash=100_000)


# --------------------------------------------------------------------- #
# PaperTrader

def test_paper_fill_applies_slippage_long(paper):
    t = paper.place_order("RELIANCE.NS", "LONG", 4, signal_hash="h1")
    assert t["status"] == "FILLED"
    assert t["entry_price"] == pytest.approx(2450.0 * 1.0005, abs=0.01)
    assert t["is_paper"] is True


def test_paper_fill_applies_slippage_short(paper):
    t = paper.place_order("RELIANCE.NS", "SHORT", 4, signal_hash="h2")
    assert t["entry_price"] == pytest.approx(2450.0 * 0.9995, abs=0.01)


def test_paper_duplicate_hash_rejected(paper):
    paper.place_order("TCS.NS", "LONG", 1, signal_hash="dup")
    r = paper.place_order("TCS.NS", "LONG", 1, signal_hash="dup")
    assert r["status"] == "REJECTED"
    assert "duplicate" in r["reason"]


def test_paper_disabled_rejects(paper):
    paper.disable()
    r = paper.place_order("INFY.NS", "LONG", 1, signal_hash="h3")
    assert r["status"] == "REJECTED"


def test_paper_trade_persisted_to_db(paper, brain):
    paper.place_order("SBIN.NS", "LONG", 10, signal_hash="h4")
    from src.db import Trade, get_session
    with get_session() as s:
        rows = s.query(Trade).filter(Trade.ticker == "SBIN.NS").all()
        assert len(rows) == 1
        assert rows[0].is_paper is True


def test_paper_close_computes_pnl(paper):
    paper.place_order("ITC.NS", "LONG", 10, signal_hash="h5")
    paper.fetcher.price = 2500.0  # price moved up
    closed = paper.close_position("h5", exit_reason="TP")
    assert closed is not None
    assert closed["pnl"] > 0
    assert closed["exit_reason"] == "TP"
    assert paper.get_positions() == []


def test_paper_cash_tracking(paper):
    start = paper.cash
    paper.place_order("LT.NS", "LONG", 2, signal_hash="h6")
    assert paper.cash < start  # notional + fees deducted


# --------------------------------------------------------------------- #
# KiteOMS (offline — no credentials, no live calls)

def test_kite_state_machine_transitions(brain):
    oms = KiteOMS(brain=brain, daily_capital_inr=25_000, max_trades_per_day=4)
    order = oms.place_order("RELIANCE.NS", "LONG", 1, signal_hash="k1",
                            price=2450.0)
    # Not connected -> REJECTED with clear reason
    assert order["state"] == OrderState.REJECTED.value
    assert "not connected" in order["reject_reason"]
    history = [s for s, _ in order["state_history"]]
    assert history == ["PENDING", "REJECTED"]


def test_kite_max_trades_cap(brain):
    oms = KiteOMS(brain=brain, daily_capital_inr=10**9, max_trades_per_day=4)
    oms._trades_today = 4  # simulate cap reached
    order = oms.place_order("TCS.NS", "LONG", 1, signal_hash="k2", price=100.0)
    assert order["state"] == OrderState.REJECTED.value
    assert "max 4 trades/day" in order["reject_reason"]
    # Rejected signal logged to DB with reason
    from src.db import Signal, get_session
    with get_session() as s:
        sig = s.query(Signal).filter(Signal.signal_hash == "k2").one()
        assert sig.executed is False
        assert "max 4" in sig.rejection_reason


def test_kite_daily_capital_enforced(brain):
    oms = KiteOMS(brain=brain, daily_capital_inr=10_000, max_trades_per_day=10)
    oms._capital_used_today = 9_000
    order = oms.place_order("INFY.NS", "LONG", 1, signal_hash="k3",
                            price=1_500.0)
    assert order["state"] == OrderState.REJECTED.value
    assert "daily capital" in order["reject_reason"]


def test_kite_duplicate_hash_rejected(brain):
    oms = KiteOMS(brain=brain)
    oms._seen_hashes.add("k4")
    order = oms.place_order("SBIN.NS", "LONG", 1, signal_hash="k4", price=800.0)
    assert order["state"] == OrderState.REJECTED.value
    assert "duplicate" in order["reject_reason"]


def test_kite_illegal_transition_raises(brain):
    oms = KiteOMS(brain=brain)
    order = oms.place_order("LT.NS", "LONG", 1, signal_hash="k5", price=3500.0)
    assert order["state"] == OrderState.REJECTED.value
    with pytest.raises(ValueError):
        oms._transition(order, OrderState.FILLED)  # REJECTED is terminal


def test_kite_connect_without_creds_raises(brain, monkeypatch):
    for var in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    oms = KiteOMS(brain=brain)
    oms.api_key = None
    oms.access_token = None
    with pytest.raises(RuntimeError, match="BLOCKER"):
        oms.connect()


def test_paper_and_kite_share_interface(brain):
    """One-config-change switch: both expose the same OMS surface."""
    for cls_obj in (PaperTrader(brain=brain, fetcher=StubFetcher()),
                    KiteOMS(brain=brain)):
        for method in ("place_order", "cancel_order", "get_positions",
                       "manage_positions", "reconcile", "disable", "enable"):
            assert callable(getattr(cls_obj, method)), \
                f"{type(cls_obj).__name__} missing {method}"
