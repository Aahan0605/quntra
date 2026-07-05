"""Tests for the database layer (Task 1-4)."""
import uuid
from datetime import datetime, timezone

import pytest

import src.db.session as db_session
from src.db import Trade, Signal, SystemState, init_db, get_session


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Isolated SQLite DB per test."""
    url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    yield url
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_all_seven_tables_created(fresh_db):
    from sqlalchemy import inspect
    from src.db.session import get_engine
    tables = set(inspect(get_engine()).get_table_names())
    assert {"trades", "signals", "agent_credibility", "backtest_results",
            "price_data", "research_notes", "system_state"}.issubset(tables)


def test_trade_write_and_read_roundtrip(fresh_db):
    trade_id = None
    with get_session() as s:
        t = Trade(
            signal_hash=uuid.uuid4().hex,
            ticker="RELIANCE.NS",
            direction="LONG",
            entry_price=2450.50,
            quantity=4,
            entry_time=datetime.now(timezone.utc),
            signal_score=10,
            regime="BULL_TREND",
            is_paper=True,
        )
        s.add(t)
        s.flush()
        trade_id = t.id

    with get_session() as s:
        back = s.get(Trade, trade_id)
        assert back is not None
        assert back.ticker == "RELIANCE.NS"
        assert back.direction == "LONG"
        assert float(back.entry_price) == 2450.50
        assert back.is_paper is True


def test_signal_hash_dedup(fresh_db):
    from sqlalchemy.exc import IntegrityError
    h = uuid.uuid4().hex
    with get_session() as s:
        s.add(Signal(signal_hash=h, ticker="TCS.NS", score=9, direction="LONG"))
    with pytest.raises(IntegrityError):
        with get_session() as s:
            s.add(Signal(signal_hash=h, ticker="TCS.NS", score=9, direction="LONG"))


def test_system_state_upsert(fresh_db):
    with get_session() as s:
        s.merge(SystemState(key="regime", value={"state": "BULL", "conf": 0.8}))
    with get_session() as s:
        s.merge(SystemState(key="regime", value={"state": "BEAR", "conf": 0.6}))
    with get_session() as s:
        row = s.get(SystemState, "regime")
        assert row.value["state"] == "BEAR"
