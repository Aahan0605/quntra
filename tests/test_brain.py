"""Tests for QuNtraBrain (Task 1-6)."""
import pytest

import src.db.session as db_session
from src.db import init_db
from src.governor.brain import QuNtraBrain


@pytest.fixture
def brain(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/brain.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    yield QuNtraBrain()
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_trade_memory_roundtrip(brain):
    tid = brain.remember_trade({
        "ticker": "INFY.NS", "direction": "LONG", "entry_price": 1500.0,
        "pnl": 320.0, "pnl_pct": 0.021, "signal_score": 10,
        "regime": "BULL_TREND", "is_paper": True, "signal_hash": "abc123",
    })
    assert tid
    recalled = brain.recall_similar_conditions("BULL_TREND")
    assert len(recalled) == 1
    assert recalled[0]["ticker"] == "INFY.NS"
    assert recalled[0]["pnl"] == 320.0


def test_recall_filters_by_regime(brain):
    brain.remember_trade({"ticker": "A.NS", "direction": "LONG",
                          "regime": "BULL_TREND"})
    brain.remember_trade({"ticker": "B.NS", "direction": "SHORT",
                          "regime": "BEAR_TREND"})
    bull = brain.recall_similar_conditions("BULL_TREND")
    assert [t["ticker"] for t in bull] == ["A.NS"]


def test_credibility_compound_updates(brain):
    assert brain.get_agent_credibility("technical") == 1.0  # default
    w1 = brain.update_agent_credibility("technical", correct=True)
    assert w1 == pytest.approx(1.05)
    w2 = brain.update_agent_credibility("technical", correct=True)
    assert w2 == pytest.approx(1.1025)
    w3 = brain.update_agent_credibility("technical", correct=False)
    assert w3 == pytest.approx(1.1025 * 0.95)


def test_credibility_floor_and_ceiling(brain):
    for _ in range(100):
        brain.update_agent_credibility("loser", correct=False)
    assert brain.get_agent_credibility("loser") == pytest.approx(0.1)
    for _ in range(100):
        brain.update_agent_credibility("winner", correct=True)
    assert brain.get_agent_credibility("winner") == pytest.approx(3.0)


def test_credibility_persists_across_instances(brain):
    brain.update_agent_credibility("macro", correct=True)
    fresh = QuNtraBrain()  # new instance, same DB
    assert fresh.get_agent_credibility("macro") == pytest.approx(1.05)


def test_lessons_learned(brain):
    brain.store_lesson("Do not fade RBI policy days", {"context": "test"})
    brain.store_lesson("Reduce size in SIDEWAYS_CHOP", {"context": "test2"})
    lessons = brain.get_lessons_learned(limit=5)
    assert len(lessons) == 2
    assert any("RBI" in les["lesson"] for les in lessons)


def test_signal_memory(brain):
    sid = brain.remember_signal({
        "ticker": "TCS.NS", "score": 11, "direction": "LONG",
        "agent_votes": {"technical": 2, "valuation": 1},
        "executed": False, "rejection_reason": "score below live threshold",
        "signal_hash": "xyz789",
    })
    assert sid
