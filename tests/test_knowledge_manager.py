"""Tests for src/knowledge/knowledge_manager.py — offline, SQLite in tmp."""

import pytest

from src.db import Base, get_engine
from src.knowledge import KnowledgeManager


@pytest.fixture
def km(tmp_path):
    url = f"sqlite:///{tmp_path / 'test_knowledge.db'}"
    Base.metadata.create_all(get_engine(url))
    return KnowledgeManager(db_url=url)


def test_store_and_recall(km):
    kid = km.store("TRADE_LESSON",
                   "Chasing breakouts in RELIANCE after 2pm lost money "
                   "three times — avoid late-session momentum entries",
                   tickers=["RELIANCE.NS"], confidence=0.8,
                   source="post_market_review")
    assert kid
    hits = km.recall("breakouts late-session")
    assert len(hits) == 1
    assert hits[0]["knowledge_type"] == "TRADE_LESSON"
    assert "RELIANCE" in hits[0]["content"]


def test_recall_filters_by_type(km):
    km.store("TRADE_LESSON", "stop placement too tight on volatile days")
    km.store("MACRO_OBSERVATION", "oil spike hit OMC stocks within two days")
    hits = km.recall("days", knowledge_type="MACRO_OBSERVATION")
    assert len(hits) == 1
    assert hits[0]["knowledge_type"] == "MACRO_OBSERVATION"


def test_recall_by_regime(km):
    km.store("MARKET_OBSERVATION", "banks led the rally",
             regime="BULL_TRENDING")
    km.store("MARKET_OBSERVATION", "defensive IT held up",
             regime="BEAR_VOLATILE")
    hits = km.recall_by_regime("BULL_TRENDING")
    assert len(hits) == 1
    assert hits[0]["regime"] == "BULL_TRENDING"


def test_recall_by_ticker(km):
    km.store("COMPANY_RESEARCH", "margins expanding two quarters running",
             tickers=["TCS.NS", "INFY.NS"])
    km.store("COMPANY_RESEARCH", "debt reduced ahead of plan",
             tickers=["RELIANCE.NS"])
    hits = km.recall_by_ticker("TCS.NS")
    assert len(hits) == 1
    assert "margins" in hits[0]["content"]


def test_recall_similar_conditions(km):
    km.store("MACRO_OBSERVATION", "high vix + weak dollar was good for IT",
             conditions={"regime": "BULL_TRENDING", "vix_high": True})
    km.store("MACRO_OBSERVATION", "quiet tape, nothing worked",
             conditions={"regime": "SIDEWAYS", "vix_high": False})
    hits = km.recall_similar_market_conditions(
        {"regime": "BULL_TRENDING", "vix_high": True})
    assert hits
    assert hits[0]["similarity"] == 2
    assert "IT" in hits[0]["content"]


def test_digest_and_count(km):
    assert km.count() == 0
    km.store("STRATEGY_INSIGHT", "weekly rebalance beat daily on costs",
             confidence=0.9)
    km.store("TRADE_LESSON", "never add to losers", confidence=0.7)
    assert km.count() == 2
    digest = km.generate_knowledge_digest()
    assert "STRATEGY_INSIGHT" in digest
    assert "TRADE_LESSON" in digest
    assert "2 new items" in digest


def test_empty_digest(km):
    digest = km.generate_knowledge_digest()
    assert "No new knowledge" in digest


def test_confidence_clamped(km):
    kid = km.store("TRADE_LESSON", "overconfident entry", confidence=1.7)
    hits = km.recall("overconfident")
    assert hits[0]["confidence"] == 1.0
