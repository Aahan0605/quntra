"""Tests for src/agents/intelligence/note_processor.py — offline."""

from unittest.mock import patch

import pytest

from src.agents.intelligence import NoteProcessor
from src.db import Base, get_engine


@pytest.fixture
def db_url(tmp_path):
    url = f"sqlite:///{tmp_path / 'notes.db'}"
    Base.metadata.create_all(get_engine(url))
    return url


def test_oil_note_verified_rising(db_url):
    np_ = NoteProcessor(db_url)
    with patch("src.agents.intelligence.note_processor.yf_pct_change",
               return_value=2.1):
        resp = np_.process("Oil prices are rising because of Iran sanctions")
    assert "OIL" in resp.entities and "IRAN" in resp.entities
    assert resp.verified is True
    assert "+2.10%" in resp.verification_detail
    assert resp.relevance == "HIGH"
    assert resp.note_id is not None
    msg = resp.to_message()
    assert "✓ verified" in msg and "HIGH" in msg


def test_false_claim_not_confirmed(db_url):
    np_ = NoteProcessor(db_url)
    with patch("src.agents.intelligence.note_processor.yf_pct_change",
               return_value=-1.5):
        resp = np_.process("Gold is surging today")
    assert resp.verified is False
    assert "not confirmed" in resp.action


def test_note_without_direction_is_unverifiable(db_url):
    resp = NoteProcessor(db_url).process("Watch the RBI policy meeting")
    assert resp.verified is None


def test_ticker_note_relevance(db_url):
    np_ = NoteProcessor(db_url)
    resp = np_.process("Tata Steel might announce expansion")
    assert "TATASTEEL.NS" in resp.entities
    assert resp.relevance in ("MEDIUM", "HIGH")


def test_stored_with_user_note_source(db_url):
    np_ = NoteProcessor(db_url)
    with patch("src.agents.intelligence.note_processor.yf_pct_change",
               return_value=1.0):
        resp = np_.process("Oil rising fast")
    from sqlalchemy import select
    from src.db import ResearchNote, get_session
    with get_session(db_url) as s:
        row = s.execute(select(ResearchNote)).scalar_one()
        assert row.source == "USER_NOTE"
        assert row.entities["verified"] is True


def test_high_relevance_verified_nudges_hermes(db_url):
    class FakeHermes:
        def __init__(self):
            self.state = {}
            self.trader = type("T", (), {"get_positions": lambda s: []})()

        def set_system_state(self, k, v):
            self.state[k] = v

    hermes = FakeHermes()
    np_ = NoteProcessor(db_url, hermes=hermes)
    with patch("src.agents.intelligence.note_processor.yf_pct_change",
               return_value=3.0):
        resp = np_.process("Oil rising on war escalation fears")
    assert resp.relevance == "HIGH"
    assert hermes.state["user_note_bias"]["bias"] == "CAUTION"


def test_processor_never_raises(db_url):
    np_ = NoteProcessor(db_url)
    with patch.object(NoteProcessor, "extract_entities",
                      side_effect=RuntimeError("boom")):
        resp = np_.process("anything")
    assert "processor error" in resp.verification_detail
