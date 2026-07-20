"""Tests for the Obsidian vault exporter — offline, temp vault."""

from datetime import datetime, timedelta, timezone

import pytest

import src.db.session as db_session
from src.db import ResearchNote, Trade, get_session, init_db
from src.integrations import ObsidianVault
from src.knowledge import KnowledgeManager


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'vault.db'}"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    yield url
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def _seed(db_url):
    now = datetime.now(timezone.utc)
    with get_session(db_url) as s:
        s.add(Trade(
            ticker="RELIANCE.NS", direction="LONG", quantity=1,
            entry_price=1300.0, exit_price=1352.0, entry_time=now,
            exit_time=now, pnl=52.0, pnl_pct=0.04, signal_score=10,
            regime="BULL_TRENDING", exit_reason="TAKE_PROFIT", is_paper=True,
        ))
        s.add(ResearchNote(
            note_type="MACRO", ticker="RELIANCE.NS",
            summary="Crude softening — positive for refiners",
            source="macro_agent", created_at=now,
        ))
    KnowledgeManager(db_url).store(
        "TRADE_LESSON", "RELIANCE gaps fade by noon", tickers=["RELIANCE.NS"])


def test_sync_creates_vault_structure(db_url, tmp_path):
    _seed(db_url)
    vault = tmp_path / "vault"
    counts = ObsidianVault(vault_dir=vault, db_url=db_url).sync()
    assert (vault / "Home.md").exists()
    assert counts["daily"] == 1
    assert counts["tickers"] == 1
    for sub in ("Daily", "Tickers", "Knowledge"):
        assert (vault / sub).is_dir()


def test_ticker_note_has_backlinks_and_data(db_url, tmp_path):
    _seed(db_url)
    vault = tmp_path / "vault"
    ObsidianVault(vault_dir=vault, db_url=db_url).sync()
    note = (vault / "Tickers" / "RELIANCE.NS.md").read_text()
    assert "# RELIANCE.NS" in note
    assert "TAKE_PROFIT" in note
    assert "₹+52" in note
    assert "[[" in note  # daily-note backlink
    assert "RELIANCE gaps fade by noon" in note  # lesson


def test_daily_note_links_ticker(db_url, tmp_path):
    _seed(db_url)
    vault = tmp_path / "vault"
    ObsidianVault(vault_dir=vault, db_url=db_url).sync()
    day = datetime.now(timezone.utc).date().isoformat()
    note = (vault / "Daily" / f"{day}.md").read_text()
    assert "[[RELIANCE.NS]]" in note
    assert "type: daily-note" in note
    assert "Crude softening" in note


def test_sync_is_idempotent(db_url, tmp_path):
    _seed(db_url)
    vault = tmp_path / "vault"
    exp = ObsidianVault(vault_dir=vault, db_url=db_url)
    first = exp.sync()
    second = exp.sync()
    assert first["tickers"] == second["tickers"] == 1


def test_empty_db_yields_empty_vault_no_crash(db_url, tmp_path):
    vault = tmp_path / "vault"
    counts = ObsidianVault(vault_dir=vault, db_url=db_url).sync()
    assert counts["daily"] == 0 and counts["tickers"] == 0
    assert (vault / "Home.md").exists()


def test_hermes_sync_obsidian(db_url, tmp_path, monkeypatch):
    from src.governor.brain import QuNtraBrain
    from src.governor.hermes import HermesCoordinator
    _seed(db_url)
    monkeypatch.setenv("OBSIDIAN_VAULT_DIR", str(tmp_path / "hvault"))
    h = HermesCoordinator(brain=QuNtraBrain(), trader=None, db_url=db_url,
                          research_team={})
    r = h.sync_obsidian()
    assert "error" not in r
    assert r["tickers"] == 1
    assert (tmp_path / "hvault" / "Home.md").exists()
