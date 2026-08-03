"""Phone-run controls: /start_trading and /breeze_token — offline, mocked."""

from unittest.mock import MagicMock

import pytest

import src.db.session as db_session
from src.alerts.telegram_bot import QuNtraTelegramBot, TelegramAlerter
from src.db import init_db
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator


class StubTrader:
    enabled = True

    def get_positions(self):
        return []


class Council:
    """Minimal council that scores a fixed watchlist."""

    def score_premarket(self, universe):
        return {t: (10 if i < 2 else 3) for i, t in enumerate(universe)}

    def live_signals(self, watch):
        return []


@pytest.fixture
def bot(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/phone.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    hermes = HermesCoordinator(
        brain=QuNtraBrain(), trader=StubTrader(), fetcher=None,
        council=Council(), research_team={}, db_url=url)
    b = QuNtraTelegramBot(hermes, alerter=TelegramAlerter(test_mode=True),
                          db_url=url)
    yield b
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_start_trading_closed_market(bot, monkeypatch):
    monkeypatch.setattr("scripts.scheduler.is_trading_day", lambda *a: False)
    out = bot.dispatch("start_trading")
    assert "closed" in out.lower()


def test_start_trading_arms_and_builds_watchlist(bot, monkeypatch):
    monkeypatch.setattr("scripts.scheduler.is_trading_day", lambda *a: True)
    out = bot.dispatch("start_trading")
    assert "started" in out.lower() or "armed" in out.lower()
    # OMS armed + watchlist persisted to shared state
    assert (bot.hermes.get_system_state("oms") or {}).get("enabled") is True
    watch = (bot.hermes.get_system_state("premarket") or {}).get("watchlist")
    assert watch  # non-empty


def test_breeze_token_usage_when_no_arg(bot, monkeypatch):
    monkeypatch.setenv("ICICI_BREEZE_API_KEY", "fake_test_api_key_123")
    out = bot.dispatch("breeze_token")
    assert "Usage" in out
    assert "session_token" in out


def test_breeze_token_sets_and_confirms(bot, monkeypatch):
    called = {}

    def fake_set(tok):
        called["tok"] = tok
        return tok

    monkeypatch.setattr("src.integrations.breeze_session.set_token", fake_set)
    monkeypatch.setattr(
        "src.integrations.breeze_session.token_status", lambda: "valid")
    out = bot.dispatch("breeze_token", "56500413")
    assert called["tok"] == "56500413"
    assert "updated" in out.lower()
    assert "565004" in out


def test_breeze_token_failure(bot, monkeypatch):
    def boom(tok):
        raise RuntimeError("Authentication Fail :: Invalid Checksum.")

    monkeypatch.setattr("src.integrations.breeze_session.set_token", boom)
    out = bot.dispatch("breeze_token", "badtoken")
    assert "couldn't use that token" in out.lower()
    assert "apisession" in out.lower()


def test_dispatch_never_raises_new_commands(bot, monkeypatch):
    # both new commands must return a string, never throw
    monkeypatch.setattr("scripts.scheduler.is_trading_day", lambda *a: False)
    for name in ("start_trading", "breeze_token"):
        r = bot.dispatch(name)
        assert isinstance(r, str) and r
