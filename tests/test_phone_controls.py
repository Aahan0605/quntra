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


# --- capital override + post-token pre-market ------------------------------
# DAILY_CAPITAL_INR lives in the Render blueprint; changing it needs a
# dashboard sync, which is not doable from a phone before the open.

class _Alloc:
    capital = 25_000.0
    universe = ["A"]


def test_capital_shows_current_when_called_bare(bot):
    bot.hermes.allocator = _Alloc()
    out = bot.dispatch("capital")
    assert "25,000" in out and "none" in out


def test_capital_sets_override_and_applies_it(bot):
    bot.hermes.allocator = _Alloc()
    bot.hermes.trader.cash = 25_000.0
    out = bot.dispatch("capital", "10000000")
    assert "10,000,000" in out
    assert bot.hermes.allocator.capital == 10_000_000.0
    # cash must follow capital, or the book buys into a size it can't fund
    assert bot.hermes.trader.cash == 10_000_000.0
    assert (bot.hermes.get_system_state("capital") or {})["inr"] == 10_000_000.0


def test_capital_override_survives_a_restart(bot):
    """The whole point of storing it in the DB rather than the process."""
    bot.hermes.set_system_state("capital", {"inr": 5_000_000.0})
    bot.hermes.allocator = _Alloc()          # fresh object, as after a restart
    assert bot.hermes.apply_capital_override() == 5_000_000.0
    assert bot.hermes.allocator.capital == 5_000_000.0


def test_capital_rejects_junk(bot):
    bot.hermes.allocator = _Alloc()
    assert "Not a number" in bot.dispatch("capital", "lots")
    assert "positive" in bot.dispatch("capital", "-5")


def test_capital_accepts_commas(bot):
    bot.hermes.allocator = _Alloc()
    bot.dispatch("capital", "1,00,00,000")
    assert bot.hermes.allocator.capital == 10_000_000.0


def test_token_upload_starts_premarket(bot, monkeypatch):
    """The operator's token upload IS the start of the day — pre_market is
    cron'd at 06:00 but the token lands after it."""
    ran = []
    monkeypatch.setattr(bot.hermes, "run_pre_market_sequence",
                        lambda: ran.append(True))
    msg = bot._premarket_after_token()
    import time
    time.sleep(0.3)
    assert "Pre-market sequence started" in msg
    assert ran == [True]


def test_token_upload_does_not_repeat_a_done_premarket(bot, monkeypatch):
    from datetime import date
    bot.hermes.set_system_state("premarket",
                                {"started_at": date.today().isoformat()})
    ran = []
    monkeypatch.setattr(bot.hermes, "run_pre_market_sequence",
                        lambda: ran.append(True))
    msg = bot._premarket_after_token()
    assert "already ran today" in msg
    assert ran == []
