"""Tests for the expanded 22-command Telegram center — offline."""

import pytest

import src.db.session as db_session
from src.alerts.telegram_bot import QuNtraTelegramBot, TelegramAlerter
from src.db import init_db
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator
from src.risk.consecutive_loss_guard import ConsecutiveLossGuard
from src.risk.drawdown_circuit import DrawdownCircuitBreaker


class StubTrader:
    enabled = True

    def __init__(self):
        self.positions = [{
            "signal_hash": "h1", "ticker": "TCS.NS", "direction": "LONG",
            "quantity": 2, "entry_price": 4000.0,
        }]
        self.closed = []

    def get_positions(self):
        return self.positions

    def close_position(self, signal_hash, price=None, exit_reason="MANUAL"):
        pos = next((p for p in self.positions
                    if p["signal_hash"] == signal_hash), None)
        if pos:
            self.positions.remove(pos)
            self.closed.append(exit_reason)
            return {**pos, "pnl": 10.0, "exit_reason": exit_reason}
        return None

    def disable(self):
        self.enabled = False


@pytest.fixture
def bot(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/tg.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    hermes = HermesCoordinator(
        brain=QuNtraBrain(), trader=StubTrader(), fetcher=None,
        circuit_breaker=DrawdownCircuitBreaker(),
        loss_guard=ConsecutiveLossGuard(),
        research_team={},
    )
    # NEVER alerter=None here: QuNtraTelegramBot.__init__ does
    # `alerter or TelegramAlerter.from_config()` — None doesn't mean "no
    # alerts," it means "fall back to the real configured bot." This
    # fixture used to do exactly that, so every run of
    # test_dispatch_never_raises (which dispatches every command,
    # including /start_trading) sent a real "Starting paper session…"
    # push to the operator's actual phone. test_mode=True is what makes
    # this offline.
    b = QuNtraTelegramBot(hermes, alerter=TelegramAlerter(test_mode=True))
    yield b
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_all_34_commands_registered(bot):
    assert len(bot.COMMANDS) == 34
    for name in bot.COMMANDS:
        assert callable(getattr(bot, f"cmd_{name}", None)), \
            f"cmd_{name} missing"


def test_dispatch_never_raises(bot):
    for name in bot.COMMANDS:
        reply = bot.dispatch(name)
        assert isinstance(reply, str) and reply, f"/{name} gave no reply"
        # dispatch() catches every exception and turns it into a "⚠️ /{name}
        # failed: {e}" string reply — so a command that always raises would
        # still pass the check above. This is the actual functional
        # assertion: no command may be silently broken behind that catch.
        assert not reply.startswith(f"⚠️ /{name} failed:"), (
            f"/{name} is raising internally: {reply}")


def test_dispatch_never_touches_the_real_telegram_api(bot):
    """The regression this file's fixture bug would have been caught by:
    /start_trading is the one command that unconditionally calls
    alerter.send(). It must land in the test-mode ledger, never a real
    HTTP request."""
    assert bot.alerter.test_mode is True
    bot.dispatch("start_trading")
    assert any("Starting paper session" in msg for msg in bot.alerter.sent)


def test_dispatch_unknown_command(bot):
    assert "Unknown command" in bot.dispatch("does_not_exist")


def test_dispatch_logs_to_system_state(bot):
    bot.dispatch("status")
    logged = bot.hermes.get_system_state("last_command")
    assert logged["command"] == "status"


def test_portfolio_shows_sector_exposure(bot):
    reply = bot.dispatch("portfolio")
    assert "TCS.NS" in reply
    assert "IT" in reply  # sector mapping


def test_risk_dashboard_fields(bot):
    reply = bot.dispatch("risk")
    for fragment in ["OMS enabled", "Circuit breaker",
                     "Consecutive losses", "Max DD", "Regime"]:
        assert fragment in reply


def test_health_names_all_services(bot):
    reply = bot.dispatch("health")
    for svc in ["PostgreSQL/DB", "Scheduler", "PaperTrader", "Brain",
                "DataFetcher"]:
        assert svc in reply


def test_emergency_stop_squares_and_disables(bot):
    reply = bot.dispatch("emergency_stop")
    assert "EMERGENCY STOP" in reply
    assert bot.hermes.trader.positions == []
    assert bot.hermes.trader.closed == ["EMERGENCY_STOP"]
    assert bot.hermes.trader.enabled is False
    oms = bot.hermes.get_system_state("oms")
    assert oms["enabled"] is False


def test_start_live_blocked_before_gate(bot):
    reply = bot.dispatch("start_live")
    assert "BLOCKED" in reply
    assert "40-day" in reply


def test_note_command_stores(bot, monkeypatch):
    import src.agents.intelligence.note_processor as np_mod
    monkeypatch.setattr(np_mod, "yf_pct_change", lambda *a, **k: 2.0)
    reply = bot.dispatch("note", "Oil", "rising", "on", "sanctions")
    assert "Note processed" in reply
    assert "verified" in reply


def test_note_requires_text(bot):
    assert "Usage" in bot.dispatch("note")


def test_chat_recalls_knowledge(bot):
    from src.knowledge import KnowledgeManager
    KnowledgeManager().store("TRADE_LESSON",
                             "RELIANCE breakouts after 2pm always failed")
    reply = bot.dispatch("chat", "what", "about", "RELIANCE", "breakouts")
    assert "RELIANCE" in reply


def test_watchlist_empty_message(bot):
    assert "empty" in bot.dispatch("watchlist")
