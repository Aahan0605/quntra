"""Tests for the Telegram command center (Task 1-8) — test mode, offline."""
import pytest

import src.db.session as db_session
from src.db import init_db
from src.alerts.telegram_bot import TelegramAlerter, QuNtraTelegramBot
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator
from src.risk.drawdown_circuit import DrawdownCircuitBreaker
from src.risk.consecutive_loss_guard import ConsecutiveLossGuard


class StubTrader:
    def __init__(self):
        self.enabled = True

    def get_positions(self):
        return []

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True


@pytest.fixture
def bot(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/tg.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    alerter = TelegramAlerter(test_mode=True)
    hermes = HermesCoordinator(
        brain=QuNtraBrain(), trader=StubTrader(), telegram=alerter,
        circuit_breaker=DrawdownCircuitBreaker(),
        loss_guard=ConsecutiveLossGuard(),
    )
    yield QuNtraTelegramBot(hermes, alerter)
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_all_six_alert_types(bot):
    a = bot.alerter
    assert a.trade_filled("RELIANCE.NS", "LONG", 2450.0, 4, 10)
    assert a.stop_hit("RELIANCE.NS", -340, -0.012)
    assert a.kill_switch(3)
    assert a.daily_circuit(2, -0.046)
    assert a.eod_report(1_250, 1.32, 3, "BULL_TREND", "cautiously long")
    assert a.error_alert("data_fetcher", "NSE timeout")
    assert len(a.sent) == 6
    assert "✅ LONG RELIANCE.NS" in a.sent[0]
    assert "Score 10/12" in a.sent[0]
    assert "🛑 STOP HIT" in a.sent[1]
    assert "KILL SWITCH" in a.sent[2]
    assert "CIRCUIT LEVEL 2" in a.sent[3]
    assert "Sharpe (30d): 1.32" in a.sent[4]
    assert "🚨 ERROR in data_fetcher" in a.sent[5]


def test_cmd_status(bot):
    out = bot.cmd_status()
    assert "QuNtra status" in out
    assert "OMS enabled" in out


def test_cmd_pause_and_resume(bot):
    bot.cmd_pause()
    assert (bot.hermes.get_system_state("oms") or {}).get("enabled") is False
    out = bot.cmd_resume()
    assert (bot.hermes.get_system_state("oms") or {}).get("enabled") is True
    assert "resumed" in out.lower()


def test_cmd_resume_clears_halts(bot):
    for _ in range(3):
        bot.hermes.loss_guard.record_trade_outcome(-1)
    assert bot.hermes.loss_guard.halted
    bot.cmd_resume()
    assert not bot.hermes.loss_guard.halted


def test_cmd_halt_disables_everything(bot):
    out = bot.cmd_halt()
    assert "HALT" in out
    assert (bot.hermes.get_system_state("oms") or {}).get("enabled") is False
    assert bot.hermes.trader.enabled is False


def test_cmd_report(bot):
    out = bot.cmd_report()
    assert "QuNtra EOD" in out


def test_cmd_override(bot):
    assert "Usage" in bot.cmd_override()
    out = bot.cmd_override("abc123")
    assert "abc123" in out
    assert (bot.hermes.get_system_state("override") or {})["signal_hash"] == "abc123"


def test_polling_refuses_without_token(bot):
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        bot.run_polling()
