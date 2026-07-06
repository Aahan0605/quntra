"""Tests for the v4.0 hardening loop: auth bootstrap, notifications,
watchdog, token rotation, status flags. All offline."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import src.db.session as db_session
from src.alerts.telegram_bot import (
    HELP_TEXT,
    QuNtraTelegramBot,
    TelegramAlerter,
)
from src.db import init_db
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator
from src.risk.consecutive_loss_guard import ConsecutiveLossGuard
from src.risk.drawdown_circuit import CircuitLevel, DrawdownCircuitBreaker


class SpyAlerter(TelegramAlerter):
    """Records sends without touching the network."""

    def __init__(self):
        super().__init__(token=None, chat_id=None, test_mode=True)


class StubTrader:
    enabled = True

    def get_positions(self):
        return []

    def disable(self):
        self.enabled = False


@pytest.fixture
def bot(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/v4.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    # secrets.env writes go to a scratch file, never the real one
    import src.alerts.telegram_bot as tg_mod
    fake_secrets = tmp_path / "secrets.env"
    fake_secrets.write_text("TELEGRAM_BOT_TOKEN=1:x\n")
    monkeypatch.setattr(tg_mod, "SECRETS", fake_secrets)
    hermes = HermesCoordinator(
        brain=QuNtraBrain(), trader=StubTrader(), research_team={})
    b = QuNtraTelegramBot(hermes, alerter=SpyAlerter())
    b._fake_secrets = fake_secrets
    yield b
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


# --------------------------------------------------------------------- #
# S1 — auth bootstrap

def test_first_message_authorizes_and_persists(bot):
    reply = bot.handle_first_contact(12345, username="aahan",
                                     first_name="Aahan")
    assert "authorized" in reply
    assert "Command Guide" in reply  # help sent automatically
    assert bot.is_authorized(12345)
    assert "TELEGRAM_CHAT_ID=12345" in bot._fake_secrets.read_text()
    state = bot.hermes.get_system_state(bot.AUTHORIZED_USERS_KEY)
    assert state["users"][0]["chat_id"] == 12345
    assert bot.alerter.chat_id == "12345"


def test_second_message_from_owner_is_normal(bot):
    bot.handle_first_contact(12345, first_name="A")
    reply = bot.handle_first_contact(12345, first_name="A")
    assert "Already authorized" in reply


def test_unknown_user_silently_rejected(bot):
    bot.handle_first_contact(12345, first_name="Owner")
    reply = bot.handle_first_contact(99999, first_name="Stranger")
    assert reply is None  # silent — bot's existence is not revealed
    assert not bot.is_authorized(99999)


def test_chat_id_survives_restart(bot, tmp_path):
    bot.handle_first_contact(12345)
    # A fresh alerter (new process) reloads the persisted chat_id lazily
    import src.alerts.telegram_bot as tg_mod
    fresh = TelegramAlerter(token="1:x", chat_id=None)
    with patch.object(tg_mod, "_load_secrets",
                      return_value={"TELEGRAM_CHAT_ID": "12345"}):
        fresh._reload_chat_id()
    assert fresh.chat_id == "12345"


# --------------------------------------------------------------------- #
# S2 — token rotation

def test_token_rotation(tmp_path, monkeypatch):
    import scripts.rotate_telegram_token as rot
    env = tmp_path / "secrets.env"
    env.write_text("TELEGRAM_BOT_TOKEN=111:oldtokenoldtokenoldtokenoldtoken0\n"
                   "OTHER=x\n")
    monkeypatch.setattr(rot, "ENV_PATH", env)
    new = "222:" + "n" * 35
    assert rot.rotate(new) == 0
    text = env.read_text()
    assert f"TELEGRAM_BOT_TOKEN={new}" in text
    assert "OTHER=x" in text


def test_token_rotation_rejects_bad_format(tmp_path, monkeypatch):
    import scripts.rotate_telegram_token as rot
    env = tmp_path / "secrets.env"
    env.write_text("TELEGRAM_BOT_TOKEN=old\n")
    monkeypatch.setattr(rot, "ENV_PATH", env)
    assert rot.rotate("not-a-token") == 1
    assert "old" in env.read_text()


# --------------------------------------------------------------------- #
# T1 — /help

def test_help_lists_all_commands(bot):
    text = bot.dispatch("help")
    for name in bot.COMMANDS:
        assert f"/{name}" in text, f"/{name} missing from help"


def test_start_is_help_alias(bot):
    assert bot.dispatch("start") == HELP_TEXT


# --------------------------------------------------------------------- #
# T2 — new commands

def test_trades_empty_and_populated(bot):
    assert "No trades" in bot.dispatch("trades")
    bot.hermes.brain.remember_trade({
        "ticker": "TCS.NS", "direction": "LONG", "entry_price": 4000,
        "quantity": 2, "entry_time": datetime.now(timezone.utc),
        "pnl": 150.0, "exit_price": 4075.0, "exit_reason": "TAKE_PROFIT",
        "signal_score": 10, "is_paper": True,
    })
    out = bot.dispatch("trades")
    assert "TCS.NS" in out and "+150" in out and "TAKE_PROFIT" in out


def test_signals_command(bot):
    bot.hermes.brain.remember_signal({
        "ticker": "INFY.NS", "direction": "LONG", "score": 7,
        "executed": False, "rejection_reason": "below gate",
        "signal_time": datetime.now(timezone.utc),
    })
    out = bot.dispatch("signals")
    assert "INFY.NS" in out and "below gate" in out


def test_regime_command(bot):
    bot.hermes.set_system_state("regime", {
        "state": "BULL_TRENDING", "confidence": 0.78,
        "history": [{"date": "2026-07-01", "regime": "SIDEWAYS"}],
    })
    out = bot.dispatch("regime")
    assert "BULL_TRENDING" in out and "78%" in out and "SIDEWAYS" in out


def test_macro_command(bot):
    bot.hermes.set_system_state("premarket", {
        "macro_bias": "POSITIVE", "global_cues": {"sp500": 0.0087}})
    out = bot.dispatch("macro")
    assert "POSITIVE" in out and "sp500" in out


def test_positions_alias(bot):
    assert bot.dispatch("positions") == bot.dispatch("open_positions")


def test_chat_uses_answer_question(bot):
    from src.knowledge import KnowledgeManager
    KnowledgeManager().store("TRADE_LESSON", "WIPRO gap-downs never recover "
                                             "intraday — avoid knife-catching")
    out = bot.dispatch("chat", "what", "about", "WIPRO", "gap-downs")
    assert "WIPRO" in out
    assert "regime" in out.lower()


# --------------------------------------------------------------------- #
# N1 — trade lifecycle notifications

class _QuoteFetcher:
    def __init__(self, price):
        self.price = price

    def get_live_quote(self, tickers):
        return pd.DataFrame([{"ticker": tickers[0],
                              "last_price": self.price}])


class _NullBrain:
    def remember_trade(self, *_a, **_k):
        return "x"


def test_paper_trade_open_and_close_notify():
    from src.execution.paper_trader import PaperTrader
    spy = SpyAlerter()
    trader = PaperTrader(brain=_NullBrain(), fetcher=_QuoteFetcher(100.0),
                         telegram=spy)
    trader.place_order("TCS.NS", "LONG", qty=1, price=100.0,
                       signal_hash="h1", score=10,
                       agent_votes={"technical": 3}, regime="BULL",
                       reasoning="council gate cleared")
    assert len(spy.sent) == 1
    assert "PAPER TRADE OPENED" in spy.sent[0]
    assert "10/12" in spy.sent[0] and "technical:3" in spy.sent[0]
    trader.fetcher.price = 105.0
    trader.manage_positions()
    assert len(spy.sent) == 2
    assert "PAPER TRADE CLOSED" in spy.sent[1]
    assert "TAKE_PROFIT" in spy.sent[1] and "%" in spy.sent[1]


def test_notification_failure_never_blocks_fill():
    from src.execution.paper_trader import PaperTrader

    class Exploder:
        def trade_opened(self, **_k):
            raise RuntimeError("telegram down")

    trader = PaperTrader(brain=_NullBrain(), fetcher=_QuoteFetcher(50.0),
                         telegram=Exploder())
    trade = trader.place_order("ITC.NS", "LONG", qty=1, price=50.0,
                               signal_hash="h2")
    assert trade["status"] == "FILLED"


# --------------------------------------------------------------------- #
# N2/N3/P2 — scheduled pushes

@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/pushes.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    spy = SpyAlerter()
    h = HermesCoordinator(brain=QuNtraBrain(), trader=StubTrader(),
                          telegram=spy, research_team={})
    yield h, spy
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_morning_briefing_structure(hermes_env):
    h, spy = hermes_env
    h.set_system_state("premarket", {
        "watchlist": ["TCS.NS", "INFY.NS"], "macro_bias": "POSITIVE",
        "earnings_blackout": ["WIPRO.NS"]})
    msg = h.send_morning_briefing()
    assert "Morning Briefing" in msg
    assert "TCS.NS" in msg and "POSITIVE" in msg
    assert "WIPRO.NS" in msg and "NOT be traded" in msg
    assert spy.sent and spy.sent[-1] == msg


def test_arm_system_sends_briefing(hermes_env):
    h, spy = hermes_env
    h.arm_system()
    assert any("Morning Briefing" in m for m in spy.sent)
    assert (h.get_system_state("oms") or {}).get("enabled")


def test_eod_summary_zero_trades(hermes_env):
    h, spy = hermes_env
    msg = h.send_eod_report()
    assert "No trades today" in msg
    assert "Paper gate" in msg


def test_weekly_paper_recap(hermes_env):
    h, spy = hermes_env
    msg = h.send_weekly_paper_recap()
    assert "Weekly Paper Trading Recap" in msg
    assert "day 0/40" in msg or "Days remaining" in msg


def test_eod_summary_with_trades(hermes_env):
    h, spy = hermes_env
    h.brain.remember_trade({
        "ticker": "SUNPHARMA.NS", "direction": "LONG", "entry_price": 1500,
        "quantity": 5, "entry_time": datetime.now(timezone.utc),
        "exit_time": datetime.now(timezone.utc), "pnl": 320.0,
        "exit_reason": "TAKE_PROFIT", "is_paper": True,
    })
    msg = h.send_eod_report()
    assert "1 trades (1W/0L)" in msg
    assert "+320" in msg
    assert "/daily_report" in msg


def test_weekly_recap_congratulates_on_gate_pass(hermes_env, monkeypatch):
    h, spy = hermes_env
    import src.reporting.metrics as M
    monkeypatch.setattr(M, "daily_pnl_series",
                        lambda db, days=120: {f"d{i}": 10.0
                                              for i in range(42)})
    monkeypatch.setattr(M, "rolling_sharpe", lambda db, **k: 1.5)
    monkeypatch.setattr(M, "max_drawdown_from_pnl", lambda db, **k: -0.05)
    msg = h.send_weekly_paper_recap()
    assert "PAPER GATE PASSED" in msg
    assert "/start_live" in msg


def test_alerter_send_reloads_persisted_chat_id(monkeypatch):
    import src.alerts.telegram_bot as tg_mod
    alerter = TelegramAlerter(token="1:x", chat_id=None)
    assert alerter.test_mode  # no chat_id yet
    monkeypatch.setattr(tg_mod, "_load_secrets",
                        lambda: {"TELEGRAM_CHAT_ID": "777"})
    # send() picks up the persisted chat_id lazily; the actual network
    # call is beyond test scope — verify the reload path only
    alerter._reload_chat_id()
    assert alerter.chat_id == "777"
    assert not alerter.test_mode


# --------------------------------------------------------------------- #
# N4 — risk alerts

def test_circuit_levels_alert():
    spy = SpyAlerter()
    cb = DrawdownCircuitBreaker(telegram=spy)
    now = datetime.now()
    cb.update(intraday_dd=-0.031, rolling_5d_dd=-0.01, now=now)
    assert "Level 1" in spy.sent[-1] and "/risk" in spy.sent[-1]
    cb.update(intraday_dd=-0.046, rolling_5d_dd=-0.01, now=now)
    assert "Level 2" in spy.sent[-1] and "/resume" in spy.sent[-1]
    cb.update(intraday_dd=-0.046, rolling_5d_dd=-0.08, now=now)
    assert "Level 3" in spy.sent[-1] and "/resume" in spy.sent[-1]
    assert len(spy.sent) == 3


def test_kill_switch_message_has_analysis(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/guard.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    spy = SpyAlerter()
    guard = ConsecutiveLossGuard(brain=QuNtraBrain(), telegram=spy,
                                 oms=StubTrader())
    for pnl in (-100, -50, -75):
        guard.record_trade_outcome(pnl, {"ticker": "TCS.NS", "pnl": pnl})
    assert guard.halted
    msg = spy.sent[-1]
    assert "KILL SWITCH" in msg and "/resume" in msg
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


# --------------------------------------------------------------------- #
# W1 — watchdog

def test_watchdog_detects_dead_pid(tmp_path):
    from scripts.watchdog import pid_alive
    dead = tmp_path / "dead.pid"
    dead.write_text("999999")  # PID that cannot exist
    assert pid_alive(dead) is False
    missing = tmp_path / "missing.pid"
    assert pid_alive(missing) is False
    import os
    alive = tmp_path / "alive.pid"
    alive.write_text(str(os.getpid()))
    assert pid_alive(alive) is True


# --------------------------------------------------------------------- #
# P1 — --telegram status output

def test_status_telegram_output_no_trades():
    from scripts.paper_trading_status import telegram_output
    out = telegram_output(None)
    assert "day 0/40" in out or "No trades" in out


def test_status_telegram_output_with_stats():
    from scripts.paper_trading_status import telegram_output
    st = {"days": 12, "total_pnl": 1500.0, "wins": 8, "losses": 4,
          "win_rate": 8 / 12, "sharpe": 1.4, "max_dd": -0.03,
          "n_closed": 12, "n_entered": 14,
          "gate_days": False, "gate_sharpe": True, "gate_dd": True}
    out = telegram_output(st)
    assert "Day 12/40" in out and "1.400" in out and "IN PROGRESS" in out
