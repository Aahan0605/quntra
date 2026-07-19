"""Tests for GateCompletionReport and its Hermes/Telegram wiring."""

from datetime import datetime, timedelta, timezone

import pytest

import src.db.session as db_session
from src.db import Base, Trade, get_engine, get_session, init_db
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator
from src.reporting import GateCompletionReport


class StubTrader:
    def get_positions(self):
        return []


class SpyTelegram:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'gate.db'}"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    yield url
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def _add_trade(db_url, day_offset, pnl):
    when = datetime.now(timezone.utc) - timedelta(days=day_offset)
    with get_session(db_url) as s:
        s.add(Trade(
            ticker="TCS.NS", direction="LONG", quantity=1,
            entry_price=100.0, exit_price=100.0 + pnl,
            entry_time=when, exit_time=when, pnl=pnl,
            is_paper=True,
        ))


def test_no_trades_returns_none(db_url):
    assert GateCompletionReport(db_url).generate() is None


def test_generate_produces_day_by_day_log(db_url):
    _add_trade(db_url, 2, 100.0)
    _add_trade(db_url, 1, -50.0)
    report = GateCompletionReport(db_url).generate()
    assert "GATE RESULT" in report
    assert "DAY-BY-DAY LOG" in report
    assert "PENDING (2/40)" in report  # only 2 trading days so far
    assert "₹+100" in report or "+100" in report
    assert "₹-50" in report or "-50" in report


def test_gate_not_reached_before_40_days(db_url):
    _add_trade(db_url, 0, 10.0)
    gcr = GateCompletionReport(db_url, telegram=SpyTelegram())
    sent = gcr.send_if_gate_reached()
    assert sent is False


def test_gate_reached_sends_once(db_url, monkeypatch):
    for d in range(40):
        _add_trade(db_url, d, 5.0)
    spy = SpyTelegram()
    gcr = GateCompletionReport(db_url, telegram=spy)

    first = gcr.send_if_gate_reached()
    assert first is True
    assert len(spy.sent) == 1
    assert "COMPLETE" in spy.sent[0] or "40 DAYS REACHED" in spy.sent[0]

    # Second call (e.g. scheduler restart) must NOT resend
    second = gcr.send_if_gate_reached()
    assert second is False
    assert len(spy.sent) == 1


def test_hermes_check_gate_completion_wires_through(db_url):
    for d in range(40):
        _add_trade(db_url, d, 1.0)
    spy = SpyTelegram()
    h = HermesCoordinator(brain=QuNtraBrain(), trader=StubTrader(),
                          telegram=spy, db_url=db_url, research_team={})
    assert h.check_gate_completion() is True
    assert len(spy.sent) == 1
    # idempotent on repeated calls (e.g. daily job re-firing)
    assert h.check_gate_completion() is False


def test_hermes_generate_gate_report_now_works_anytime(db_url):
    _add_trade(db_url, 0, 25.0)
    h = HermesCoordinator(brain=QuNtraBrain(), trader=StubTrader(),
                          db_url=db_url, research_team={})
    report = h.generate_gate_report_now()
    assert "DAY-BY-DAY LOG" in report


def test_generate_gate_report_now_handles_empty_db(db_url):
    h = HermesCoordinator(brain=QuNtraBrain(), trader=StubTrader(),
                          db_url=db_url, research_team={})
    assert "No paper trades" in h.generate_gate_report_now()


def test_telegram_command_gate_report(db_url):
    from src.alerts.telegram_bot import QuNtraTelegramBot, TelegramAlerter
    _add_trade(db_url, 0, 15.0)
    h = HermesCoordinator(brain=QuNtraBrain(), trader=StubTrader(),
                          db_url=db_url, research_team={})
    bot = QuNtraTelegramBot(h, alerter=TelegramAlerter(test_mode=True),
                            db_url=db_url)
    assert "gate_report" in bot.COMMANDS
    out = bot.dispatch("gate_report")
    assert "DAY-BY-DAY LOG" in out
