"""Sleep-gap self-healing: Hermes.handle_missed_job — offline, mocked."""

import pytest

import src.db.session as db_session
from src.db import init_db
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator


class StubTrader:
    def get_positions(self):
        return []


class SpyTelegram:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/missed.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    h = HermesCoordinator(brain=QuNtraBrain(), trader=StubTrader(),
                         telegram=SpyTelegram(), research_team={},
                         db_url=url)
    yield h
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_alerts_on_any_missed_job(hermes):
    hermes.handle_missed_job("market_loop", "2026-07-23T11:53:00+05:30")
    assert len(hermes.telegram.sent) == 1
    assert "market_loop" in hermes.telegram.sent[0]
    assert "missed" in hermes.telegram.sent[0].lower()


def test_no_catchup_for_non_catchup_job(hermes):
    hermes.handle_missed_job("market_loop", "2026-07-23T11:53:00+05:30")
    assert "No catch-up" in hermes.telegram.sent[0]


def test_catchup_runs_for_eod_report(hermes, monkeypatch):
    called = []
    monkeypatch.setattr(hermes, "send_eod_report",
                        lambda: called.append(True))
    hermes.handle_missed_job("eod_report", "2026-07-23T17:00:00+05:30")
    assert called == [True]
    assert "catch-up" in hermes.telegram.sent[0].lower()


def test_catchup_runs_once_per_day(hermes, monkeypatch):
    calls = []
    monkeypatch.setattr(hermes, "send_eod_report",
                        lambda: calls.append(1))
    hermes.handle_missed_job("eod_report", "2026-07-23T17:00:00+05:30")
    hermes.handle_missed_job("eod_report", "2026-07-23T17:05:00+05:30")
    assert len(calls) == 1  # second misfire event doesn't re-run it


def test_market_hour_catchup_skipped_on_non_trading_day(hermes, monkeypatch):
    called = []
    monkeypatch.setattr(hermes, "begin_close_management",
                        lambda: called.append(True))
    monkeypatch.setattr("scripts.scheduler.is_trading_day", lambda *a: False)
    hermes.handle_missed_job("close_mgmt", "2026-07-19T14:30:00+05:30")
    assert called == []


def test_market_hour_catchup_runs_on_trading_day(hermes, monkeypatch):
    called = []
    monkeypatch.setattr(hermes, "begin_close_management",
                        lambda: called.append(True))
    monkeypatch.setattr("scripts.scheduler.is_trading_day", lambda *a: True)
    hermes.handle_missed_job("close_mgmt", "2026-07-23T14:30:00+05:30")
    assert called == [True]


def test_catchup_failure_does_not_raise(hermes, monkeypatch):
    def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(hermes, "sync_obsidian", boom)
    hermes.handle_missed_job("obsidian_sync", "2026-07-23T17:20:00+05:30")  # must not raise
    assert len(hermes.telegram.sent) == 1  # alert still sent


def test_no_telegram_configured_never_raises():
    h = HermesCoordinator(brain=QuNtraBrain(), trader=StubTrader(),
                          telegram=None, research_team={})
    h.handle_missed_job("market_loop", "2026-07-23T11:53:00+05:30")  # no crash


def test_pre_market_and_arm_system_have_catchup(hermes, monkeypatch):
    """A slept-through morning must not silently cancel the trading day.
    Without catch-up for these two, pre_market never builds the watchlist
    and arm_system never enables the OMS — everything downstream then
    no-ops against an empty watchlist and nothing looks broken.
    """
    from src.governor.hermes import HermesCoordinator
    for job in ("pre_market", "arm_system"):
        assert job in HermesCoordinator._CATCHUP_JOBS, f"{job} has no catch-up"
        # market-hour gated: a catch-up must not fire on a holiday/weekend
        assert job in HermesCoordinator._MARKET_HOUR_CATCHUP
