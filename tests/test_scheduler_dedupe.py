"""dedupe_cron_fire — two scheduler processes starting within the same
cron-eligible minute (e.g. a watchdog restart racing a manual one) each
independently decide a job is due and both fire it. Happened for real:
pre_market ran twice, 2 seconds apart, on 2026-07-29.

Must only suppress the CRON path's re-fire — a manual trigger (e.g.
/start_trading calling hermes.run_pre_market_sequence() directly) never
goes through this wrapper, so it's untouched by design; nothing here
tests that path because nothing here should ever call it.
"""

from datetime import datetime, timedelta, timezone

from scripts.scheduler import CRON_DEDUPE_WINDOW_SECONDS, dedupe_cron_fire


class FakeHermes:
    """DB-backed system_state, shared across "processes" — the same
    contract dedupe_cron_fire relies on to catch a cross-process race."""

    def __init__(self):
        self.state: dict = {}

    def get_system_state(self, key):
        return self.state.get(key)

    def set_system_state(self, key, value):
        self.state[key] = value


def test_second_fire_within_window_is_skipped():
    hermes = FakeHermes()
    calls = []
    fn = dedupe_cron_fire(lambda: calls.append(1), "pre_market", hermes)

    fn()
    fn()  # simulates a second process racing the first

    assert calls == [1]


def test_fire_outside_the_window_runs_again():
    hermes = FakeHermes()
    calls = []
    fn = dedupe_cron_fire(lambda: calls.append(1), "pre_market", hermes)
    fn()

    # Backdate the recorded fire past the window, as if real time had passed.
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=CRON_DEDUPE_WINDOW_SECONDS + 1)
    hermes.state["cron_fired_pre_market"] = {"at": stale.isoformat()}
    fn()

    assert calls == [1, 1]


def test_different_job_ids_do_not_interfere():
    hermes = FakeHermes()
    calls = []
    a = dedupe_cron_fire(lambda: calls.append("a"), "pre_market", hermes)
    b = dedupe_cron_fire(lambda: calls.append("b"), "arm_system", hermes)

    a()
    b()

    assert calls == ["a", "b"]


def test_a_direct_manual_call_bypasses_dedupe_entirely():
    """The whole point: /start_trading calls hermes.run_pre_market_sequence()
    directly, never through dedupe_cron_fire — so a cron fire and a manual
    fire in the same minute must NOT suppress each other. Simulated here by
    simply not wrapping the manual call, which is exactly what the real
    Telegram handler does."""
    hermes = FakeHermes()
    calls = []
    cron_wrapped = dedupe_cron_fire(lambda: calls.append("cron"),
                                    "pre_market", hermes)

    cron_wrapped()
    calls.append("manual")  # the unwrapped path — always runs
    cron_wrapped()          # a second cron fire in the same window — skipped

    assert calls == ["cron", "manual"]


def test_bad_stored_timestamp_does_not_crash_and_still_runs():
    hermes = FakeHermes()
    hermes.state["cron_fired_pre_market"] = {"at": "not-a-timestamp"}
    calls = []
    fn = dedupe_cron_fire(lambda: calls.append(1), "pre_market", hermes)
    fn()
    assert calls == [1]
