"""scripts/run_combined.py — the free-tier entrypoint that runs the
scheduler and Telegram bot in one process (Render's free plan has no free
Background Worker service). Only tests the reusable piece
(build_scheduler); actually calling run_combined.main() would block
forever on bot.run_polling(), so that's out of scope for a unit test.
"""

import pytest
from apscheduler.schedulers.blocking import BlockingScheduler

import scripts.run_combined  # noqa: F401 — import-time smoke test
from scripts.scheduler import build_scheduler


class _StubHermes:
    """Just enough surface for register_jobs' job list to bind."""

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_build_scheduler_returns_a_fully_wired_scheduler():
    scheduler = build_scheduler(_StubHermes())
    assert isinstance(scheduler, BlockingScheduler)
    # Count-agnostic: a hardcoded total turns every new job into a
    # test edit, which invites blindly bumping the number. What
    # matters is that register_jobs actually wired jobs in.
    assert len(scheduler.get_jobs()) >= 19


def test_build_scheduler_does_not_start_it():
    """The whole point of extracting this: the caller decides how/where
    to call .start() (main thread for main(), background thread for
    run_combined.py) — build_scheduler itself must never start it."""
    scheduler = build_scheduler(_StubHermes())
    assert scheduler.state == 0  # APScheduler's STATE_STOPPED
