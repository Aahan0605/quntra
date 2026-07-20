#!/usr/bin/env python3
"""
QuNtra 24/7 scheduler — all times IST, NSE-holiday aware.

    python3 scripts/scheduler.py           # run forever
    python3 scripts/scheduler.py --dry-run # verify all jobs fire correctly

Jobs (IST):
    06:00  pre-market sequence          14:30  close management
    08:45  arm system                   15:30  post-market sequence
    09:15  observe market open          17:00  EOD Telegram report
    09:30  start trading session        18:00  overnight research prep
    09:30–14:29 every 60s market loop   22:00  overnight batch
                                        03:00  health check
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db import init_db  # noqa: E402
from src.governor.brain import QuNtraBrain  # noqa: E402
from src.governor.hermes import HermesCoordinator  # noqa: E402
from src.risk.drawdown_circuit import DrawdownCircuitBreaker  # noqa: E402
from src.risk.consecutive_loss_guard import ConsecutiveLossGuard  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("quntra.scheduler")

IST = pytz.timezone("Asia/Kolkata")
NSE_CAL = mcal.get_calendar("NSE")


def is_trading_day(when: datetime | None = None) -> bool:
    today = (when or pd.Timestamp.now(tz="Asia/Kolkata")).date()
    schedule = NSE_CAL.schedule(start_date=today, end_date=today)
    return not schedule.empty


def trading_day_only(fn):
    """Decorator: run the job only on NSE trading days."""
    def wrapper(*args, **kwargs):
        if not is_trading_day():
            logger.info("Not an NSE trading day — skipping %s", fn.__name__)
            return None
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def build_hermes() -> HermesCoordinator:
    init_db()
    brain = QuNtraBrain()

    from src.utils.data_fetcher import UnifiedDataFetcher
    fetcher = UnifiedDataFetcher()

    telegram = None
    try:
        from src.alerts.telegram_bot import TelegramAlerter
        telegram = TelegramAlerter.from_config()
    except Exception as e:  # noqa: BLE001
        logger.warning("Telegram unavailable: %s", e)

    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    paper = os.environ.get("PAPER_TRADE", "true").lower() != "false"
    if paper:
        from src.execution.paper_trader import PaperTrader
        trader = PaperTrader(brain=brain, fetcher=fetcher, telegram=telegram)
    else:
        from src.execution.kite_oms import KiteOMS
        trader = KiteOMS(brain=brain)

    from src.governor.council import SignalCouncil
    capital = float(os.environ.get("DAILY_CAPITAL_INR", "25000"))
    council = SignalCouncil(capital=capital)

    return HermesCoordinator(
        brain=brain, trader=trader, fetcher=fetcher, telegram=telegram,
        circuit_breaker=DrawdownCircuitBreaker(telegram=telegram),
        loss_guard=ConsecutiveLossGuard(brain=brain, telegram=telegram,
                                        oms=trader),
        council=council,
    )


def register_jobs(scheduler: BlockingScheduler, hermes: HermesCoordinator):
    jobs = [
        ("pre_market", hermes.run_pre_market_sequence, dict(hour=6, minute=0)),
        ("arm_system", hermes.arm_system, dict(hour=8, minute=45)),
        ("observe_open", hermes.observe_market_open, dict(hour=9, minute=15)),
        ("start_session", hermes.start_trading_session, dict(hour=9, minute=30)),
        ("market_loop", hermes.run_market_session,
         dict(minute="*/1", hour="9-14")),
        ("close_mgmt", hermes.begin_close_management, dict(hour=14, minute=30)),
        ("post_market", hermes.run_post_market_sequence, dict(hour=15, minute=30)),
        ("eod_report", hermes.send_eod_report, dict(hour=17, minute=0)),
        ("overnight_research", hermes.start_overnight_research,
         dict(hour=18, minute=0)),
        ("overnight_batch", hermes.run_overnight_batch, dict(hour=22, minute=0)),
        ("health_check", hermes.health_check, dict(hour=3, minute=0)),
        ("weekly_board_report", hermes.generate_weekly_board_report,
         dict(day_of_week="sun", hour=20, minute=0)),
        ("monthly_letter", hermes.generate_monthly_investment_letter,
         dict(day=1, hour=9, minute=0)),
        ("weekly_paper_recap", hermes.send_weekly_paper_recap,
         dict(day_of_week="fri", hour=18, minute=0)),
        # Runs every day (not gated by trading_day_only): it only reads the
        # DB and no-ops until day 40 is reached, then sends the full
        # day-by-day history exactly once. 17:10 = just after eod_report
        # settles the day's trade rows.
        ("gate_completion_check", hermes.check_gate_completion,
         dict(hour=17, minute=10)),
        # Mirror the DB into the Obsidian vault every evening after the
        # day's trades and reports have settled. Read-only; runs daily
        # (weekends too) so the vault's report archive stays current.
        ("obsidian_sync", hermes.sync_obsidian, dict(hour=17, minute=20)),
        # Kite tokens expire ~07:30 IST daily — check just after and ping
        # the operator if a re-login is needed. Every day; no-op alert when
        # Kite isn't configured, so it never nags during pure paper trading.
        ("kite_token_check", hermes.check_kite_token, dict(hour=7, minute=45)),
    ]
    market_hour_jobs = {"pre_market", "arm_system", "observe_open",
                        "start_session", "market_loop", "close_mgmt",
                        "post_market", "eod_report"}
    for job_id, fn, cron in jobs:
        target = trading_day_only(fn) if job_id in market_hour_jobs else fn
        scheduler.add_job(target, CronTrigger(timezone=IST, **cron),
                          id=job_id, name=job_id, misfire_grace_time=300,
                          coalesce=True, max_instances=1)
    return [j[0] for j in jobs]


def dry_run() -> int:
    """Verify: jobs register, IST times correct, holiday check works."""
    class _Stub:
        def __getattr__(self, name):
            return lambda *a, **k: None

    scheduler = BlockingScheduler(timezone=IST)
    hermes = _Stub()
    ids = register_jobs(scheduler, hermes)
    assert len(ids) == 17, f"expected 17 jobs, got {len(ids)}"
    for job in scheduler.get_jobs():
        nxt = job.trigger.get_next_fire_time(None, datetime.now(IST))
        assert nxt is not None, f"job {job.id} would never fire"
        assert str(nxt.tzinfo) in ("Asia/Kolkata", "IST"), \
            f"job {job.id} not IST: {nxt.tzinfo}"
        print(f"  {job.id:20s} next fire {nxt}")

    # NSE holiday check — Diwali (Laxmi Pujan) is a closed day.
    # 2026-11-08 is a Sunday; use Republic Day 2026 (Mon Jan 26) as the
    # deterministic closed-weekday probe, plus a known open day.
    closed = pd.Timestamp("2026-01-26", tz="Asia/Kolkata")  # Republic Day
    open_day = pd.Timestamp("2026-07-03", tz="Asia/Kolkata")  # regular Friday
    assert not is_trading_day(closed), "Republic Day should be closed"
    assert is_trading_day(open_day), "2026-07-03 should be a trading day"
    print("  holiday calendar OK (Republic Day closed, regular Friday open)")
    print("DRY RUN PASSED — 17 jobs registered, IST-correct, holiday-aware")
    print("--dry-run complete: 17/17 jobs passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--env", default=None,
                    help="extra env file to load (e.g. config/secrets.env)")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    if args.env:
        from dotenv import load_dotenv
        load_dotenv(args.env)
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dry_run:
        return dry_run()

    hermes = build_hermes()
    scheduler = BlockingScheduler(timezone=IST)
    register_jobs(scheduler, hermes)

    # /health reads system_state["last_job_run"] to prove the scheduler
    # is not just alive but actually firing jobs.
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

    def _record_job(event):
        try:
            hermes.set_system_state("last_job_run", {
                "name": event.job_id,
                "at": datetime.now(IST).isoformat(),
                "error": str(event.exception) if getattr(
                    event, "exception", None) else None,
            })
        except Exception:  # noqa: BLE001 — bookkeeping never kills a job
            logger.exception("could not record last_job_run")

    scheduler.add_listener(_record_job, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    logger.info("QuNtra scheduler starting — %d jobs, timezone IST",
                len(scheduler.get_jobs()))
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
