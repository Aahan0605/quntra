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
import time
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


# Two scheduler processes starting within the same cron-eligible minute
# (e.g. a watchdog restart racing a manual one) each independently decide
# a job is due and both fire it — happened for real: pre_market ran twice
# 2 seconds apart on 2026-07-29. system_state is shared across processes,
# so it's the only thing that can catch a cross-process race; an in-memory
# guard on one process can't see the other process's timer.
CRON_DEDUPE_WINDOW_SECONDS = 180


def dedupe_cron_fire(fn, job_id: str, hermes: HermesCoordinator):
    """Decorator: skip this cron-triggered call if the same job_id already
    fired within CRON_DEDUPE_WINDOW_SECONDS, regardless of which process
    fired it. Wraps the CRON registration only — manual triggers (e.g.
    /start_trading calling hermes.run_pre_market_sequence() directly)
    never go through this wrapper, so they always run on demand."""
    def wrapper(*args, **kwargs):
        key = f"cron_fired_{job_id}"
        now = datetime.now(IST)
        last = (hermes.get_system_state(key) or {}).get("at")
        if last:
            try:
                elapsed = (now - datetime.fromisoformat(last)).total_seconds()
                if elapsed < CRON_DEDUPE_WINDOW_SECONDS:
                    logger.info("skipping duplicate cron fire for %s "
                               "(last ran %ds ago)", job_id, int(elapsed))
                    return None
            except ValueError:
                pass
        hermes.set_system_state(key, {"at": now.isoformat()})
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
    capital = float(os.environ.get("DAILY_CAPITAL_INR", "25000"))
    paper = os.environ.get("PAPER_TRADE", "true").lower() != "false"
    if paper:
        from src.execution.paper_trader import PaperTrader
        # cash used to be hardcoded at ₹25k, so raising DAILY_CAPITAL_INR
        # sized the orders up but left the book unable to pay for them.
        trader = PaperTrader(brain=brain, fetcher=fetcher, telegram=telegram,
                             starting_cash=capital)
    else:
        from src.execution.kite_oms import KiteOMS
        trader = KiteOMS(brain=brain)

    from src.governor.council import SignalCouncil
    council = SignalCouncil(capital=capital)

    from src.portfolio.live_allocator import PassiveAllocator
    from src.utils.universe import UNIVERSE
    allocator = PassiveAllocator(universe=UNIVERSE, trader=trader,
                                 capital=capital)

    return HermesCoordinator(
        brain=brain, trader=trader, fetcher=fetcher, telegram=telegram,
        circuit_breaker=DrawdownCircuitBreaker(telegram=telegram),
        loss_guard=ConsecutiveLossGuard(brain=brain, telegram=telegram,
                                        oms=trader),
        council=council, allocator=allocator,
    )


HEARTBEAT_FILE = ROOT / "quntra.heartbeat"


def _write_heartbeat() -> None:
    """Prove the scheduler's event loop is still turning.

    A hung-but-alive process keeps its PID, so a PID check alone reports
    a healthy system while nothing runs (this exact failure went unnoticed
    for five days). The watchdog treats a stale heartbeat as death.
    """
    HEARTBEAT_FILE.write_text(str(int(time.time())))


def register_jobs(scheduler: BlockingScheduler, hermes: HermesCoordinator):
    jobs = [
        ("heartbeat", _write_heartbeat, dict(minute="*")),
        ("pre_market", hermes.run_pre_market_sequence, dict(hour=6, minute=0)),
        ("arm_system", hermes.arm_system, dict(hour=8, minute=45)),
        ("observe_open", hermes.observe_market_open, dict(hour=9, minute=15)),
        ("start_session", hermes.start_trading_session, dict(hour=9, minute=30)),
        # The live strategy (docs/CEO_REVIEW.md Path A) — inverse-vol
        # weights, vetoes applied, crash-scaled. Runs Monday pre-open;
        # Rebalancer's own weekly/3%-drift/20%-turnover gates decide
        # whether anything actually trades.
        ("allocator_rebalance", hermes.run_allocator_rebalance,
         dict(day_of_week="mon", hour=9, minute=20)),
        ("market_loop", hermes.run_market_session,
         dict(minute="*/1", hour="9-14")),
        ("close_mgmt", hermes.begin_close_management, dict(hour=14, minute=30)),
        ("post_market", hermes.run_post_market_sequence, dict(hour=15, minute=30)),
        ("eod_report", hermes.send_eod_report, dict(hour=17, minute=0)),
        # 20:00 — after post_market (15:30) and eod_report (17:00), so the
        # screen sees the day's realised results before planning tomorrow.
        ("deep_screen", hermes.run_deep_screen, dict(hour=20, minute=0)),
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
        # ICICI Breeze session tokens expire daily — check ahead of market
        # open and ping the operator if a re-login is needed. Every day;
        # no-op alert when Breeze isn't configured, so it never nags during
        # pure paper trading.
        ("breeze_token_check", hermes.check_breeze_token,
         dict(hour=7, minute=45)),
    ]
    market_hour_jobs = {"pre_market", "arm_system", "observe_open",
                        "start_session", "market_loop", "close_mgmt",
                        "post_market", "eod_report", "allocator_rebalance"}
    # heartbeat/market_loop fire every minute by design — a dedupe window
    # would suppress their normal cadence, not just a genuine double-fire.
    # Every other job runs at most once a day, so a 3-minute window can
    # only ever catch the cross-process race, never a legitimate re-fire.
    NO_DEDUPE = {"heartbeat", "market_loop"}
    for job_id, fn, cron in jobs:
        target = trading_day_only(fn) if job_id in market_hour_jobs else fn
        if job_id not in NO_DEDUPE:
            target = dedupe_cron_fire(target, job_id, hermes)
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
    assert len(ids) == 20, f"expected 20 jobs, got {len(ids)}"
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
    print(f"DRY RUN PASSED — {len(ids)} jobs registered, IST-correct, "
         f"holiday-aware")
    print(f"--dry-run complete: {len(ids)}/{len(ids)} jobs passed")
    return 0


def _start_keepalive(interval: int = 600) -> None:
    """Ping our own public URL so the free tier never idles us out.

    Render's free plan spins a web service down after 15 minutes with no
    INBOUND traffic, and APScheduler fires nothing while it is down — a
    06:00 pre-market and a 09:15 open would simply never happen, silently.
    That made the whole schedule depend on an external uptime pinger
    nobody can see from in here.

    ponytail: this prevents idling, it cannot wake a service that has
    already spun down (the process is gone; nothing is left to ping). It
    holds as long as the service is up when this starts. Keep the external
    pinger as the belt to this pair of braces — or move off the free tier,
    where the problem does not exist at all.
    """
    import os
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    import threading
    import urllib.request

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=30)
            except Exception as e:  # noqa: BLE001 — never kill the thread
                logger.warning("keepalive ping failed: %s", e)

    threading.Thread(target=_loop, daemon=True, name="keepalive").start()
    logger.info("Keepalive pinging %s every %ds (free tier sleeps at 15min "
                "idle, which would silently skip every scheduled job)",
                url, interval)


def _status_token_ok(supplied: str) -> bool:
    """Constant-time check of /status's token against the bot token's hash."""
    import hashlib
    import hmac
    import os
    secret = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not secret or not supplied:
        return False
    expected = hashlib.sha256(secret.encode()).hexdigest()
    return hmac.compare_digest(expected, supplied)


def _start_healthcheck_server(max_stale_seconds: int = 300) -> None:
    """Minimal HTTP healthcheck for cloud platforms (Render, Railway, etc.).

    A platform's restart policy only detects a process EXIT, not a hang —
    exactly the failure mode that went undetected for 5 days on the Mac
    (caught there by watchdog.py polling this same heartbeat file; there
    is no watchdog process in a cloud deploy). Only starts when PORT is
    set, so local Mac runs are completely unaffected.

    On Render's free tier this endpoint does double duty: an external
    uptime pinger hits it every 5 min to stop the service sleeping after
    15 min idle, which would otherwise silently halt all trading jobs.

    /status?token=... additionally returns the gate dashboard as JSON.
    The deployed service was previously a black box: its Postgres is only
    reachable with the platform-injected connection string, so there was
    no way to see a single trade without opening the dashboard.
    """
    import os
    port = os.environ.get("PORT")
    if not port:
        return
    import http.server
    import json
    import threading
    from urllib.parse import parse_qs, urlparse

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's naming
            if urlparse(self.path).path == "/status":
                return self._status(parse_qs(urlparse(self.path).query))
            try:
                age = int(time.time()) - int(HEARTBEAT_FILE.read_text().strip())
            except (FileNotFoundError, ValueError):
                age = None
            healthy = age is not None and age < max_stale_seconds
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"heartbeat_age={age}".encode())

        def _status(self, query: dict) -> None:
            """Gate dashboard as JSON. Public URL — so it is token-gated.

            The token is sha256(TELEGRAM_BOT_TOKEN): a secret this service
            already has, so nothing new goes in the dashboard, and the bot
            token itself never travels over the wire. No token configured
            means no endpoint, rather than an open one.
            """
            if not _status_token_ok(query.get("token", [""])[0]):
                self.send_response(404)
                self.end_headers()
                return
            try:
                from scripts.paper_trading_status import gather_stats
                body = json.dumps(gather_stats() or {"days": 0},
                                  default=str).encode()
                code = 200
            except Exception as e:  # noqa: BLE001 — never 500 the pinger's host
                body = json.dumps({"error": str(e)}).encode()
                code = 503
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # noqa: A002 — silence per-request noise
            pass

    server = http.server.HTTPServer(("0.0.0.0", int(port)), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Healthcheck server listening on :%s (platform restarts on "
               "a stale/failed check)", port)


def build_scheduler(hermes: HermesCoordinator) -> BlockingScheduler:
    """Fully wired scheduler — jobs, missed/executed listeners, the /health
    server — everything except actually calling .start(). Shared by
    main() (which starts it in the main thread) and
    scripts/run_combined.py (which starts it in a background thread so
    the Telegram bot's polling loop can own the main thread instead —
    built for Render's free tier, which has no free Background Worker
    service, so both processes have to share one).
    """
    scheduler = BlockingScheduler(timezone=IST)
    register_jobs(scheduler, hermes)

    # /health reads system_state["last_job_run"] to prove the scheduler
    # is not just alive but actually firing jobs.
    from apscheduler.events import (EVENT_JOB_ERROR, EVENT_JOB_EXECUTED,
                                    EVENT_JOB_MISSED)

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

    def _record_missed(event):
        try:
            hermes.handle_missed_job(
                event.job_id, event.scheduled_run_time.isoformat())
        except Exception:  # noqa: BLE001 — bookkeeping never kills a job
            logger.exception("could not handle missed job %s", event.job_id)

    scheduler.add_listener(_record_job, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    scheduler.add_listener(_record_missed, EVENT_JOB_MISSED)
    logger.info("QuNtra scheduler wired — %d jobs, timezone IST",
                len(scheduler.get_jobs()))
    return scheduler


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

    # After env load and handler setup, so it sees live values and covers
    # every handler including the log file.
    from src.utils.log_redaction import install_redaction
    install_redaction()

    if args.dry_run:
        return dry_run()

    hermes = build_hermes()
    scheduler = build_scheduler(hermes)
    _start_healthcheck_server()
    _start_keepalive()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
