#!/usr/bin/env python3
"""
QuNtra combined entrypoint — scheduler + Telegram bot in ONE process.

Built specifically for Render's free tier: free plans only allow Web
Service type (no free Background Worker), and free services sleep after
15 minutes with no inbound HTTP traffic. Running both here means only one
(free) service is needed, and an external pinger hitting this process's
/health endpoint (e.g. UptimeRobot, every ~10 min) keeps the whole thing —
scheduler and bot together — from sleeping.

Trade-off, stated plainly: scripts/scheduler.py and
scripts/run_telegram_bot.py are deliberately SEPARATE processes locally
so a bot crash never touches trading and vice versa. This combined
entrypoint gives that isolation up to fit the free tier — a real
trade-off, not a strict improvement. If Render billing ever gets set up,
switch back to the two-service render.yaml Blueprint instead.

    nohup ./venv/bin/python scripts/run_combined.py >> logs/combined.log 2>&1 &
"""
import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("quntra.combined")


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / "secrets.env")
    load_dotenv(ROOT / ".env")

    # After load_dotenv so the filter knows the live secret values.
    from src.utils.log_redaction import install_redaction
    install_redaction()

    from scripts.scheduler import (_start_healthcheck_server,
                                   _start_keepalive, build_hermes,
                                   build_scheduler)
    from src.alerts.telegram_bot import QuNtraTelegramBot

    hermes = build_hermes()
    scheduler = build_scheduler(hermes)
    _start_healthcheck_server()
    _start_keepalive()

    def _run_scheduler():
        logger.info("Scheduler starting in background thread — %d jobs",
                    len(scheduler.get_jobs()))
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass

    threading.Thread(target=_run_scheduler, daemon=True, name="scheduler").start()

    def _catchup_allocator():
        """Run the weekly rebalance at startup if one is owed.

        allocator_rebalance is cron'd Monday 09:20 IST only. A service that
        is down, deploying, or newly created at that moment holds NO
        positions until the FOLLOWING Monday, with nothing in the logs
        saying so — that is exactly what happened here: the job had never
        executed once since the Render deploy. is_due() is week-based and
        DB-backed, so this cannot double-trade: once a rebalance is
        recorded for the current ISO week, restarts are no-ops.
        """
        try:
            alloc = getattr(hermes, "allocator", None)
            if alloc is None or not alloc.is_due():
                logger.info("startup: no rebalance owed this week")
                return
            logger.warning("startup: rebalance owed — running catch-up now")
            res = hermes.run_allocator_rebalance()
            logger.warning("startup catch-up result: %s",
                           {k: v for k, v in res.items() if k != "trades"})
        except Exception:  # noqa: BLE001 — must never block the bot
            logger.exception("startup allocator catch-up failed")

    threading.Thread(target=_catchup_allocator, daemon=True,
                     name="alloc-catchup").start()

    bot = QuNtraTelegramBot(hermes, alerter=hermes.telegram)
    logger.info("Starting Telegram command center (%d commands) in the "
               "main thread…", len(bot.COMMANDS))
    bot.run_polling()  # blocks until killed — owns the main thread
    return 0


if __name__ == "__main__":
    sys.exit(main())
