#!/usr/bin/env python3
"""
QuNtra Telegram bot runner — long-polls for operator commands.

Runs as its own process (separate from the scheduler) so a bot crash
never touches trading, and vice versa. Needs only TELEGRAM_BOT_TOKEN;
the chat_id is captured from your first message and persisted.

    nohup ./venv/bin/python scripts/run_telegram_bot.py \
        >> logs/telegram_bot.log 2>&1 &
    echo $! > telegram_bot.pid
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("quntra.bot_runner")


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / "config" / "secrets.env")
    load_dotenv(ROOT / ".env")

    from scripts.scheduler import build_hermes
    from src.alerts.telegram_bot import QuNtraTelegramBot

    hermes = build_hermes()
    bot = QuNtraTelegramBot(hermes, alerter=hermes.telegram)
    logger.info("Starting Telegram command center (%d commands)…",
                len(bot.COMMANDS))
    bot.run_polling()  # blocks until killed
    return 0


if __name__ == "__main__":
    sys.exit(main())
