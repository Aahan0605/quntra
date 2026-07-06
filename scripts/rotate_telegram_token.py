#!/usr/bin/env python3
"""
QuNtra Telegram token rotation helper.

The original token was exposed in a chat session — rotate it:
  1. Telegram → @BotFather → /mybots → your bot → API Token → Revoke
  2. python3 scripts/rotate_telegram_token.py --new-token <NEW_TOKEN>
  3. Restart both processes:
       kill $(cat quntra.pid) $(cat telegram_bot.pid)
       python3 scripts/complete_local_setup.py --from 8
       nohup ./venv/bin/python scripts/run_telegram_bot.py \
           >> logs/telegram_bot.log 2>&1 & echo $! > telegram_bot.pid
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "config" / "secrets.env"

TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")


def rotate(new_token: str) -> int:
    if not TOKEN_RE.match(new_token):
        print("ERROR: Token format invalid. Expected: 123456789:ABCdef…")
        return 1
    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found")
        return 1

    lines = ENV_PATH.read_text().splitlines()
    old_token = None
    new_lines = []
    replaced = False
    for line in lines:
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            old_token = line.split("=", 1)[1]
            new_lines.append(f"TELEGRAM_BOT_TOKEN={new_token}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"TELEGRAM_BOT_TOKEN={new_token}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n")
    print("✓ Token rotated in config/secrets.env")
    if old_token:
        print(f"  Old: {old_token[:10]}… (revoke in @BotFather if not done)")
    print(f"  New: {new_token[:10]}…")
    print("\nNext: restart the scheduler and the bot runner:")
    print("  kill $(cat quntra.pid) $(cat telegram_bot.pid) 2>/dev/null")
    print("  ./venv/bin/python scripts/complete_local_setup.py --from 8")
    print("  nohup ./venv/bin/python scripts/run_telegram_bot.py "
          ">> logs/telegram_bot.log 2>&1 & echo $! > telegram_bot.pid")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-token", required=True)
    sys.exit(rotate(ap.parse_args().new_token))
