#!/usr/bin/env python3
"""
Verify every external connection that has credentials (Task R6).

Reports per service: CONNECTED / FAILED / SKIPPED (not configured).
Safe to run anywhere — never raises, exits 1 only if a CONFIGURED
service fails.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "config" / "secrets.env")
load_dotenv(ROOT / ".env")

PLACEHOLDERS = {"", "your_kite_api_key", "your_bot_token",
                "your_token_here", "your_chat_id_here", "optional"}


def main() -> int:
    results: dict[str, str] = {}
    hard_fail = False

    # PostgreSQL (or SQLite fallback)
    try:
        import sqlalchemy as sa
        url = os.getenv("POSTGRES_URL")
        if url and not url.startswith("sqlite"):
            engine = sa.create_engine(url)
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            results["PostgreSQL"] = "CONNECTED ✓"
        else:
            from src.db import init_db
            init_db()
            results["PostgreSQL"] = ("SQLITE FALLBACK ✓ (set POSTGRES_URL "
                                     "for production)")
    except Exception as e:  # noqa: BLE001
        results["PostgreSQL"] = f"FAILED: {e}"
        hard_fail = True

    # Telegram
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if token in PLACEHOLDERS:
        results["Telegram"] = "SKIPPED (token not set — see RUNBOOK step 6)"
    else:
        try:
            import requests
            r = requests.get(f"https://api.telegram.org/bot{token}/getMe",
                             timeout=8)
            if r.status_code == 200:
                results["Telegram"] = \
                    f"CONNECTED ✓ (@{r.json()['result']['username']})"
            else:
                results["Telegram"] = f"FAILED: HTTP {r.status_code} {r.text[:80]}"
                hard_fail = True
        except Exception as e:  # noqa: BLE001
            results["Telegram"] = f"FAILED: {e}"
            hard_fail = True

    # Kite (key presence only — access token is generated daily at login)
    api_key = os.getenv("KITE_API_KEY", "")
    if api_key in PLACEHOLDERS:
        results["Kite"] = ("SKIPPED (not needed until the 40-day paper "
                           "gate passes)")
    else:
        try:
            from kiteconnect import KiteConnect
            KiteConnect(api_key=api_key)
            results["Kite"] = ("API KEY SET ✓ (access token needed daily "
                               "for live session)")
        except Exception as e:  # noqa: BLE001
            results["Kite"] = f"FAILED: {e}"
            hard_fail = True

    # NSE market data (needed by the paper trader during market hours)
    try:
        import socket
        socket.create_connection(("www.nseindia.com", 443), timeout=5).close()
        results["NSE data"] = "REACHABLE ✓"
    except OSError:
        results["NSE data"] = ("UNREACHABLE — fine outside market hours / "
                               "sandbox; required on the trading machine")

    print("\nConnection verification:")
    for k, v in results.items():
        print(f"  {k:12s}: {v}")
    if hard_fail:
        print("\nOne or more CONFIGURED services failed — fix before starting.")
        return 1
    print("\nAll configured services verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
