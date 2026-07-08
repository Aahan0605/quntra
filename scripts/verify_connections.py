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

    # MLflow (local experiment tracking — no account, just a store URI)
    ml_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not ml_uri:
        results["MLflow"] = ("DEFAULT ✓ (local SQLite — set "
                             "MLFLOW_TRACKING_URI to override)")
    else:
        try:
            import mlflow
            mlflow.set_tracking_uri(ml_uri)
            mlflow.search_experiments()  # touches the store
            results["MLflow"] = f"OK ✓ ({ml_uri.split('://')[0]} store)"
        except Exception as e:  # noqa: BLE001
            results["MLflow"] = f"FAILED: {str(e)[:80]}"
            hard_fail = True

    # IBM Quantum (OPTIONAL — QAOA runs on the local simulator without it,
    # so a missing/bad key never hard-fails)
    try:
        from src.quantum.ibm_provider import verify as ibm_verify
        st = ibm_verify()
        if not st.configured:
            results["IBM Quantum"] = "SKIPPED (optional — local simulator)"
        elif st.connected:
            results["IBM Quantum"] = f"CONNECTED ✓ ({st.detail})"
        else:
            results["IBM Quantum"] = (f"CONFIGURED but {st.detail} "
                                      f"(falls back to simulator)")
    except Exception as e:  # noqa: BLE001
        results["IBM Quantum"] = f"SKIPPED (check unavailable: {e})"

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
