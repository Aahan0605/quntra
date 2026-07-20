"""
Kite session status — is today's access token still valid?

Kite access tokens expire every morning (~07:30 IST). This checks the
stored token with a lightweight authenticated call (profile()) that works
with basic order permissions — so it reports token validity even on an
account without the paid market-data subscription.

Returns one of:
    "not_configured"  — no API key / token in secrets
    "valid"           — token authenticates
    "expired"         — token rejected (TokenException) → needs re-login
    "error: ..."      — some other failure (network, etc.)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("quntra.kite")

_ROOT = Path(__file__).resolve().parents[2]


def token_status() -> str:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / "config" / "secrets.env")
    api_key = os.getenv("KITE_API_KEY")
    token = os.getenv("KITE_ACCESS_TOKEN")
    if not (api_key and token):
        return "not_configured"
    try:
        from kiteconnect import KiteConnect
        from kiteconnect.exceptions import TokenException
    except ImportError:
        return "error: kiteconnect not installed"
    try:
        k = KiteConnect(api_key=api_key)
        k.set_access_token(token)
        k.profile()  # cheap authenticated call; works with basic permissions
        return "valid"
    except TokenException:
        return "expired"
    except Exception as e:  # noqa: BLE001 — permission/network issues
        # A PermissionException here still means the TOKEN is valid (the
        # account merely lacks that endpoint), so treat non-token errors
        # as "valid" for the purpose of the daily reminder.
        name = type(e).__name__
        if name == "PermissionException":
            return "valid"
        return f"error: {name}"
