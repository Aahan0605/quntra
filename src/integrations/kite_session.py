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


def _write_secret(key: str, value: str) -> None:
    """Upsert one KEY=value line in config/secrets.env (gitignored)."""
    env = _ROOT / "config" / "secrets.env"
    lines = env.read_text().splitlines() if env.exists() else []
    lines = [l for l in lines if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    env.write_text("\n".join(lines) + "\n")


def exchange_request_token(request_token: str) -> str:
    """Exchange a login request_token for today's access token and persist
    it to secrets.env. Returns the access token. Raises on failure.

    The operator gets request_token from the Kite login redirect URL (works
    in a phone browser), so this is the piece that lets the whole daily
    re-login happen from Telegram without a laptop.
    """
    from dotenv import load_dotenv
    load_dotenv(_ROOT / "config" / "secrets.env")
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    if not (api_key and api_secret):
        raise RuntimeError("KITE_API_KEY / KITE_API_SECRET missing in secrets")
    from kiteconnect import KiteConnect
    k = KiteConnect(api_key=api_key)
    data = k.generate_session(request_token.strip(), api_secret=api_secret)
    access_token = data["access_token"]
    _write_secret("KITE_ACCESS_TOKEN", access_token)
    _refresh_caches(access_token)
    return access_token


def set_token(token: str) -> tuple[str, str]:
    """Accept EITHER a ready access_token OR a login request_token and make
    it the live token. Returns (access_token, how) where how is 'direct'
    (it was already a valid access token) or 'exchanged' (it was a
    request_token we swapped for an access token). Raises with a clear
    message if it is neither.

    This spares the operator from having to know which string Kite handed
    them — they just paste whatever they have.
    """
    from dotenv import load_dotenv
    load_dotenv(_ROOT / "config" / "secrets.env")
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    if not api_key:
        raise RuntimeError("KITE_API_KEY missing in secrets")
    token = token.strip()

    from kiteconnect import KiteConnect

    # 1) Try it as an access token: a cheap authenticated call. A
    #    request_token used here just fails auth (it is NOT consumed), so
    #    falling through to the exchange below is safe.
    try:
        k = KiteConnect(api_key=api_key)
        k.set_access_token(token)
        k.profile()
        _write_secret("KITE_ACCESS_TOKEN", token)
        _refresh_caches(token)
        return token, "direct"
    except Exception:  # noqa: BLE001 — not a valid access token; try exchange
        pass

    # 2) Try it as a request_token (needs the api_secret).
    if not api_secret:
        raise RuntimeError("Not a valid access token, and KITE_API_SECRET is "
                           "missing so it can't be exchanged as a "
                           "request_token.")
    k = KiteConnect(api_key=api_key)
    data = k.generate_session(token, api_secret=api_secret)
    access_token = data["access_token"]
    _write_secret("KITE_ACCESS_TOKEN", access_token)
    _refresh_caches(access_token)
    return access_token, "exchanged"


def _refresh_caches(access_token: str) -> None:
    os.environ["KITE_ACCESS_TOKEN"] = access_token
    try:
        from src.utils.data_fetcher import UnifiedDataFetcher
        UnifiedDataFetcher._kite = None
        UnifiedDataFetcher._kite_tried = False
    except Exception:  # noqa: BLE001
        pass


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
