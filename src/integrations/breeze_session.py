"""ICICI Breeze session status — is today's session_token still valid?

Breeze session_tokens expire daily, same as Kite's access_token — extract
a fresh one from the `apisession=` query param on the login redirect and
re-set it every morning. Unlike Kite there's no separate "access token";
generate_session(api_secret, session_token) IS the authentication, done
fresh from the stored session_token on every process start
(src/utils/data_fetcher.py:_get_breeze).

Also unlike Kite, breeze_connect does not raise on a bad session — it
returns {"Status": 5, "Error": "Authentication Fail :: Invalid Checksum."}
as an ordinary response, so validity has to be read out of that dict.

Returns one of:
    "not_configured"  — no API key / secret / session token in secrets
    "valid"            — session authenticates
    "expired"          — session rejected -> needs a fresh apisession= value
    "error: ..."       — some other failure (network, etc.)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.utils.secrets_file import write_secret

logger = logging.getLogger("quntra.breeze")

_ROOT = Path(__file__).resolve().parents[2]

# breeze_connect downloads its security-master zip at import time via bare
# urllib — the Python.org macOS build's bundled cert.pem lacks the issuing
# CA, so first import 500s on CERTIFICATE_VERIFY_FAILED without this.
def _fix_ssl_cert_env() -> None:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def login_url(api_key: str) -> str:
    """The ICICI Breeze login page — approve here, then read apisession=
    off the http://127.0.0.1:.../?apisession=... redirect.

    quote_plus (not quote) per breeze_connect's own documented example —
    api keys can contain '+', which plain quote() leaves unescaped.
    """
    from urllib.parse import quote_plus
    return ("https://api.icicidirect.com/apiuser/login?api_key="
           + quote_plus(api_key))


def set_token(session_token: str) -> str:
    """Persist a fresh session_token and verify it actually authenticates.
    Raises with a clear message on failure. Returns the session_token."""
    from dotenv import load_dotenv
    load_dotenv(_ROOT / "config" / "secrets.env")
    api_key = os.getenv("ICICI_BREEZE_API_KEY")
    api_secret = os.getenv("ICICI_BREEZE_API_SECRET")
    if not (api_key and api_secret):
        raise RuntimeError("ICICI_BREEZE_API_KEY / _API_SECRET missing in secrets")
    session_token = session_token.strip()

    _fix_ssl_cert_env()
    from breeze_connect import BreezeConnect
    b = BreezeConnect(api_key=api_key)
    b.generate_session(api_secret=api_secret, session_token=session_token)
    resp = b.get_quotes(stock_code="RELIND", exchange_code="NSE",
                        product_type="cash")
    if resp.get("Error"):
        raise RuntimeError(resp["Error"])

    write_secret("ICICI_BREEZE_SESSION_TOKEN", session_token)
    _refresh_caches(session_token)
    return session_token


def _refresh_caches(session_token: str) -> None:
    os.environ["ICICI_BREEZE_SESSION_TOKEN"] = session_token
    try:
        from src.utils.data_fetcher import UnifiedDataFetcher
        UnifiedDataFetcher._breeze = None
        UnifiedDataFetcher._breeze_tried = False
    except Exception:  # noqa: BLE001
        pass


def token_status() -> str:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / "config" / "secrets.env")
    api_key = os.getenv("ICICI_BREEZE_API_KEY")
    api_secret = os.getenv("ICICI_BREEZE_API_SECRET")
    token = os.getenv("ICICI_BREEZE_SESSION_TOKEN")
    if not (api_key and api_secret and token):
        return "not_configured"
    try:
        _fix_ssl_cert_env()
        from breeze_connect import BreezeConnect
        b = BreezeConnect(api_key=api_key)
        b.generate_session(api_secret=api_secret, session_token=token)
        resp = b.get_quotes(stock_code="RELIND", exchange_code="NSE",
                            product_type="cash")
        if resp.get("Error"):
            return "expired"
        return "valid"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}"
