"""
IBM Quantum provider — OPTIONAL real-hardware backend for the QAOA optimizer.

QuNtra's quantum optimizer runs on the local Aer simulator by default, with
no account required. This module lets you point it at real IBM Quantum
hardware IF (and only if) you provide credentials:

    IBM_QUANTUM_API_KEY   your IBM Quantum / QSA API token
    IBM_QUANTUM_QSA_URL   base URL of the QSA (the /version endpoint's host)

verify() authenticates the token against the QSA `/versions` endpoint (which
returns 401 on a bad token, 200 with the supported versions otherwise) using
a plain HTTPS GET — no heavyweight SDK is pulled in just to check a key.
Everything degrades to the simulator, so a missing or bad key never blocks
the system.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("quntra.quantum.ibm")

DEFAULT_TIMEOUT = 10


@dataclass
class IBMStatus:
    configured: bool          # both key and URL present
    connected: bool           # /versions authenticated OK
    detail: str
    versions: list[str] | None = None


def _load_credentials() -> tuple[str | None, str | None]:
    key = os.getenv("IBM_QUANTUM_API_KEY") or None
    url = os.getenv("IBM_QUANTUM_QSA_URL") or None
    if not (key and url):
        # Fall back to config/secrets.env without importing dotenv globally
        try:
            from pathlib import Path
            secrets = Path(__file__).resolve().parents[2] / "config" / "secrets.env"
            if secrets.exists():
                for line in secrets.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("IBM_QUANTUM_API_KEY=") and not key:
                        key = line.split("=", 1)[1].strip() or None
                    elif line.startswith("IBM_QUANTUM_QSA_URL=") and not url:
                        url = line.split("=", 1)[1].strip() or None
        except Exception:  # noqa: BLE001
            pass
    return key, url


def verify(timeout: int = DEFAULT_TIMEOUT) -> IBMStatus:
    """Check IBM Quantum credentials. Never raises."""
    key, url = _load_credentials()
    if not (key and url):
        return IBMStatus(
            configured=False, connected=False,
            detail="not configured — using local Aer simulator (no key needed)")

    base = url.rstrip("/")
    try:
        import requests
        # /versions requires auth (401 on bad token); /version does not.
        r = requests.get(
            f"{base}/versions",
            headers={"Accept": "application/json",
                     "Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        if r.status_code == 200:
            versions = None
            try:
                body = r.json()
                versions = body.get("versions") if isinstance(body, dict) else None
            except Exception:  # noqa: BLE001
                pass
            return IBMStatus(configured=True, connected=True,
                             detail="authenticated ✓", versions=versions)
        if r.status_code == 401:
            return IBMStatus(configured=True, connected=False,
                             detail="401 — API key rejected (check the token)")
        return IBMStatus(configured=True, connected=False,
                         detail=f"HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:  # noqa: BLE001
        return IBMStatus(configured=True, connected=False,
                         detail=f"unreachable: {e}")


def latest_version(timeout: int = DEFAULT_TIMEOUT) -> str | None:
    """The QSA's latest supported version via the public /version endpoint.
    Returns None when not configured or unreachable."""
    _, url = _load_credentials()
    if not url:
        return None
    try:
        import json

        import requests
        r = requests.get(f"{url.rstrip('/')}/version",
                         headers={"Accept": "application/json"}, timeout=timeout)
        if r.status_code == 200:
            body = r.json()
            raw = body.get("version") if isinstance(body, dict) else None
            # The endpoint double-encodes: {"version": "{\"version\": \"...\"}"}
            if isinstance(raw, str) and raw.strip().startswith("{"):
                try:
                    return json.loads(raw).get("version")
                except Exception:  # noqa: BLE001
                    return raw
            return raw
    except Exception as e:  # noqa: BLE001
        logger.warning("IBM /version fetch failed: %s", e)
    return None


def is_enabled() -> bool:
    """True only when credentials are present AND authenticate. Callers use
    this to decide hardware vs simulator; default is always simulator."""
    return verify().connected
