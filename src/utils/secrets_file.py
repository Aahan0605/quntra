"""Upsert a single KEY=value line in config/secrets.env (gitignored).

Shared by every daily-refreshed-credential flow (Kite, ICICI Breeze) so
there is exactly one implementation of "safely rewrite the secrets file"
rather than one per integration quietly drifting apart.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def write_secret(key: str, value: str) -> None:
    env = _ROOT / "config" / "secrets.env"
    lines = env.read_text().splitlines() if env.exists() else []
    lines = [l for l in lines if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    env.write_text("\n".join(lines) + "\n")
