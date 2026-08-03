"""Keep secrets out of log files.

httpx logs every request URL at INFO, and the Telegram API puts the bot
token *in the URL path*. That wrote the live token into
logs/telegram_bot.log 13,143 times (11 MB, mode 644) — anything that could
read logs owned the bot. This module scrubs known secret shapes from every
log record and turns the noisiest offender down.

Usage (once, right after logging.basicConfig):

    from src.utils.log_redaction import install_redaction
    install_redaction()
"""

from __future__ import annotations

import logging
import os
import re

# Telegram bot tokens: <numeric id>:<35-char base64ish secret>. No leading
# \b: the real-world shape is "/bot8874867230:AAG8.../getUpdates" — digits
# are glued directly onto "bot" with no boundary, so a leading \b silently
# never matched real tokens (verified only against literal-value matching,
# which happened to mask this everywhere except watchdog.py).
_TELEGRAM_TOKEN = re.compile(r"\d{6,12}:[A-Za-z0-9_-]{30,}\b")

# Env vars whose literal values must never reach a log line.
_SECRET_ENV_VARS = (
    "TELEGRAM_BOT_TOKEN", "KITE_API_KEY", "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN", "ANTHROPIC_API_KEY", "IBM_QUANTUM_API_KEY",
    "POSTGRES_URL",
)

_MASK = "<redacted>"


def _literal_patterns() -> list[re.Pattern]:
    """Exact-value patterns for whatever is actually in the environment.

    Read at install time; short values are skipped so a stray "true" or a
    2-char setting can't blank out half of every log line.
    """
    pats = []
    for var in _SECRET_ENV_VARS:
        val = os.getenv(var)
        if val and len(val) >= 12:
            pats.append(re.compile(re.escape(val)))
    return pats


class RedactingFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._patterns = [_TELEGRAM_TOKEN] + _literal_patterns()

    def _scrub(self, text: str) -> str:
        for pat in self._patterns:
            text = pat.sub(_MASK, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        # Render args in now, so the scrub sees the final text rather than
        # a format string with the secret hiding in record.args.
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — never break logging
            return True
        scrubbed = self._scrub(msg)
        if scrubbed != msg:
            record.msg = scrubbed
            record.args = ()
        return True


def install_redaction(quiet_http: bool = True) -> None:
    """Attach the filter to every existing handler on the root logger."""
    f = RedactingFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(f)
    if quiet_http:
        # These log full request URLs — the token's hiding place.
        for name in ("httpx", "httpcore", "telegram.request", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
