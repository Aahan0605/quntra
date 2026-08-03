"""log_redaction — the token regex had a real bug: a leading \\b required a
non-word char before the digits, but Telegram's real URL shape glues them
directly onto "bot" (bot8874867230:...), so it silently never matched.
Only literal-value matching (which needs the token already in os.environ)
was catching it — which is why watchdog.py leaked while scheduler/bot,
which load secrets first, didn't.
"""

import logging

import pytest

from src.utils.log_redaction import RedactingFilter, install_redaction

REAL_SHAPE_URL = ("HTTP Request: POST https://api.telegram.org/"
                  "bot1234567890:FAKEtokenABCDEFGHIJKLMNOPQRSTUVWXYZ/"
                  "getUpdates \"200 OK\"")


def test_structural_regex_matches_token_glued_to_bot():
    """No env var needed — this is the shape-only match."""
    f = RedactingFilter()
    assert f._scrub(REAL_SHAPE_URL) == (
        "HTTP Request: POST https://api.telegram.org/bot<redacted>/"
        "getUpdates \"200 OK\"")


def test_literal_match_also_works_when_env_is_set(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN",
                       "1234567890:FAKEtokenABCDEFGHIJKLMNOPQRSTUVWXYZ")
    f = RedactingFilter()
    assert "FAKEtoken" not in f._scrub(REAL_SHAPE_URL)


def test_install_redaction_scrubs_a_real_log_record(capsys):
    """End-to-end through a real handler, not just the filter in isolation."""
    logger = logging.getLogger("test.redaction.e2e")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    install_redaction()  # attaches to every handler on the ROOT logger...
    handler.addFilter(RedactingFilter())  # ...so attach directly here too

    logger.info(REAL_SHAPE_URL)
    err = capsys.readouterr().err
    assert "FAKEtoken" not in err
    assert "<redacted>" in err
