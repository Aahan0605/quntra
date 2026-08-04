"""The daily Breeze session token must survive a process restart.

This pins a bug that would have broken the first cloud paper-trading day:
set_token() persisted only to config/secrets.env, but on Render that file
isn't in the image (correctly gitignored) AND the filesystem is ephemeral.
So /breeze_token would have either crashed outright or "succeeded" and then
silently lost the token on the next restart — leaving live quotes on
delayed yfinance with no obvious signal that anything was wrong.

system_state is the only durable, shared store a cloud deploy has, so the
DB is the source of truth; the env var is a deploy-time fallback and the
local file is a best-effort convenience.
"""

import tempfile

import pytest

import src.db.session as db_session
import src.integrations.breeze_session as bs
from src.db import init_db


@pytest.fixture
def db(monkeypatch):
    url = f"sqlite:///{tempfile.mktemp(suffix='.db')}"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    # Never touch the developer's real config/secrets.env from a test.
    monkeypatch.setattr(bs, "write_secret", lambda k, v: None)
    yield url
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_token_persists_to_db_and_survives_restart(db, monkeypatch):
    monkeypatch.delenv("ICICI_BREEZE_SESSION_TOKEN", raising=False)
    assert bs.stored_token() is None

    bs._persist_token("SESSION_ABC")
    assert bs.stored_token() == "SESSION_ABC"

    # A restart clears process env but not the DB.
    monkeypatch.delenv("ICICI_BREEZE_SESSION_TOKEN", raising=False)
    assert bs.stored_token() == "SESSION_ABC"


def test_db_token_wins_over_stale_env(db, monkeypatch):
    """The env var holds the deploy-time token, which goes stale daily —
    a token refreshed via /breeze_token must take precedence."""
    monkeypatch.setenv("ICICI_BREEZE_SESSION_TOKEN", "STALE_DEPLOY_TIME")
    bs._persist_token("FRESH_FROM_TELEGRAM")
    assert bs.stored_token() == "FRESH_FROM_TELEGRAM"


def test_env_is_the_fallback_before_any_token_is_set(db, monkeypatch):
    monkeypatch.setenv("ICICI_BREEZE_SESSION_TOKEN", "FROM_ENV")
    assert bs.stored_token() == "FROM_ENV"


def test_persist_survives_unwritable_secrets_file(db, monkeypatch):
    """On Render config/secrets.env does not exist. write_secret() blowing
    up must not lose the token — the DB write is what matters."""
    def boom(key, value):
        raise FileNotFoundError("config/secrets.env")

    monkeypatch.setattr(bs, "write_secret", boom)
    monkeypatch.delenv("ICICI_BREEZE_SESSION_TOKEN", raising=False)
    bs._persist_token("SURVIVES_NO_FILE")
    assert bs.stored_token() == "SURVIVES_NO_FILE"


def test_stored_token_falls_back_to_env_when_db_is_down(monkeypatch):
    """A DB outage must degrade to the env var, not raise."""
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    monkeypatch.setenv("POSTGRES_URL", "postgresql://nope:nope@127.0.0.1:1/x")
    monkeypatch.setenv("ICICI_BREEZE_SESSION_TOKEN", "ENV_FALLBACK")
    assert bs.stored_token() == "ENV_FALLBACK"
