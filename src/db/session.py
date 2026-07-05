"""
Engine/session factory.

Backend selection:
  1. POSTGRES_URL env var (or config/secrets.env) — production.
  2. Fallback: SQLite at data/quntra.db — dev/tests, zero setup.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SQLITE = f"sqlite:///{ROOT / 'data' / 'quntra.db'}"

_engine = None
_SessionLocal = None


def _database_url() -> str:
    url = os.environ.get("POSTGRES_URL")
    if url:
        return url
    secrets = ROOT / "config" / "secrets.env"
    if secrets.exists():
        for line in secrets.read_text().splitlines():
            if line.strip().startswith("POSTGRES_URL="):
                candidate = line.split("=", 1)[1].strip()
                if candidate:
                    return candidate
    return _DEFAULT_SQLITE


def get_engine(url: str | None = None):
    global _engine, _SessionLocal
    if _engine is None or url is not None:
        target = url or _database_url()
        if target.startswith("sqlite"):
            Path(target.replace("sqlite:///", "")).parent.mkdir(
                parents=True, exist_ok=True
            )
        _engine = create_engine(target, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db(url: str | None = None):
    """Create all tables (idempotent). Production should use Alembic."""
    from src.db.models import Base
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(url: str | None = None) -> Session:
    get_engine(url)
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
