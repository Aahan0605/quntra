"""
QuNtra database schema — SQLAlchemy 2.0 models for the 7 core tables.

Production backend: PostgreSQL 15 (POSTGRES_URL in config/secrets.env).
Tests/CI: SQLite (same models — JSON/UUID types degrade gracefully).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class Trade(Base):
    """Every trade, paper and live."""
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    signal_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(5), nullable=False)  # LONG/SHORT
    entry_price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    exit_price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    quantity: Mapped[int | None] = mapped_column(Integer)
    entry_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pnl: Mapped[float | None] = mapped_column(Numeric(12, 2))
    pnl_pct: Mapped[float | None] = mapped_column(Numeric(8, 4))
    signal_score: Mapped[int | None] = mapped_column(Integer)
    regime: Mapped[str | None] = mapped_column(String(30))
    exit_reason: Mapped[str | None] = mapped_column(String(50))
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Signal(Base):
    """Every signal generated — executed and rejected."""
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    signal_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    score: Mapped[int | None] = mapped_column(Integer)
    direction: Mapped[str | None] = mapped_column(String(5))
    agent_votes: Mapped[dict | None] = mapped_column(JSON)
    regime: Mapped[str | None] = mapped_column(String(30))
    reasoning: Mapped[str | None] = mapped_column(Text)
    executed: Mapped[bool] = mapped_column(Boolean, default=False)
    rejection_reason: Mapped[str | None] = mapped_column(String(100))
    signal_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AgentCredibility(Base):
    """Self-learning agent weights."""
    __tablename__ = "agent_credibility"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    credibility_weight: Mapped[float] = mapped_column(Numeric(6, 4), default=1.0)
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    correct_calls: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class BacktestResult(Base):
    """All backtest runs."""
    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    strategy: Mapped[str | None] = mapped_column(String(50))
    start_date: Mapped[datetime | None] = mapped_column(Date)
    end_date: Mapped[datetime | None] = mapped_column(Date)
    universe: Mapped[list | None] = mapped_column(JSON)
    params: Mapped[dict | None] = mapped_column(JSON)
    sharpe: Mapped[float | None] = mapped_column(Numeric(8, 4))
    calmar: Mapped[float | None] = mapped_column(Numeric(8, 4))
    max_dd: Mapped[float | None] = mapped_column(Numeric(8, 4))
    annual_return: Mapped[float | None] = mapped_column(Numeric(8, 4))
    total_trades: Mapped[int | None] = mapped_column(Integer)
    win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PriceData(Base):
    """Market data cache — point-in-time safe."""
    __tablename__ = "price_data"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(Date, nullable=False)
    open: Mapped[float | None] = mapped_column(Numeric(12, 2))
    high: Mapped[float | None] = mapped_column(Numeric(12, 2))
    low: Mapped[float | None] = mapped_column(Numeric(12, 2))
    close: Mapped[float | None] = mapped_column(Numeric(12, 2))
    volume: Mapped[int | None] = mapped_column(BigInteger)
    adjusted_close: Mapped[float | None] = mapped_column(Numeric(12, 2))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ResearchNote(Base):
    """Agent memory: research, lessons learned, mistake reports."""
    __tablename__ = "research_notes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    note_type: Mapped[str | None] = mapped_column(String(30), index=True)
    ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    content: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    sentiment: Mapped[float | None] = mapped_column(Numeric(4, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SystemState(Base):
    """Hermes coordinator state — key/value with JSON payloads."""
    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    value: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
