"""
Shared metric queries for all QuNtra reports.

Every number a report prints comes from PostgreSQL through here —
no hardcoded values, and every function tolerates an empty database
(day 1 of paper trading is mostly zeros).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select

from src.db import (
    AgentCredibility,
    KnowledgeItem,
    ResearchNote,
    SystemState,
    Trade,
    get_session,
)


def trades_between(db_url, start: datetime, end: datetime) -> list[Trade]:
    with get_session(db_url) as s:
        rows = s.execute(
            select(Trade)
            .where(Trade.created_at >= start, Trade.created_at < end)
            .order_by(Trade.created_at)
        ).scalars().all()
        s.expunge_all()
        return rows


def pnl_stats(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.pnl is not None]
    pnls = [float(t.pnl) for t in closed]
    wins = [p for p in pnls if p > 0]
    return {
        "n_trades": len(trades),
        "n_closed": len(closed),
        "net_pnl": sum(pnls),
        "n_wins": len(wins),
        "n_losses": len([p for p in pnls if p <= 0]),
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
    }


def daily_pnl_series(db_url, days: int = 30) -> dict[str, float]:
    """date-iso -> net P&L of trades closed that day."""
    start = datetime.now(timezone.utc) - timedelta(days=days)
    series: dict[str, float] = {}
    with get_session(db_url) as s:
        rows = s.execute(
            select(Trade).where(Trade.created_at >= start,
                                Trade.pnl.is_not(None))
        ).scalars().all()
        for t in rows:
            when = (t.exit_time or t.created_at)
            key = when.date().isoformat()
            series[key] = series.get(key, 0.0) + float(t.pnl)
    return dict(sorted(series.items()))


def rolling_sharpe(db_url, days: int = 30,
                   capital: float = 25_000.0) -> float | None:
    """Annualized Sharpe (rf=0 — same convention as the Phase-0 gate)
    of daily P&L returns on nominal capital. None until 5 trading days."""
    series = daily_pnl_series(db_url, days)
    if len(series) < 5:
        return None
    rets = np.array(list(series.values())) / capital
    if rets.std() == 0:
        return None
    return float(rets.mean() / rets.std() * np.sqrt(252))


def max_drawdown_from_pnl(db_url, days: int = 90,
                          capital: float = 25_000.0) -> float:
    series = daily_pnl_series(db_url, days)
    if not series:
        return 0.0
    equity = capital + np.cumsum(list(series.values()))
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def agent_credibility_table(db_url) -> list[dict]:
    with get_session(db_url) as s:
        rows = s.execute(
            select(AgentCredibility).order_by(AgentCredibility.agent_name)
        ).scalars().all()
        return [
            {
                "agent": r.agent_name,
                "weight": float(r.credibility_weight),
                "total": r.total_calls,
                "correct": r.correct_calls,
                "accuracy": (r.correct_calls / r.total_calls
                             if r.total_calls else 0.0),
            }
            for r in rows
        ]


def system_state(db_url, key: str) -> dict:
    with get_session(db_url) as s:
        row = s.get(SystemState, key)
        return row.value or {} if row else {}


def research_notes_since(db_url, hours: int = 24) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with get_session(db_url) as s:
        rows = s.execute(
            select(ResearchNote)
            .where(ResearchNote.created_at >= cutoff)
            .order_by(ResearchNote.created_at.desc())
        ).scalars().all()
        return [{"source": r.source, "summary": (r.summary or "")[:120],
                 "type": r.note_type} for r in rows]


def knowledge_count(db_url) -> int:
    from sqlalchemy import func
    with get_session(db_url) as s:
        return int(s.execute(
            select(func.count()).select_from(KnowledgeItem)).scalar_one())


def nifty_move_today() -> float | None:
    """Benchmark %move — network-dependent, degrades to None."""
    from src.agents.research.base import yf_pct_change
    return yf_pct_change("^NSEI", period="5d")


def credibility_bar(weight: float, lo: float = 0.1, hi: float = 3.0) -> str:
    filled = int(round((weight - lo) / (hi - lo) * 10))
    filled = max(0, min(10, filled))
    return "▓" * filled + "░" * (10 - filled)
