"""
QuNtra Brain — persistent memory for all agents.

Inspired by gBrain architecture (gBrain itself is TypeScript; this is a
native Python implementation). Everything persists to PostgreSQL/SQLite
via the src.db layer, so memory survives process restarts.

Credibility update rule:
    correct   -> weight *= 1.05
    incorrect -> weight *= 0.95
    floor 0.1, ceiling 3.0
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from src.db import (
    AgentCredibility,
    ResearchNote,
    Signal,
    Trade,
    get_session,
)

CRED_UP = 1.05
CRED_DOWN = 0.95
CRED_FLOOR = 0.1
CRED_CEILING = 3.0


class QuNtraBrain:
    def __init__(self, db_url: str | None = None):
        self.db_url = db_url  # None -> default from session factory

    def _session(self):
        return get_session(self.db_url)

    # ------------------------------------------------------------------ #
    # Trades / signals / research

    def remember_trade(self, trade_data: dict) -> str:
        allowed = {c.name for c in Trade.__table__.columns}
        row = Trade(**{k: v for k, v in trade_data.items() if k in allowed})
        with self._session() as s:
            s.add(row)
            s.flush()
            return row.id

    def remember_signal(self, signal_data: dict) -> str:
        allowed = {c.name for c in Signal.__table__.columns}
        row = Signal(**{k: v for k, v in signal_data.items() if k in allowed})
        with self._session() as s:
            s.add(row)
            s.flush()
            return row.id

    def remember_research(self, note: dict) -> str:
        allowed = {c.name for c in ResearchNote.__table__.columns}
        row = ResearchNote(**{k: v for k, v in note.items() if k in allowed})
        with self._session() as s:
            s.add(row)
            s.flush()
            return row.id

    def close_trade(self, signal_hash: str, exit_data: dict) -> bool:
        """Close an open trade in place.

        Exits used to be written as a *second* row keyed `<hash>:exit`,
        which left the original row reading as open forever — every P&L
        and open-position query double-counted. This updates the row that
        is actually open. Returns False if there was nothing to close.
        """
        allowed = {c.name for c in Trade.__table__.columns}
        with self._session() as s:
            row = s.execute(
                select(Trade)
                .where(Trade.signal_hash == signal_hash)
                .where(Trade.exit_time.is_(None))
            ).scalar_one_or_none()
            if row is None:
                return False
            for k, v in exit_data.items():
                if k in allowed and k not in ("id", "signal_hash"):
                    setattr(row, k, v)
            return True

    def update_position_size(self, signal_hash: str, quantity: int,
                             entry_price: float | None = None) -> bool:
        """Persist a rebalance's new size on the OPEN row — never a second
        insert. `signal_hash` has a unique DB constraint; remember_trade()
        always inserts, so calling it twice on the same hash (e.g. two
        rebalances of the same position) would violate that constraint.
        """
        with self._session() as s:
            row = s.execute(
                select(Trade)
                .where(Trade.signal_hash == signal_hash)
                .where(Trade.exit_time.is_(None))
            ).scalar_one_or_none()
            if row is None:
                return False
            row.quantity = quantity
            if entry_price is not None:
                row.entry_price = entry_price
            return True

    def get_open_positions(self, is_paper: bool = True) -> list[dict]:
        """Every trade with no exit — the source of truth across restarts."""
        with self._session() as s:
            rows = s.execute(
                select(Trade)
                .where(Trade.exit_time.is_(None))
                .where(Trade.is_paper == is_paper)
                .order_by(Trade.entry_time.asc())
            ).scalars().all()
            return [
                {
                    "signal_hash": r.signal_hash,
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "entry_price": float(r.entry_price) if r.entry_price else None,
                    "quantity": r.quantity,
                    "entry_time": r.entry_time,
                    "signal_score": r.signal_score,
                    "regime": r.regime,
                    "is_paper": r.is_paper,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------ #
    # Recall

    def recall_similar_conditions(self, regime: str, macro: dict | None = None,
                                  limit: int = 20) -> list[dict]:
        """Past trades taken in the same regime, most recent first."""
        with self._session() as s:
            rows = s.execute(
                select(Trade)
                .where(Trade.regime == regime)
                .order_by(Trade.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "pnl": float(r.pnl) if r.pnl is not None else None,
                    "pnl_pct": float(r.pnl_pct) if r.pnl_pct is not None else None,
                    "signal_score": r.signal_score,
                    "regime": r.regime,
                    "exit_reason": r.exit_reason,
                    "entry_time": r.entry_time.isoformat() if r.entry_time else None,
                }
                for r in rows
            ]

    def get_todays_trades(self) -> list[dict]:
        """Trades entered today (UTC date), most recent last."""
        from datetime import timedelta
        today = datetime.now(timezone.utc).date()
        return [t for t in self.get_recent_trades(days=2)
                if (t.get("entry_time") or datetime.now(timezone.utc)
                    ).date() == today]

    def get_todays_signals(self) -> list[dict]:
        """All signals recorded today: executed and rejected."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        with self._session() as s:
            rows = s.execute(
                select(Signal)
                .where(Signal.created_at >= cutoff)
                .order_by(Signal.created_at.desc())
            ).scalars().all()
            today = datetime.now(timezone.utc).date()
            return [
                {
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "score": r.score,
                    "executed": bool(r.executed),
                    "rejection_reason": r.rejection_reason,
                    "agent_votes": r.agent_votes,
                    "at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
                if r.created_at and r.created_at.date() == today
            ]

    def get_recent_trades(self, days: int = 90) -> list[dict]:
        """All closed trades from the last N calendar days, oldest first."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._session() as s:
            rows = s.execute(
                select(Trade)
                .where(Trade.created_at >= cutoff)
                .order_by(Trade.created_at.asc())
            ).scalars().all()
            return [
                {
                    "id": r.id,
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "entry_price": float(r.entry_price) if r.entry_price else None,
                    "exit_price": float(r.exit_price) if r.exit_price else None,
                    "pnl": float(r.pnl) if r.pnl is not None else None,
                    "entry_time": r.entry_time,
                    "exit_time": r.exit_time,
                    "signal_score": r.signal_score,
                    "regime": r.regime,
                    "exit_reason": r.exit_reason,
                    "is_paper": r.is_paper,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------ #
    # Agent credibility

    def get_agent_credibility(self, agent_name: str) -> float:
        with self._session() as s:
            row = s.execute(
                select(AgentCredibility).where(
                    AgentCredibility.agent_name == agent_name
                )
            ).scalar_one_or_none()
            return float(row.credibility_weight) if row else 1.0

    def update_agent_credibility(self, agent_name: str, correct: bool | None = None,
                                 was_correct: bool | None = None) -> float:
        # `was_correct` accepted as an alias (completion-loop prompt name)
        if correct is None:
            correct = bool(was_correct)
        with self._session() as s:
            row = s.execute(
                select(AgentCredibility).where(
                    AgentCredibility.agent_name == agent_name
                )
            ).scalar_one_or_none()
            if row is None:
                row = AgentCredibility(agent_name=agent_name,
                                       credibility_weight=1.0,
                                       total_calls=0, correct_calls=0)
                s.add(row)
            w = float(row.credibility_weight)
            w *= CRED_UP if correct else CRED_DOWN
            w = max(CRED_FLOOR, min(CRED_CEILING, w))
            row.credibility_weight = w
            row.total_calls = (row.total_calls or 0) + 1
            if correct:
                row.correct_calls = (row.correct_calls or 0) + 1
            row.updated_at = datetime.now(timezone.utc)
            return w

    # ------------------------------------------------------------------ #
    # Lessons

    def store_lesson(self, lesson: str, context: dict | None = None) -> str:
        return self.remember_research({
            "note_type": "lesson",
            "content": str(context or {}),
            "summary": lesson,
            "sentiment": None,
        })

    def get_lessons_learned(self, limit: int = 10) -> list[dict]:
        with self._session() as s:
            rows = s.execute(
                select(ResearchNote)
                .where(ResearchNote.note_type == "lesson")
                .order_by(ResearchNote.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {"lesson": r.summary, "context": r.content,
                 "at": r.created_at.isoformat() if r.created_at else None}
                for r in rows
            ]
