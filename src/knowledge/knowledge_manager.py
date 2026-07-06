"""
QuNtra Knowledge Manager — organizational memory.

The objective is not storing data; it is building organizational memory:
every lesson, observation, and insight becomes a queryable KnowledgeItem
that future decisions can recall by keyword, regime, ticker, or similar
macro conditions.

Search is keyword-based over PostgreSQL/SQLite (portable ILIKE matching).
Phase 3 upgrades recall() to semantic search via Qdrant — the interface
is stable so callers won't change.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select

from src.db import KnowledgeItem, get_session

logger = logging.getLogger("quntra.knowledge")

KNOWLEDGE_TYPES = {
    "TRADE_LESSON",         # what worked / what failed + why
    "MARKET_OBSERVATION",   # how specific events affected Indian markets
    "STRATEGY_INSIGHT",     # pattern discovered in backtesting
    "COMPANY_RESEARCH",     # fundamental findings per ticker
    "MACRO_OBSERVATION",    # how global events rippled into NSE/BSE
    "GEOPOLITICAL_NOTE",    # verified geopolitical events + market impact
    "MODEL_VERSION",        # model changes + performance delta
    "OVERNIGHT_RESEARCH",   # outputs of the overnight pipeline
}

# Words too common to be useful search terms
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "with", "by", "at", "it", "this", "that",
}


class KnowledgeManager:
    """Stores, indexes, and retrieves structured knowledge."""

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url

    def _session(self):
        return get_session(self.db_url)

    # ------------------------------------------------------------------ #
    # Store

    def store(self, knowledge_type: str, content: str,
              tickers: list | None = None, confidence: float = 0.5,
              source: str | None = None, regime: str | None = None,
              conditions: dict | None = None) -> str:
        """Store a knowledge item. Returns knowledge_id."""
        if knowledge_type not in KNOWLEDGE_TYPES:
            logger.warning("Unknown knowledge_type %r — storing anyway",
                           knowledge_type)
        row = KnowledgeItem(
            knowledge_type=knowledge_type,
            content=content,
            tickers=tickers or [],
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            regime=regime,
            conditions=conditions or {},
        )
        with self._session() as s:
            s.add(row)
            s.flush()
            return row.id

    # ------------------------------------------------------------------ #
    # Recall

    def recall(self, query: str, knowledge_type: str | None = None,
               limit: int = 10) -> list[dict]:
        """Keyword recall: items matching any non-stopword query term,
        ranked by number of matched terms then recency."""
        terms = [t for t in query.lower().split()
                 if t not in _STOPWORDS and len(t) >= 3]
        if not terms:
            return []
        with self._session() as s:
            stmt = select(KnowledgeItem)
            if knowledge_type:
                stmt = stmt.where(KnowledgeItem.knowledge_type == knowledge_type)
            stmt = stmt.where(or_(
                *[KnowledgeItem.content.ilike(f"%{t}%") for t in terms]
            )).order_by(KnowledgeItem.created_at.desc()).limit(limit * 3)
            rows = s.execute(stmt).scalars().all()
            scored = sorted(
                rows,
                key=lambda r: (-sum(t in r.content.lower() for t in terms),
                               -(r.created_at.timestamp() if r.created_at else 0)),
            )
            return [self._to_dict(r) for r in scored[:limit]]

    def recall_by_regime(self, regime: str, limit: int = 20) -> list[dict]:
        """What happened last time we were in this regime?"""
        with self._session() as s:
            rows = s.execute(
                select(KnowledgeItem)
                .where(KnowledgeItem.regime == regime)
                .order_by(KnowledgeItem.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [self._to_dict(r) for r in rows]

    def recall_by_ticker(self, ticker: str, limit: int = 20) -> list[dict]:
        """What do we know about this company?

        tickers is a JSON list — portable containment check happens in
        Python after a cheap LIKE prefilter on the serialized column.
        """
        with self._session() as s:
            rows = s.execute(
                select(KnowledgeItem)
                .order_by(KnowledgeItem.created_at.desc())
                .limit(500)
            ).scalars().all()
            hits = [r for r in rows if ticker in (r.tickers or [])]
            return [self._to_dict(r) for r in hits[:limit]]

    def recall_similar_market_conditions(self, conditions: dict,
                                         limit: int = 10) -> list[dict]:
        """Find historical episodes with similar macro conditions.

        conditions: e.g. {"regime": "BULL_TRENDING", "vix_high": True,
        "macro_bias": "NEGATIVE"}. Similarity = count of matching keys.
        """
        with self._session() as s:
            rows = s.execute(
                select(KnowledgeItem)
                .where(KnowledgeItem.conditions.is_not(None))
                .order_by(KnowledgeItem.created_at.desc())
                .limit(500)
            ).scalars().all()

        def similarity(item: KnowledgeItem) -> int:
            stored = item.conditions or {}
            return sum(1 for k, v in conditions.items() if stored.get(k) == v)

        scored = [(similarity(r), r) for r in rows]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: (-x[0], -(x[1].created_at.timestamp()
                                            if x[1].created_at else 0)))
        return [{**self._to_dict(r), "similarity": sc}
                for sc, r in scored[:limit]]

    # ------------------------------------------------------------------ #
    # Digest

    def generate_knowledge_digest(self, days: int = 7) -> str:
        """Weekly digest: what did QuNtra learn this week?"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._session() as s:
            rows = s.execute(
                select(KnowledgeItem)
                .where(KnowledgeItem.created_at >= cutoff)
                .order_by(KnowledgeItem.knowledge_type,
                          KnowledgeItem.created_at.desc())
            ).scalars().all()

        if not rows:
            return (f"KNOWLEDGE DIGEST (last {days} days)\n"
                    f"No new knowledge items recorded.")

        by_type: dict[str, list[KnowledgeItem]] = {}
        for r in rows:
            by_type.setdefault(r.knowledge_type, []).append(r)

        lines = [f"KNOWLEDGE DIGEST (last {days} days) — "
                 f"{len(rows)} new items"]
        for ktype, items in sorted(by_type.items()):
            lines.append(f"\n{ktype} ({len(items)}):")
            for item in items[:5]:
                snippet = item.content.replace("\n", " ")[:160]
                conf = float(item.confidence or 0)
                lines.append(f"  • [{conf:.0%}] {snippet}")
            if len(items) > 5:
                lines.append(f"  … and {len(items) - 5} more")
        return "\n".join(lines)

    def count(self) -> int:
        """Total stored knowledge items (paper-gate requires >= 50)."""
        from sqlalchemy import func
        with self._session() as s:
            return int(s.execute(
                select(func.count()).select_from(KnowledgeItem)
            ).scalar_one())

    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_dict(r: KnowledgeItem) -> dict:
        return {
            "id": r.id,
            "knowledge_type": r.knowledge_type,
            "content": r.content,
            "tickers": r.tickers or [],
            "regime": r.regime,
            "confidence": float(r.confidence) if r.confidence is not None else 0.5,
            "source": r.source,
            "conditions": r.conditions or {},
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
