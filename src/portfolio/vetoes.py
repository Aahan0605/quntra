"""Ticker-level vetoes: exclude, never select.

The backtested signal-council strategy (scripts/backtest_signal_council.py)
lost to buy-and-hold NIFTY over 5 real years — 5-day direction prediction is
not a statistical bar this system clears. Excluding an obviously bad name
is a much lower bar than predicting a good one: it needs "is this
noticeably wrong," not "will this go up." That's the only role research
agents play in position selection now — see docs/CEO_REVIEW.md Path B.

Mirrors SignalCouncil._fundamental_flagged / ._earnings_blacklist /
._news_ticker_sentiment (src/governor/council.py) but reads the DB
directly, so the passive allocator doesn't depend on the stock-picking
machinery it's replacing for live trading.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from src.db import ResearchNote, SystemState, get_session

NEGATIVE_NEWS_THRESHOLD = -0.3   # matches council.py's "clearly negative" bar


def _latest_note_payload(source: str, hours: int, db_url: str | None) -> dict:
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with get_session(db_url) as s:
            row = s.execute(
                select(ResearchNote)
                .where(ResearchNote.source == source,
                       ResearchNote.created_at >= cutoff)
                .order_by(ResearchNote.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            return (row.entities or {}) if row else {}
    except Exception:  # noqa: BLE001 — a missing note is "no veto", not a crash
        return {}


def earnings_blacklist(db_url: str | None = None) -> set[str]:
    try:
        with get_session(db_url) as s:
            row = s.get(SystemState, "earnings_blacklist")
            return set((row.value or {}).get("tickers", []) if row else [])
    except Exception:  # noqa: BLE001
        return set()


def fundamental_flagged(db_url: str | None = None) -> set[str]:
    flagged = _latest_note_payload("fundamental_agent", hours=24 * 8,
                                   db_url=db_url).get("flagged", []) or []
    return {f.get("ticker") for f in flagged if isinstance(f, dict)}


def negative_news_tickers(db_url: str | None = None,
                          threshold: float = NEGATIVE_NEWS_THRESHOLD) -> set[str]:
    sentiment = _latest_note_payload("news_agent", hours=48,
                                     db_url=db_url).get(
        "ticker_sentiment", {}) or {}
    return {t for t, s in sentiment.items() if s <= threshold}


def vetoed_tickers(db_url: str | None = None) -> set[str]:
    """Union of every exclusion reason. Vetoes only ever remove a name."""
    return (earnings_blacklist(db_url) | fundamental_flagged(db_url)
            | negative_news_tickers(db_url))
