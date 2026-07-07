"""
Base interface for QuNtra research agents.

Every agent has exactly one responsibility, runs independently, degrades
gracefully when its data source is down, and stores findings to the
research_notes table. Hermes composes them — agents never call each other.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("quntra.research")

# NSE/RSS endpoints tarpit blocked clients — nothing may hang the pipeline
socket.setdefaulttimeout(15)


@dataclass
class ResearchOutput:
    """Uniform output envelope for every research agent."""
    agent: str
    summary: str                      # one readable paragraph
    findings: list[dict] = field(default_factory=list)
    confidence: float = 0.5           # 0-1: how much to trust this output
    sources: list[str] = field(default_factory=list)
    reasoning: str = ""
    payload: dict = field(default_factory=dict)   # agent-specific extras
    error: str | None = None
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseResearchAgent:
    """Contract: run(context) -> ResearchOutput; store() persists it."""

    name: str = "base"
    description: str = "abstract research agent"
    note_type: str = "research"

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url

    def run(self, context: dict) -> ResearchOutput:
        """context: today's date, current regime, watchlist, positions."""
        raise NotImplementedError

    def safe_run(self, context: dict) -> ResearchOutput:
        """run() that never raises — Hermes calls this."""
        try:
            return self.run(context)
        except Exception as e:  # noqa: BLE001
            logger.exception("%s failed", self.name)
            return ResearchOutput(
                agent=self.name,
                summary=f"{self.name} unavailable: {e}",
                confidence=0.0,
                error=str(e),
            )

    def store(self, output: ResearchOutput) -> str | None:
        """Persist findings to the research_notes table."""
        try:
            from src.db import ResearchNote, get_session
            with get_session(self.db_url) as s:
                row = ResearchNote(
                    note_type=self.note_type,
                    content="\n".join(str(f) for f in output.findings)
                            or output.summary,
                    summary=output.summary,
                    source=self.name,
                    confidence=output.confidence,
                    entities=output.payload or None,
                )
                s.add(row)
                s.flush()
                return row.id
        except Exception as e:  # noqa: BLE001
            logger.error("%s could not store research note: %s", self.name, e)
            return None


def fetch_rss(url: str, limit: int = 30) -> list[dict]:
    """Fetch and normalize an RSS feed. Returns [] on any failure.

    published_dt is a datetime when the feed provides a parseable
    timestamp, else None — callers use it for freshness filtering.
    """
    try:
        import feedparser
        parsed = feedparser.parse(url)
        out = []
        for e in (parsed.entries or [])[:limit]:
            published_dt = None
            struct = (getattr(e, "published_parsed", None)
                      or getattr(e, "updated_parsed", None))
            if struct is not None:
                try:
                    published_dt = datetime(*struct[:6], tzinfo=timezone.utc)
                except Exception:  # noqa: BLE001
                    pass
            out.append({
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", "")[:500],
                "link": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
                "published_dt": published_dt,
            })
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("RSS fetch failed for %s: %s", url, e)
        return []


def yf_pct_change(ticker: str, period: str = "5d") -> float | None:
    """Last close-over-close %change via yfinance. None on failure."""
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period=period)
        if len(h) >= 2:
            return round(float(h["Close"].iloc[-1] / h["Close"].iloc[-2] - 1) * 100, 2)
    except Exception as e:  # noqa: BLE001
        logger.warning("yfinance %s failed: %s", ticker, e)
    return None
