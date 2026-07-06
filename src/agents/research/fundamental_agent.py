"""
FundamentalAgent — valuation and balance-sheet health per ticker.

P/E, P/B, ROE, debt-to-equity, EPS and revenue growth. Weekly cadence:
fundamentals don't change daily, so runs are skipped when a fresh note
(< 7 days) already exists. Primary source: yfinance info (Bharat-SM-Data
MoneyControl is the upgrade path once NSE stops 403ing).
Flags tickers whose fundamentals have deteriorated significantly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.research.base import BaseResearchAgent, ResearchOutput
from src.utils.universe import UNIVERSE

STALE_AFTER_DAYS = 7

# Deterioration flags
MAX_REASONABLE_PE = 80
MAX_DEBT_TO_EQUITY = 2.0
MIN_ROE = 0.05


class FundamentalAgent(BaseResearchAgent):
    name = "fundamental_agent"
    description = "P/E, P/B, ROE, debt ratios per ticker (weekly)"
    note_type = "fundamentals"

    def run(self, context: dict) -> ResearchOutput:
        if not context.get("force") and self._has_fresh_note():
            return ResearchOutput(
                agent=self.name,
                summary="Fundamentals fresh (< 7 days) — skipped",
                confidence=0.5,
                payload={"skipped": True},
            )

        tickers = context.get("watchlist") or UNIVERSE
        rows, flagged = [], []
        for ticker in tickers:
            data = self._fundamentals_for(ticker)
            if not data:
                continue
            warnings = self._deterioration_flags(data)
            if warnings:
                flagged.append({"ticker": ticker, "warnings": warnings})
            rows.append({"ticker": ticker, **data, "warnings": warnings})

        summary = (f"Fundamentals for {len(rows)}/{len(tickers)} tickers; "
                   f"{len(flagged)} flagged: "
                   f"{[f['ticker'] for f in flagged] or 'none'}")
        return ResearchOutput(
            agent=self.name,
            summary=summary,
            findings=rows,
            confidence=0.6 if rows else 0.0,
            sources=["yfinance info"],
            reasoning=f"flags: P/E>{MAX_REASONABLE_PE}, "
                      f"D/E>{MAX_DEBT_TO_EQUITY}, ROE<{MIN_ROE:.0%}",
            payload={"flagged": flagged, "n_covered": len(rows)},
        )

    def _has_fresh_note(self) -> bool:
        try:
            from sqlalchemy import select
            from src.db import ResearchNote, get_session
            cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_AFTER_DAYS)
            with get_session(self.db_url) as s:
                row = s.execute(
                    select(ResearchNote)
                    .where(ResearchNote.source == self.name,
                           ResearchNote.created_at >= cutoff)
                    .limit(1)
                ).scalar_one_or_none()
                return row is not None
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _fundamentals_for(ticker: str) -> dict | None:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            if not info.get("trailingPE") and not info.get("priceToBook"):
                return None
            return {
                "pe": info.get("trailingPE"),
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": (info.get("debtToEquity") / 100
                                   if info.get("debtToEquity") else None),
                "eps": info.get("trailingEps"),
                "revenue_growth": info.get("revenueGrowth"),
            }
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _deterioration_flags(d: dict) -> list[str]:
        flags = []
        if d.get("pe") and d["pe"] > MAX_REASONABLE_PE:
            flags.append(f"P/E stretched at {d['pe']:.0f}")
        if d.get("debt_to_equity") and d["debt_to_equity"] > MAX_DEBT_TO_EQUITY:
            flags.append(f"D/E high at {d['debt_to_equity']:.2f}")
        if d.get("roe") is not None and d["roe"] < MIN_ROE:
            flags.append(f"ROE weak at {d['roe']:.1%}")
        if d.get("revenue_growth") is not None and d["revenue_growth"] < -0.05:
            flags.append(f"revenue shrinking {d['revenue_growth']:.1%}")
        return flags
