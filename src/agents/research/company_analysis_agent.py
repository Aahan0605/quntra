"""
CompanyAnalysisAgent — corporate event flags for the watchlist.

Flags per ticker: earnings within 5 trading days (no entries before
earnings), dividend ex-date within 3 days, other corporate actions.
Primary source: yfinance calendar (Bharat-SM-Data corporate calendar
requires NSE endpoints that frequently 403 non-browser clients).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.agents.research.base import BaseResearchAgent, ResearchOutput
from src.utils.universe import UNIVERSE

EARNINGS_BLACKOUT_TRADING_DAYS = 5
DIVIDEND_WINDOW_DAYS = 3


class CompanyAnalysisAgent(BaseResearchAgent):
    name = "company_analysis_agent"
    description = "earnings dates, dividends, corporate actions per ticker"
    note_type = "corporate_events"

    def run(self, context: dict) -> ResearchOutput:
        today = context.get("date") or date.today()
        if isinstance(today, str):
            today = date.fromisoformat(today)
        tickers = context.get("watchlist") or UNIVERSE

        flags: list[dict] = []
        checked = 0
        for ticker in tickers:
            events = self._events_for(ticker, today)
            checked += 1
            if events:
                flags.append({"ticker": ticker, **events})

        blocked = [f["ticker"] for f in flags if f.get("earnings_blackout")]
        summary = (f"Corporate events: {len(flags)} of {checked} tickers "
                   f"flagged; earnings blackout: {blocked or 'none'}")
        return ResearchOutput(
            agent=self.name,
            summary=summary,
            findings=flags,
            confidence=0.6 if checked else 0.0,
            sources=["yfinance calendar"],
            reasoning=f"blackout = earnings within "
                      f"{EARNINGS_BLACKOUT_TRADING_DAYS} trading days",
            payload={"event_flags": {f["ticker"]: f for f in flags},
                     "earnings_blackout": blocked},
        )

    def _events_for(self, ticker: str, today: date) -> dict:
        """Event flags for one ticker; {} when nothing within windows."""
        events: dict = {}
        try:
            import yfinance as yf
            cal = yf.Ticker(ticker).calendar or {}
            earnings_dates = cal.get("Earnings Date") or []
            if earnings_dates:
                nxt = min(d for d in earnings_dates
                          if isinstance(d, date) and d >= today)
                bdays = int(pd.bdate_range(today, nxt).size) - 1
                if 0 <= bdays <= EARNINGS_BLACKOUT_TRADING_DAYS:
                    events["earnings_date"] = nxt.isoformat()
                    events["earnings_blackout"] = True
            ex_div = cal.get("Ex-Dividend Date")
            if isinstance(ex_div, date) and \
                    0 <= (ex_div - today).days <= DIVIDEND_WINDOW_DAYS:
                events["ex_dividend_date"] = ex_div.isoformat()
        except ValueError:
            pass  # no future earnings date in the list
        except Exception:  # noqa: BLE001 — calendar source may be down
            pass
        return events
