"""
MonthlyLetter — the investment letter, 1st of each month, 9 AM IST.

An honest post-mortem: P&L attribution, what worked and what did not,
strategy vs Nifty, model performance, and the plan for next month.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.knowledge import KnowledgeManager
from src.reporting import metrics as M

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "reports" / "monthly"


class MonthlyLetter:
    def __init__(self, db_url: str | None = None, telegram=None):
        self.db_url = db_url
        self.telegram = telegram
        self.capital = float(os.getenv("DAILY_CAPITAL_INR", "25000"))

    def generate(self) -> str:
        now = datetime.now(timezone.utc)
        month_start = now - timedelta(days=30)
        month_name = datetime.now(IST).strftime("%B %Y")

        trades = M.trades_between(self.db_url, month_start, now)
        stats = M.pnl_stats(trades)
        sharpe = M.rolling_sharpe(self.db_url, capital=self.capital)
        dd = M.max_drawdown_from_pnl(self.db_url, days=30,
                                     capital=self.capital)
        cred = M.agent_credibility_table(self.db_url)
        km = KnowledgeManager(self.db_url)
        lessons = km.recall("lesson failed worked", limit=5)
        regime = M.system_state(self.db_url, "regime").get("state", "UNKNOWN")

        by_ticker = self._attribution(trades)
        winners = sorted(by_ticker.items(), key=lambda x: -x[1])[:3]
        losers = sorted(by_ticker.items(), key=lambda x: x[1])[:3]
        ret_pct = stats["net_pnl"] / self.capital

        lines = [
            f"QUNTRA MONTHLY INVESTMENT LETTER — {month_name}",
            "=" * 44,
            "P&L AND ATTRIBUTION",
            f"Net P&L: ₹{stats['net_pnl']:+,.0f} ({ret_pct:+.2%} on "
            f"₹{self.capital:,.0f})",
            f"Trades: {stats['n_trades']} | win rate "
            f"{stats['win_rate']:.0%} | max DD {dd:+.2%}",
            "Top contributors: "
            + (", ".join(f"{t} ₹{p:+,.0f}" for t, p in winners
                         if p > 0) or "none"),
            "Top detractors:  "
            + (", ".join(f"{t} ₹{p:+,.0f}" for t, p in losers
                         if p < 0) or "none"),
            "",
            "STRATEGY VS BENCHMARK",
            f"Rolling Sharpe: "
            + (f"{sharpe:.2f}" if sharpe is not None else "n/a")
            + " (paper gate needs > 1.0)",
            f"Market regime: {regime}",
            "",
            "WHAT WORKED / WHAT DID NOT (honest post-mortem)",
            *([f"• {item['content'][:140]}" for item in lessons]
              or ["• not enough history for lessons yet — "
                  "the knowledge base is still filling"]),
            "",
            "MODEL PERFORMANCE",
            *([f"• {c['agent']}: credibility {c['weight']:.2f}, "
               f"accuracy {c['accuracy']:.0%} over {c['total']} calls"
               for c in cred] or ["• no scored council calls this month"]),
            "",
            "PLAN FOR NEXT MONTH",
            *self._plan(stats, sharpe),
            "=" * 44,
        ]
        report = "\n".join(lines)
        self._archive(report)
        if self.telegram is not None:
            try:
                self.telegram.send(report)
            except Exception:  # noqa: BLE001
                pass
        return report

    @staticmethod
    def _attribution(trades) -> dict[str, float]:
        by_ticker: dict[str, float] = {}
        for t in trades:
            if t.pnl is not None:
                by_ticker[t.ticker] = by_ticker.get(t.ticker, 0.0) + float(t.pnl)
        return by_ticker

    def _plan(self, stats: dict, sharpe: float | None) -> list[str]:
        plan = ["• hold the 40-day paper gate discipline — no shortcuts"]
        if sharpe is not None and sharpe < 1.0:
            plan.append("• diagnose Sharpe shortfall: turnover? cost model? "
                        "signal quality?")
        if stats["n_closed"] and stats["win_rate"] < 0.5:
            plan.append("• review losing trades in knowledge base for "
                        "pattern (entry timing, regime mismatch)")
        plan.append("• grow organizational memory: every closed trade "
                    "stores a lesson")
        return plan

    def _archive(self, report: str) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{datetime.now(IST).strftime('%Y-%m')}.txt"
        path.write_text(report)
