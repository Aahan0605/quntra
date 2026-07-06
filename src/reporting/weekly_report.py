"""
WeeklyReport — the internal board report, Sundays 8 PM IST.

Research highlights come from the knowledge base; performance from the
trades table; agent credibility deltas from agent_credibility.
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
OUT_DIR = ROOT / "data" / "reports" / "weekly"


class WeeklyReport:
    def __init__(self, db_url: str | None = None, telegram=None):
        self.db_url = db_url
        self.telegram = telegram
        self.capital = float(os.getenv("DAILY_CAPITAL_INR", "25000"))

    def generate(self) -> str:
        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=7)
        week_num = datetime.now(IST).isocalendar().week

        trades = M.trades_between(self.db_url, week_start, now)
        stats = M.pnl_stats(trades)
        sharpe = M.rolling_sharpe(self.db_url, capital=self.capital)
        nifty = M.nifty_move_today()
        cred = M.agent_credibility_table(self.db_url)
        km = KnowledgeManager(self.db_url)
        digest = km.generate_knowledge_digest(days=7)
        health = M.system_state(self.db_url, "health")
        regime = M.system_state(self.db_url, "regime").get("state", "UNKNOWN")
        n_knowledge = M.knowledge_count(self.db_url)

        improved = [c for c in cred if c["weight"] > 1.0]
        degraded = [c for c in cred if c["weight"] < 1.0]

        lines = [
            f"QUNTRA INTERNAL BOARD REPORT — Week {week_num}",
            "=" * 40,
            "PERFORMANCE",
            f"Week's trades: {stats['n_trades']} | "
            f"P&L: ₹{stats['net_pnl']:+,.0f} | "
            f"win rate {stats['win_rate']:.0%}",
            f"Rolling 30d Sharpe: "
            + (f"{sharpe:.2f}" if sharpe is not None else "n/a"),
            f"Nifty last session: "
            + (f"{nifty:+.2f}%" if nifty is not None else "n/a"),
            "",
            "RESEARCH HIGHLIGHTS",
            *digest.splitlines()[:12],
            "",
            "STRATEGY HEALTH",
            f"Regime: {regime}",
            f"Knowledge items total: {n_knowledge} (paper gate needs ≥ 50)",
            "",
            "AGENT PERFORMANCE",
            *([f"↑ {c['agent']}: {c['weight']:.2f} "
               f"{M.credibility_bar(c['weight'])}" for c in improved]
              + [f"↓ {c['agent']}: {c['weight']:.2f} "
                 f"{M.credibility_bar(c['weight'])}" for c in degraded]
              or ["no credibility movement yet"]),
            "",
            "INFRASTRUCTURE",
            f"Last health check: "
            f"{'OK' if health.get('db') else 'DEGRADED'} "
            f"at {health.get('at', 'never')}",
            "",
            "RISKS",
            *self._open_risks(stats, sharpe),
            "",
            "NEXT WEEK PRIORITIES",
            *self._priorities(stats, n_knowledge),
            "=" * 40,
        ]
        report = "\n".join(lines)
        self._archive(report, week_num)
        if self.telegram is not None:
            try:
                self.telegram.send(report)
            except Exception:  # noqa: BLE001
                pass
        return report

    def _open_risks(self, stats: dict, sharpe: float | None) -> list[str]:
        risks = []
        if stats["n_closed"] and stats["win_rate"] < 0.45:
            risks.append(f"• win rate {stats['win_rate']:.0%} below 45%")
        if sharpe is not None and sharpe < 1.0:
            risks.append(f"• rolling Sharpe {sharpe:.2f} below paper-gate 1.0")
        if stats["net_pnl"] < 0:
            risks.append(f"• losing week (₹{stats['net_pnl']:+,.0f})")
        return risks or ["• none flagged this week"]

    def _priorities(self, stats: dict, n_knowledge: int) -> list[str]:
        prios = []
        if stats["n_trades"] == 0:
            prios.append("• investigate zero signal generation "
                         "(score threshold? regime?)")
        if n_knowledge < 50:
            prios.append(f"• grow knowledge base ({n_knowledge}/50 "
                         f"toward paper gate)")
        prios.append("• continue 40-day paper gate accumulation")
        return prios

    def _archive(self, report: str, week_num: int) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{datetime.now(IST).date()}_W{week_num}.txt"
        path.write_text(report)
