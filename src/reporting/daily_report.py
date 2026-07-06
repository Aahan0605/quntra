"""
DailyReport — the 5 PM IST end-of-day report.

All metrics from PostgreSQL via src.reporting.metrics; sent over
Telegram and archived to data/reports/daily/.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.reporting import metrics as M

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "reports" / "daily"


class DailyReport:
    def __init__(self, db_url: str | None = None, telegram=None):
        self.db_url = db_url
        self.telegram = telegram
        self.capital = float(os.getenv("DAILY_CAPITAL_INR", "25000"))
        self.max_trades = int(os.getenv("MAX_TRADES_PER_DAY", "3"))

    def generate(self) -> str:
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades = M.trades_between(self.db_url, day_start, now)
        stats = M.pnl_stats(trades)
        open_positions = [t for t in trades if t.exit_time is None]
        deployed = sum(
            float(t.entry_price or 0) * (t.quantity or 0)
            for t in open_positions
        )
        sharpe = M.rolling_sharpe(self.db_url, capital=self.capital)
        dd90 = M.max_drawdown_from_pnl(self.db_url, capital=self.capital)
        cred = M.agent_credibility_table(self.db_url)
        regime = M.system_state(self.db_url, "regime").get("state", "UNKNOWN")
        premarket = M.system_state(self.db_url, "premarket_draft")
        research = M.research_notes_since(self.db_url, hours=24)
        nifty = M.nifty_move_today()

        today_dd = min(0.0, stats["net_pnl"] / self.capital)
        lines = [
            f"📊 QUNTRA DAILY REPORT — {datetime.now(IST).date()}",
            "=" * 32,
            f"REGIME: {regime}",
            f"MARKET: Nifty 50 {nifty:+.2f}%" if nifty is not None
            else "MARKET: Nifty 50 n/a",
            "",
            "TODAY'S TRADING",
            f"Trades executed: {stats['n_trades']} / {self.max_trades} max",
            f"Net P&L: ₹{stats['net_pnl']:+,.0f}",
            f"Winners: {stats['n_wins']} | Losers: {stats['n_losses']} | "
            f"Win rate: {stats['win_rate']:.0%}",
            "",
            "PORTFOLIO",
            f"Capital deployed: ₹{deployed:,.0f} / ₹{self.capital:,.0f}",
            f"Open positions: {len(open_positions)}",
            f"Today's drawdown: {today_dd:+.2%}",
            "",
            "ROLLING METRICS (30 days)",
            f"Sharpe: {sharpe:.2f}" if sharpe is not None
            else "Sharpe: n/a (< 5 trading days)",
            f"Max DD (90d): {dd90:+.2%}",
            "",
            "AGENT COUNCIL CREDIBILITY",
            *([f"{c['agent']}: {c['weight']:.2f} "
               f"({c['accuracy']:.0%} of {c['total']})"
               for c in cred] or ["no council calls scored yet"]),
            "",
            "RESEARCH COMPLETED (24h)",
            *([f"• [{r['source']}] {r['summary']}" for r in research[:6]]
              or ["none recorded"]),
            "",
            "TOMORROW'S BIAS",
            (premarket.get("report", "").split("RECOMMENDATION:")[-1].strip()
             if premarket.get("report") else "pre-market draft not ready yet"),
            "=" * 32,
        ]
        report = "\n".join(lines)
        self._archive(report)
        if self.telegram is not None:
            try:
                self.telegram.send(report)
            except Exception:  # noqa: BLE001
                pass
        return report

    def _archive(self, report: str) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{datetime.now(IST).date()}.txt"
        path.write_text(report)
