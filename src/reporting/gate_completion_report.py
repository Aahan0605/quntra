"""
GateCompletionReport — the full day-by-day paper trading history, sent once
the 40-day gate is reached (pass or fail — the operator needs to see both).

Reuses scripts.paper_trading_status.gather_stats() for the same numbers
(Sharpe rf=0, max DD, gate checks) the /paper_progress command already
shows, so this report can never disagree with the dashboard.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.db import Trade, get_session

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "reports" / "gate_completion"
GATE_SENT_KEY = "gate_completion_report_sent"


class GateCompletionReport:
    def __init__(self, db_url: str | None = None, telegram=None):
        self.db_url = db_url
        self.telegram = telegram

    def generate(self) -> str | None:
        """Full report text, or None when there's no trade history yet."""
        from scripts.paper_trading_status import gather_stats
        st = gather_stats()
        if st is None:
            return None

        day_log = self._day_by_day()
        all_pass = st["gate_days"] and st["gate_sharpe"] and st["gate_dd"]
        header = "🏁 QUNTRA PAPER TRADING GATE" + (
            " — COMPLETE ✅" if all_pass else " — 40 DAYS REACHED"
            if st["gate_days"] else " — PROGRESS SNAPSHOT")

        lines = [
            header,
            "=" * 34,
            f"Period: {day_log[0][0]} to {day_log[-1][0]} "
            f"({len(day_log)} trading days)" if day_log else "No trades yet",
            "",
            "GATE RESULT",
            f"  40 trading days:  " +
            ("PASS ✓" if st["gate_days"] else f"PENDING ({st['days']}/40)"),
            f"  Sharpe > 1.0:      " +
            (f"PASS ✓ ({st['sharpe']:.3f})" if st["gate_sharpe"]
             else "FAIL/PENDING" if np.isnan(st["sharpe"])
             else f"FAIL ({st['sharpe']:.3f})"),
            f"  Max DD > -15%:     " +
            (f"PASS ✓ ({st['max_dd']:+.2%})" if st["gate_dd"]
             else f"FAIL ({st['max_dd']:+.2%})"),
            "",
            "OVERALL",
            f"  Total trades: {st['n_entered']} "
            f"({st['n_closed']} closed, {st['n_entered'] - st['n_closed']} open)",
            f"  Net P&L: ₹{st['total_pnl']:+,.0f}",
            f"  Win rate: {st['win_rate']:.1%} ({st['wins']}W / {st['losses']}L)",
            "",
            "DAY-BY-DAY LOG",
        ]
        lines += [f"  {d}: {n} trade(s), P&L ₹{pnl:+,.0f}, "
                  f"{w}W/{l}L" for d, n, pnl, w, l in day_log] or \
                 ["  (no trades recorded)"]
        lines += ["=" * 34]
        if not all_pass and st["gate_days"]:
            lines.append("⚠️ 40 days elapsed but Sharpe/DD gate NOT met — "
                         "do not proceed to live capital. Review the log "
                         "above and extend paper trading.")
        report = "\n".join(lines)
        self._archive(report)
        return report

    def _day_by_day(self) -> list[tuple[str, int, float, int, int]]:
        """(date, n_trades, day_pnl, wins, losses) per trading day."""
        with get_session(self.db_url) as s:
            rows = s.execute(
                select(Trade).where(Trade.is_paper.is_(True))
                .order_by(Trade.entry_time)
            ).scalars().all()
        if not rows:
            return []
        df = pd.DataFrame([{
            "date": (r.entry_time or r.created_at).date(),
            "pnl": float(r.pnl) if r.pnl is not None else np.nan,
        } for r in rows])
        out = []
        for day, g in df.groupby("date"):
            closed = g.dropna(subset=["pnl"])
            out.append((
                day.isoformat(), len(g), float(closed["pnl"].sum()),
                int((closed["pnl"] > 0).sum()), int((closed["pnl"] <= 0).sum()),
            ))
        return sorted(out, key=lambda x: x[0])

    def send_if_gate_reached(self) -> bool:
        """Called daily (post-EOD). Sends the report exactly once, the day
        the 40-trading-day mark is first reached. Returns True if sent."""
        from src.db import SystemState

        with get_session(self.db_url) as s:
            already = s.get(SystemState, GATE_SENT_KEY)
            if already and already.value and already.value.get("sent"):
                return False

        report = self.generate()
        if report is None:
            return False

        from scripts.paper_trading_status import gather_stats
        st = gather_stats()
        if st is None or not st["gate_days"]:
            return False  # not yet at 40 trading days — nothing to send

        if self.telegram is not None:
            self.telegram.send(report)
        with get_session(self.db_url) as s:
            s.merge(SystemState(key=GATE_SENT_KEY, value={
                "sent": True, "at": datetime.now(IST).isoformat(),
            }))
        return True

    def _archive(self, report: str) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{datetime.now(IST).strftime('%Y-%m-%d_%H%M')}.txt"
        path.write_text(report)
