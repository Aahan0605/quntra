"""
Consecutive-loss kill switch.

After `threshold` consecutive losing trades: disable the OMS, alert via
Telegram, store a lesson in the Brain, and generate a mistake report.
Counter resets on any winning trade and at 09:15 IST each trading day.

Dependencies (brain, telegram, oms) are injected and duck-typed so the
guard is trivially testable with stubs.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class ConsecutiveLossGuard:
    def __init__(self, brain=None, telegram=None, oms=None, threshold: int = 3):
        self.brain = brain
        self.telegram = telegram
        self.oms = oms
        self.threshold = threshold
        self.counter = 0
        self.halted = False
        self.recent_losses: list[dict] = []

    # ------------------------------------------------------------------ #

    def record_trade_outcome(self, pnl: float, trade: dict | None = None) -> None:
        if pnl < 0:
            self.counter += 1
            if trade:
                self.recent_losses.append(trade)
                self.recent_losses = self.recent_losses[-self.threshold:]
        else:
            self.counter = 0
            self.recent_losses.clear()

        if self.counter >= self.threshold and not self.halted:
            self.trigger_halt()

    def trigger_halt(self) -> None:
        self.halted = True
        if self.oms is not None:
            self.oms.disable()
        report = self.generate_mistake_report()
        if self.telegram is not None:
            losses = ", ".join(
                f"{l.get('ticker', '?')} ₹{l.get('pnl', 0):+,.0f}"
                for l in self.recent_losses) or "n/a"
            analysis = (report.get("summary")
                        or str(report))[:200] if report else "pending"
            self.telegram.send(
                f"🛑 KILL SWITCH TRIGGERED\n"
                f"{self.counter} consecutive losing trades\n"
                f"OMS disabled for the rest of the session\n\n"
                f"Losses: {losses}\n"
                f"Analysis: {analysis}\n\n"
                f"Counter resets tomorrow at 09:15; halt itself needs "
                f"/resume after review."
            )
        if self.brain is not None:
            self.brain.store_lesson(
                f"{self.counter} consecutive losses triggered kill switch",
                {
                    "counter": self.counter,
                    "time": datetime.now(IST).isoformat(),
                    "report": report,
                },
            )

    def reset(self) -> None:
        """Called at 09:15 IST each trading day."""
        self.counter = 0
        self.recent_losses.clear()
        # NOTE: `halted` is NOT auto-cleared — operator must /resume.

    def resume(self) -> None:
        """Operator /resume after reviewing the mistake report."""
        self.halted = False
        self.counter = 0
        self.recent_losses.clear()
        if self.oms is not None:
            self.oms.enable()

    # ------------------------------------------------------------------ #

    def generate_mistake_report(self) -> dict:
        """
        Analyze the losing streak: regimes, signal scores, entry timing.
        Persisted to research_notes via the Brain when available.
        """
        report = {
            "type": "mistake_report",
            "generated_at": datetime.now(IST).isoformat(),
            "n_losses": self.counter,
            "trades": self.recent_losses,
            "observations": [],
        }
        regimes = [t.get("regime") for t in self.recent_losses if t.get("regime")]
        if regimes and len(set(regimes)) == 1:
            report["observations"].append(
                f"All losses in regime '{regimes[0]}' — model may be misreading it."
            )
        scores = [t.get("signal_score") for t in self.recent_losses
                  if t.get("signal_score") is not None]
        if scores and max(scores) < 9:
            report["observations"].append(
                "All losing signals scored <9/12 — consider raising MIN_SIGNAL_SCORE."
            )
        if self.brain is not None:
            self.brain.remember_research(
                {
                    "note_type": "mistake_report",
                    "ticker": None,
                    "content": str(report),
                    "summary": f"Kill switch after {self.counter} losses",
                    "sentiment": -1.0,
                }
            )
        return report
