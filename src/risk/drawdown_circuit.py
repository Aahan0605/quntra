"""
QuNtra drawdown circuit breaker — recalibrated for NSE volatility.

Old thresholds (-2% intraday) sat inside normal NSE daily volatility
(1.5–3%), so the system squared positions at the worst prices and
re-entered higher, compounding losses instead of limiting them.

Recalibrated levels:
  Level 1  -3.0% intraday      -> tighten trailing stops, no new entries,
                                  30-minute cooldown before re-entry
  Level 2  -4.5% intraday      -> square all positions, OMS disabled
                                  for the rest of the session
  Level 3  -7.0% rolling 5-day -> full halt, manual /resume required

Plus: 3 consecutive losing trades -> halt + analysis mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum


class CircuitLevel(IntEnum):
    NORMAL = 0
    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3


LEVEL1_INTRADAY_DD = -0.030
LEVEL2_INTRADAY_DD = -0.045
LEVEL3_ROLLING_5D_DD = -0.070
COOLDOWN_MINUTES = 30
CONSECUTIVE_LOSS_LIMIT = 3


@dataclass
class CircuitState:
    level: CircuitLevel = CircuitLevel.NORMAL
    tighten_stops: bool = False
    square_all: bool = False
    oms_disabled: bool = False
    manual_resume_required: bool = False
    cooldown_until: datetime | None = None
    consecutive_losses: int = 0
    halted_by_losses: bool = False
    events: list[str] = field(default_factory=list)


class DrawdownCircuitBreaker:
    def __init__(
        self,
        level1_dd: float = LEVEL1_INTRADAY_DD,
        level2_dd: float = LEVEL2_INTRADAY_DD,
        level3_dd: float = LEVEL3_ROLLING_5D_DD,
        cooldown_minutes: int = COOLDOWN_MINUTES,
        consecutive_loss_limit: int = CONSECUTIVE_LOSS_LIMIT,
    ):
        if not (level3_dd < level2_dd < level1_dd < 0):
            raise ValueError("Thresholds must satisfy level3 < level2 < level1 < 0")
        self.level1_dd = level1_dd
        self.level2_dd = level2_dd
        self.level3_dd = level3_dd
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.loss_limit = consecutive_loss_limit
        self.state = CircuitState()

    # ------------------------------------------------------------------ #

    def update(
        self,
        intraday_dd: float,
        rolling_5d_dd: float,
        now: datetime,
    ) -> CircuitState:
        """
        Feed current drawdowns (negative fractions, e.g. -0.031 = -3.1%).
        Escalates level; never de-escalates within a session.
        """
        s = self.state

        if rolling_5d_dd <= self.level3_dd and s.level < CircuitLevel.LEVEL3:
            s.level = CircuitLevel.LEVEL3
            s.square_all = True
            s.oms_disabled = True
            s.manual_resume_required = True
            s.events.append(f"{now.isoformat()} LEVEL3 rolling5d={rolling_5d_dd:.3%}")
        elif intraday_dd <= self.level2_dd and s.level < CircuitLevel.LEVEL2:
            s.level = CircuitLevel.LEVEL2
            s.square_all = True
            s.oms_disabled = True
            s.events.append(f"{now.isoformat()} LEVEL2 intraday={intraday_dd:.3%}")
        elif intraday_dd <= self.level1_dd and s.level < CircuitLevel.LEVEL1:
            s.level = CircuitLevel.LEVEL1
            s.tighten_stops = True
            s.cooldown_until = now + self.cooldown
            s.events.append(f"{now.isoformat()} LEVEL1 intraday={intraday_dd:.3%}")

        return s

    def record_trade_result(self, pnl: float, now: datetime) -> CircuitState:
        s = self.state
        if pnl < 0:
            s.consecutive_losses += 1
        else:
            s.consecutive_losses = 0
        if s.consecutive_losses >= self.loss_limit and not s.halted_by_losses:
            s.halted_by_losses = True
            s.oms_disabled = True
            s.events.append(
                f"{now.isoformat()} HALT {s.consecutive_losses} consecutive losses"
            )
        return s

    # ------------------------------------------------------------------ #

    def can_enter_new_position(self, now: datetime) -> bool:
        s = self.state
        if s.oms_disabled or s.halted_by_losses or s.manual_resume_required:
            return False
        if s.level >= CircuitLevel.LEVEL1:
            if s.cooldown_until is not None and now < s.cooldown_until:
                return False
            # After cooldown at Level 1, cautious re-entry is allowed
            # but stops remain tightened.
            return s.level == CircuitLevel.LEVEL1
        return True

    def manual_resume(self) -> None:
        """Operator /resume — the only way out of Level 3 or a loss halt."""
        self.state = CircuitState()

    def reset_daily(self) -> None:
        """Called at 09:15 IST. Level 3 and loss halts survive the reset."""
        if self.state.manual_resume_required or self.state.halted_by_losses:
            self.state.consecutive_losses = 0
            return
        self.state = CircuitState()
