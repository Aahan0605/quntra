"""
QuNtra portfolio rebalancer — weekly cadence with drift threshold.

Fixes the turnover problem that destroyed the Sharpe ratio after costs:
daily rebalancing across 25 tickers at ~0.15% round-trip produced
~1,200% annual turnover (Sharpe +1.11 -> -4.78 after ICICI costs).

Policy:
  * Rebalance at most WEEKLY (first trading day of each ISO week).
  * Skip any ticker whose |current - target| weight drift <= 3%.
  * Cap total one-way turnover per rebalance at 20% of portfolio value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

DRIFT_THRESHOLD = 0.03  # 3% minimum weight drift before a ticker is traded
MAX_TURNOVER = 0.20     # 20% max one-way turnover per rebalance event
FREQUENCY = "weekly"


@dataclass
class RebalanceDecision:
    should_rebalance: bool
    trades: dict[str, float] = field(default_factory=dict)  # ticker -> weight delta
    one_way_turnover: float = 0.0
    reason: str = ""


class Rebalancer:
    def __init__(
        self,
        drift_threshold: float = DRIFT_THRESHOLD,
        max_turnover: float = MAX_TURNOVER,
        frequency: str = FREQUENCY,
    ):
        if frequency not in ("daily", "weekly", "monthly"):
            raise ValueError(f"Unsupported frequency: {frequency}")
        self.drift_threshold = drift_threshold
        self.max_turnover = max_turnover
        self.frequency = frequency
        self._last_rebalance: date | None = None

    # ------------------------------------------------------------------ #

    def is_rebalance_day(self, today: date) -> bool:
        """True when a new period starts relative to the last rebalance."""
        if self._last_rebalance is None:
            return True
        if self.frequency == "daily":
            return today > self._last_rebalance
        if self.frequency == "weekly":
            return today.isocalendar()[:2] != self._last_rebalance.isocalendar()[:2]
        # monthly
        return (today.year, today.month) != (
            self._last_rebalance.year,
            self._last_rebalance.month,
        )

    def compute_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        today: date,
    ) -> RebalanceDecision:
        """
        Decide which tickers to trade. Applies frequency gate, per-ticker
        drift filter, and portfolio-level turnover cap (scales all trades
        down proportionally if the cap would be exceeded).
        """
        if not self.is_rebalance_day(today):
            return RebalanceDecision(False, reason="not a rebalance day")

        tickers = set(current_weights) | set(target_weights)
        deltas = {
            t: target_weights.get(t, 0.0) - current_weights.get(t, 0.0)
            for t in tickers
        }
        # Drift filter: only trade meaningful deviations
        trades = {t: d for t, d in deltas.items() if abs(d) > self.drift_threshold}
        if not trades:
            return RebalanceDecision(False, reason="all drifts within threshold")

        one_way = 0.5 * sum(abs(d) for d in trades.values())
        if one_way > self.max_turnover:
            scale = self.max_turnover / one_way
            trades = {t: d * scale for t, d in trades.items()}
            one_way = self.max_turnover

        self._last_rebalance = today
        return RebalanceDecision(True, trades=trades, one_way_turnover=one_way,
                                 reason="rebalanced")

    # ------------------------------------------------------------------ #

    def simulate_annual_turnover(
        self,
        returns: pd.DataFrame,
        target_weights: dict[str, float],
    ) -> float:
        """
        Simulate one pass over a daily returns panel with static target
        weights, letting weights drift with returns and rebalancing per
        policy. Returns annualized one-way turnover (e.g. 2.5 = 250%).
        """
        self._last_rebalance = None
        cols = list(returns.columns)
        target = np.array([target_weights.get(t, 0.0) for t in cols])
        target = target / target.sum()
        current = target.copy()

        total_turnover = 0.0
        for ts, row in returns.iterrows():
            # weights drift with daily returns
            growth = current * (1.0 + row.values)
            denom = growth.sum()
            if denom <= 0:
                continue
            current = growth / denom

            decision = self.compute_trades(
                dict(zip(cols, current)), dict(zip(cols, target)),
                ts.date() if hasattr(ts, "date") else ts,
            )
            if decision.should_rebalance:
                total_turnover += decision.one_way_turnover
                for i, t in enumerate(cols):
                    current[i] += decision.trades.get(t, 0.0)
                current = np.clip(current, 0, None)
                current = current / current.sum()

        years = len(returns) / 252.0
        return total_turnover / years if years > 0 else 0.0


# Alias used by the completion-loop prompts (weekly is the default policy)
WeeklyRebalancer = Rebalancer
