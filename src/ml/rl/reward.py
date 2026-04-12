"""
Reward Shaper — Sophisticated reward function for RL trading agent.
====================================================================

Principles:
  1. Reward risk-adjusted returns, not raw returns
  2. Penalize losses HARDER than reward gains (loss aversion 2x)
  3. Penalize holding losing positions too long
  4. Reward hitting targets at the right time
  5. Penalize overtrading (costs accumulate)
  6. Reward respecting stop losses (discipline)
"""

import numpy as np


class RewardShaper:
    """
    Computes shaped rewards that teach the RL agent what we care about.

    The reward function is the most critical component — it defines
    what behavior the agent learns. Each component targets a specific
    aspect of good trading behavior.
    """

    def __init__(self, loss_multiplier: float = 2.0,
                 holding_cost: float = 0.001,
                 sl_reward: float = 0.05,
                 target_reward: float = 0.10,
                 overtrade_penalty: float = 0.02,
                 max_trades_per_day: int = 5):
        self.loss_multiplier = loss_multiplier
        self.holding_cost = holding_cost
        self.sl_reward = sl_reward
        self.target_reward = target_reward
        self.overtrade_penalty = overtrade_penalty
        self.max_trades_per_day = max_trades_per_day

    def compute_reward(self,
                       action: int,
                       pnl: float,
                       position_duration: int,
                       consecutive_losses: int,
                       volatility: float,
                       stop_loss_hit: bool,
                       target_hit: bool,
                       capital: float = 100000,
                       trades_today: int = 0) -> float:
        """
        Compute shaped reward from multiple components.

        Parameters
        ----------
        action : int
            Action taken (0-4).
        pnl : float
            P&L from this step (normalized by capital or raw trade P&L).
        position_duration : int
            How long current position has been held.
        consecutive_losses : int
            Number of consecutive losing trades.
        volatility : float
            Recent price volatility (for Sharpe-like scaling).
        stop_loss_hit : bool
            Whether stop loss was triggered.
        target_hit : bool
            Whether profit target was hit.
        capital : float
            Account capital for normalization.
        trades_today : int
            Number of trades today.

        Returns
        -------
        float : reward clipped to [-1, +1].
        """
        reward = 0.0

        # ───────────── BASE REWARD (Sharpe-like scaling) ─────────────
        # Scale P&L by volatility — rewards consistent returns over lucky trades
        r_base = pnl / (volatility + 1e-6) * 0.1
        reward += r_base

        # ───────────── ASYMMETRIC LOSS PENALTY ─────────────
        # Losses hurt 2x more than gains feel good (loss aversion)
        if pnl < 0:
            r_loss = self.loss_multiplier * pnl / max(capital, 1)
        else:
            r_loss = pnl / max(capital, 1)
        reward += r_loss * 10  # Scale up for impact

        # ───────────── HOLDING COST ─────────────
        # Encourages decisive action — lingering in a position costs
        if position_duration > 0:
            r_hold = -self.holding_cost * position_duration
            reward += r_hold

        # ───────────── STOP LOSS DISCIPLINE ─────────────
        # Taking a small loss at stop loss = GOOD behavior (discipline)
        if stop_loss_hit:
            reward += self.sl_reward

        # ───────────── TARGET ACHIEVEMENT ─────────────
        if target_hit:
            reward += self.target_reward

        # ───────────── OVERTRADING PENALTY ─────────────
        # Too many trades = too many transaction costs
        if trades_today > self.max_trades_per_day:
            extra = trades_today - self.max_trades_per_day
            reward -= self.overtrade_penalty * extra

        # ───────────── CONSECUTIVE LOSS PENALTY ─────────────
        # Drawdown control: penalty grows quadratically with losses
        if consecutive_losses > 0:
            r_consec = -0.01 * consecutive_losses ** 2
            reward += r_consec

        # ───────────── INACTION PENALTY ─────────────
        # Small penalty for doing nothing (encourages engagement)
        if action == 0 and position_duration == 0:
            reward -= 0.0005

        # Clip to [-1, +1]
        return float(np.clip(reward, -1.0, 1.0))
