"""
Quantra Trading Environment — Custom Gymnasium env for stock trading.
======================================================================

State space:
  - All technical + fundamental + sentiment features (normalized)
  - Current position: {0: flat, 1: long, 2: short}
  - Current P&L of open position (normalized)
  - Time remaining in session (for intraday)
  - Consecutive losses count

Action space: Discrete(5)
  0 = HOLD, 1 = BUY, 2 = SELL SHORT, 3 = CLOSE LONG, 4 = CLOSE SHORT

Episode: one trading day (intraday) or 20 days (swing)
"""

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from ..rl.reward import RewardShaper

logger = logging.getLogger(__name__)


class QuantraTradingEnv(gym.Env if HAS_GYM else object):
    """
    Custom OpenAI Gym environment for NSE/global stock trading.
    Realistic simulation with transaction costs and slippage.
    """

    metadata = {'render_modes': ['human']}
    ACTION_NAMES = {0: 'HOLD', 1: 'BUY', 2: 'SHORT', 3: 'CLOSE_LONG', 4: 'CLOSE_SHORT'}

    def __init__(self, df: np.ndarray,
                 initial_capital: float = 100000,
                 mode: str = 'intraday',
                 transaction_cost: float = 0.001,
                 allow_short: bool = True,
                 max_position_pct: float = 0.10):
        """
        Parameters
        ----------
        df : np.ndarray or pd.DataFrame
            Feature matrix with price data. Must include a 'close' column
            or close prices in the first column.
        initial_capital : float
            Starting capital in INR.
        mode : str
            'intraday' or 'swing'.
        transaction_cost : float
            Cost per trade as fraction (0.001 = 0.1%).
        allow_short : bool
            Whether short selling is allowed.
        max_position_pct : float
            Maximum position as fraction of capital.
        """
        super().__init__()

        import pandas as pd

        if isinstance(df, pd.DataFrame):
            if 'close' in df.columns:
                self._close_prices = df['close'].values.astype(float)
            else:
                self._close_prices = df.iloc[:, 0].values.astype(float)
            self._features = df.select_dtypes(include=[np.number]).values.astype(float)
        else:
            self._features = df.astype(float)
            self._close_prices = df[:, 0]  # Assume close is first column

        self.n_steps = len(self._features)
        self.n_features = self._features.shape[1]
        self.initial_capital = initial_capital
        self.mode = mode
        self.transaction_cost = transaction_cost
        self.allow_short = allow_short
        self.max_position_pct = max_position_pct

        # State augmentation: +5 for position, pnl, time, consecutive losses, capital
        self.obs_size = self.n_features + 5

        # Gym spaces
        if HAS_GYM:
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_size,), dtype=np.float32
            )

        # Reward shaper
        self.reward_shaper = RewardShaper()

        # Episode state (set in reset)
        self._reset_state()

    def _reset_state(self):
        """Reset all episode state variables."""
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0  # 0=flat, 1=long, 2=short
        self.position_price = 0.0
        self.position_size = 0
        self.pnl = 0.0
        self.total_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.position_duration = 0
        self.episode_trades = []

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Reset to start of episode.
        Randomize starting date within training data.
        """
        if seed is not None:
            np.random.seed(seed)

        self._reset_state()

        # Determine episode length
        if self.mode == 'intraday':
            self.episode_len = min(78, self.n_steps)  # ~6.5 hours at 5m
        else:
            self.episode_len = min(20, self.n_steps)  # 20 trading days

        # Random start within valid range
        max_start = max(0, self.n_steps - self.episode_len - 1)
        if max_start > 0:
            self.current_step = np.random.randint(0, max_start)
        else:
            self.current_step = 0

        self.start_step = self.current_step
        self.end_step = min(self.current_step + self.episode_len, self.n_steps - 1)

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action, advance one timestep.
        Returns (observation, reward, terminated, truncated, info).
        """
        # Validate action
        action = self._validate_action(action)

        current_price = self._close_prices[self.current_step]
        next_step = min(self.current_step + 1, self.n_steps - 1)
        next_price = self._close_prices[next_step]

        # Execute trade at NEXT candle open (realistic — no look-ahead)
        trade_pnl = 0.0
        stop_loss_hit = False
        target_hit = False

        if action == 1 and self.position == 0:
            # BUY — enter long
            cost = next_price * self.transaction_cost
            self.position = 1
            self.position_price = next_price + cost
            position_value = self.capital * self.max_position_pct
            self.position_size = int(position_value / next_price)
            self.capital -= self.position_size * self.position_price
            self.position_duration = 0
            self.trades_today += 1

        elif action == 2 and self.position == 0 and self.allow_short:
            # SHORT — enter short
            cost = next_price * self.transaction_cost
            self.position = 2
            self.position_price = next_price - cost
            position_value = self.capital * self.max_position_pct
            self.position_size = int(position_value / next_price)
            self.capital += self.position_size * self.position_price
            self.position_duration = 0
            self.trades_today += 1

        elif action == 3 and self.position == 1:
            # CLOSE LONG
            cost = next_price * self.transaction_cost
            exit_price = next_price - cost
            trade_pnl = (exit_price - self.position_price) * self.position_size
            self.capital += self.position_size * exit_price
            self.total_pnl += trade_pnl
            self.episode_trades.append(trade_pnl)

            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            self.position = 0
            self.position_price = 0
            self.position_size = 0
            self.trades_today += 1

        elif action == 4 and self.position == 2:
            # CLOSE SHORT
            cost = next_price * self.transaction_cost
            exit_price = next_price + cost
            trade_pnl = (self.position_price - exit_price) * self.position_size
            self.capital -= self.position_size * exit_price
            self.capital += self.position_size * self.position_price + trade_pnl
            self.total_pnl += trade_pnl
            self.episode_trades.append(trade_pnl)

            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            self.position = 0
            self.position_price = 0
            self.position_size = 0
            self.trades_today += 1

        # Update position P&L
        if self.position == 1:
            self.pnl = (next_price - self.position_price) * self.position_size
            self.position_duration += 1
        elif self.position == 2:
            self.pnl = (self.position_price - next_price) * self.position_size
            self.position_duration += 1
        else:
            self.pnl = 0.0
            self.position_duration = 0

        # Compute ATR-like volatility for reward shaping
        lookback = min(14, self.current_step - self.start_step + 1)
        if lookback > 1:
            recent_prices = self._close_prices[
                max(0, self.current_step - lookback):self.current_step + 1
            ]
            volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
        else:
            volatility = 0.01

        # Compute reward
        reward = self.reward_shaper.compute_reward(
            action=action,
            pnl=trade_pnl if trade_pnl != 0 else self.pnl / max(self.initial_capital, 1),
            position_duration=self.position_duration,
            consecutive_losses=self.consecutive_losses,
            volatility=volatility,
            stop_loss_hit=stop_loss_hit,
            target_hit=target_hit,
            capital=self.initial_capital,
            trades_today=self.trades_today,
        )

        # Advance step
        self.current_step = next_step

        # Terminal conditions
        terminated = False
        truncated = False

        # End of episode
        if self.current_step >= self.end_step:
            truncated = True
            # Force close any open position
            if self.position != 0:
                close_price = self._close_prices[self.current_step]
                if self.position == 1:
                    trade_pnl = (close_price - self.position_price) * self.position_size
                else:
                    trade_pnl = (self.position_price - close_price) * self.position_size
                self.total_pnl += trade_pnl
                self.position = 0

        # Max drawdown circuit breaker (> 10% loss)
        total_value = self.capital + self.pnl
        if total_value < self.initial_capital * 0.90:
            terminated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _validate_action(self, action: int) -> int:
        """Validate and potentially override invalid actions."""
        if action == 1 and self.position != 0:
            return 0  # Can't buy if already in position
        if action == 2 and (self.position != 0 or not self.allow_short):
            return 0  # Can't short if in position or shorts disabled
        if action == 3 and self.position != 1:
            return 0  # Can't close long if not long
        if action == 4 and self.position != 2:
            return 0  # Can't close short if not short
        return action

    def _get_observation(self) -> np.ndarray:
        """
        Return flattened state vector:
        [feature_vector, position_encoding, pnl_normalized,
         time_remaining_normalized, consecutive_losses_normalized, capital_pct]
        """
        features = self._features[self.current_step]

        # Augmented state
        position_enc = float(self.position)
        pnl_norm = self.pnl / max(self.initial_capital, 1)
        time_remaining = (self.end_step - self.current_step) / max(self.episode_len, 1)
        consec_loss_norm = min(self.consecutive_losses / 5.0, 1.0)
        capital_pct = self.capital / self.initial_capital

        obs = np.concatenate([
            features,
            [position_enc, pnl_norm, time_remaining, consec_loss_norm, capital_pct]
        ]).astype(np.float32)

        # Replace NaN/inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _get_info(self) -> dict:
        """Return episode info dict."""
        return {
            'step': self.current_step,
            'capital': self.capital,
            'position': self.position,
            'pnl': self.pnl,
            'total_pnl': self.total_pnl,
            'trades_today': self.trades_today,
            'consecutive_losses': self.consecutive_losses,
        }

    def render(self, mode='human'):
        """Print current step, price, position, P&L to console."""
        price = self._close_prices[self.current_step]
        pos_name = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}[self.position]
        print(
            f"Step {self.current_step} | Price: {price:.2f} | "
            f"Position: {pos_name} | P&L: {self.pnl:.2f} | "
            f"Total: {self.total_pnl:.2f} | Capital: {self.capital:.2f}"
        )
