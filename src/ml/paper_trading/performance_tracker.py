"""
Performance Tracker — Portfolio analytics and metrics dashboard.
================================================================
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks and computes portfolio performance metrics from trade history.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.equity_curve: List[float] = [initial_capital]

    def update(self, current_value: float):
        """Add current portfolio value to equity curve."""
        self.equity_curve.append(current_value)

    def compute_metrics(self) -> Dict:
        """
        Compute comprehensive performance metrics.
        """
        if len(self.equity_curve) < 2:
            return {'insufficient_data': True}

        eq = np.array(self.equity_curve)
        returns = np.diff(eq) / eq[:-1]

        # Total return
        total_return = (eq[-1] / eq[0] - 1) * 100

        # Sharpe Ratio (annualized, assume 252 trading days)
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0

        # Sortino ratio (downside deviation only)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / downside.std()
        else:
            sortino = 0

        # Max Drawdown
        peak = np.maximum.accumulate(eq)
        drawdowns = (eq - peak) / peak
        max_dd = drawdowns.min() * 100

        # Calmar ratio (return / max drawdown)
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0

        # Win days / loss days
        win_days = (returns > 0).sum()
        loss_days = (returns < 0).sum()

        # Best/worst day
        best_day = returns.max() * 100 if len(returns) > 0 else 0
        worst_day = returns.min() * 100 if len(returns) > 0 else 0

        return {
            'total_return_pct': round(total_return, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'calmar_ratio': round(calmar, 3),
            'win_days': int(win_days),
            'loss_days': int(loss_days),
            'win_pct': round(win_days / max(win_days + loss_days, 1) * 100, 1),
            'best_day_pct': round(best_day, 2),
            'worst_day_pct': round(worst_day, 2),
            'current_value': round(eq[-1], 2),
            'peak_value': round(peak.max(), 2),
            'n_trading_days': len(returns),
            'avg_daily_return_pct': round(returns.mean() * 100, 3),
            'volatility_annual_pct': round(returns.std() * np.sqrt(252) * 100, 2),
        }

    def get_equity_curve(self) -> List[float]:
        """Return equity curve for plotting."""
        return self.equity_curve.copy()

    def reset(self):
        """Reset tracker."""
        self.equity_curve = [self.initial_capital]
