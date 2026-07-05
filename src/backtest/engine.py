"""
QuNtra backtest engine.

v2 (Phase 0 fix): realistic friction and rebalancing.
  * Costs load from config/costs.env via src.utils.costs.CostModel —
    no more hardcoded "10% average turnover" approximation.
  * Weights drift with returns; a Rebalancer (weekly + 3% drift
    threshold + 20% turnover cap) decides actual trades, and costs are
    charged on actual traded notional.
  * Prices can be injected (offline/cached runs) instead of fetched.
"""

import numpy as np
import pandas as pd

from src.utils.data_loader import fetch_nifty50_prices, get_returns
from src.utils.costs import CostModel
from src.portfolio.rebalancer import Rebalancer
from src.backtest.metrics import BacktestMetrics
from src.backtest.attribution import PerformanceAttribution


class BacktestEngine:
    """Historical backtester with drift-aware rebalancing and real costs."""

    def __init__(
        self,
        tickers: list[str],
        weights_dict: dict,
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        transaction_cost_bps: float | None = None,   # legacy override
        slippage_bps: float | None = None,           # legacy override
        cost_model: CostModel | None = None,
        rebalancer: Rebalancer | None = None,
    ):
        self.tickers = tickers
        self.weights = {k: v for k, v in weights_dict.items() if v > 0}
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.cost_model = cost_model or CostModel.from_config()
        # One-way friction per unit traded notional (fees + slippage)
        if transaction_cost_bps is not None:
            self._one_way_fee = transaction_cost_bps / 10_000.0
        else:
            self._one_way_fee = self.cost_model.one_way_cost_rate(delivery=True)
        if slippage_bps is not None:
            self._one_way_slip = slippage_bps / 10_000.0
        else:
            self._one_way_slip = self.cost_model.slippage_rate()
        self._one_way_friction = self._one_way_fee + self._one_way_slip

        self.rebalancer = rebalancer  # None -> constructed in run()

        # Results
        self.metrics = None
        self.equity_curve = None
        self.drawdown_curve = None
        self.attribution = None
        self.annual_turnover = None

    # ------------------------------------------------------------------ #

    def run(
        self,
        rebalance_freq: str = "weekly",
        prices: pd.DataFrame | None = None,
    ) -> dict:
        """
        Run the backtest. `prices` may be injected (cached/offline data);
        otherwise fetched via the data loader.
        """
        if prices is None:
            prices = fetch_nifty50_prices(self.tickers, self.start_date, self.end_date)
        if prices.empty:
            raise ValueError("No price data retrieved for backtest.")
        prices = prices[[c for c in prices.columns if c in self.weights or c in self.tickers]]

        returns = get_returns(prices)
        cols = list(returns.columns)

        target = np.array([self.weights.get(t, 0.0) for t in cols], dtype=float)
        if target.sum() <= 0:
            raise ValueError("Target weights sum to zero for available tickers.")
        target = target / target.sum()
        target_map = dict(zip(cols, target))

        rebalancer = self.rebalancer or Rebalancer(frequency=rebalance_freq)
        rebalancer._last_rebalance = None

        current = target.copy()
        port_returns = pd.Series(0.0, index=returns.index)
        total_turnover = 0.0

        # Day 0: enter the market — pay one-way friction on 100% notional
        port_returns.iloc[0] = float(returns.iloc[0] @ current) - self._one_way_friction

        for i in range(1, len(returns)):
            row = returns.iloc[i]
            daily_ret = float(row @ current)

            # Drift weights with today's returns
            growth = current * (1.0 + row.values)
            denom = growth.sum()
            if denom > 0:
                current = growth / denom

            ts = returns.index[i]
            decision = rebalancer.compute_trades(
                dict(zip(cols, current)), target_map,
                ts.date() if hasattr(ts, "date") else ts,
            )
            if decision.should_rebalance:
                traded_notional = 2.0 * decision.one_way_turnover  # buys + sells
                daily_ret -= traded_notional * self._one_way_friction
                total_turnover += decision.one_way_turnover
                for j, t in enumerate(cols):
                    current[j] += decision.trades.get(t, 0.0)
                current = np.clip(current, 0, None)
                s = current.sum()
                if s > 0:
                    current = current / s

            port_returns.iloc[i] = daily_ret

        years = len(returns) / 252.0
        self.annual_turnover = total_turnover / years if years > 0 else 0.0

        self.metrics = BacktestMetrics.calculate_metrics(port_returns)
        self.metrics["annual_turnover"] = float(self.annual_turnover)
        self.metrics["one_way_friction_rate"] = float(self._one_way_friction)
        self.drawdown_curve = BacktestMetrics.get_drawdown_curve(port_returns)
        self.equity_curve = self.initial_capital * (1 + port_returns).cumprod()
        self.attribution = PerformanceAttribution.calculate_attribution(self.weights, returns)

        return {
            "metrics": self.metrics,
            "equity_curve": {k.strftime("%Y-%m-%d"): v for k, v in self.equity_curve.items()},
            "drawdown_curve": {k.strftime("%Y-%m-%d"): round(v, 4) for k, v in self.drawdown_curve.items()},
            "attribution": self.attribution,
        }
