import pandas as pd
import numpy as np

from src.utils.data_loader import fetch_nifty50_prices, get_returns
from src.backtest.metrics import BacktestMetrics
from src.backtest.attribution import PerformanceAttribution

class BacktestEngine:
    """Historical backtester evaluating constant-weight strategies."""
    
    def __init__(self, tickers: list[str], weights_dict: dict, start_date: str, end_date: str, 
                 initial_capital: float = 1000000.0, transaction_cost_bps: float = 10.0, slippage_bps: float = 5.0):
        self.tickers = tickers
        self.weights = {k: v for k, v in weights_dict.items() if v > 0}
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Costs
        self.t_cost_rate = transaction_cost_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        self.total_cost_rate = self.t_cost_rate + self.slippage_rate
        
        # Results
        self.metrics = None
        self.equity_curve = None
        self.drawdown_curve = None
        self.attribution = None
        
    def run(self, rebalance_freq: str = 'monthly') -> dict:
        """
        Run the backtest loop.
        Frequency strings: 'daily', 'weekly', 'monthly', 'quarterly'
        """
        # Fetch adjusted prices
        prices = fetch_nifty50_prices(self.tickers, self.start_date, self.end_date)
        if prices.empty:
            raise ValueError("No price data retrieved for backtest.")
            
        returns = get_returns(prices)
        
        # Identify rebalance days
        if rebalance_freq == 'monthly':
            rebalance_dates = returns.groupby([returns.index.year, returns.index.month]).apply(lambda x: x.index[0])
        elif rebalance_freq == 'weekly':
            rebalance_dates = returns.groupby([returns.index.year, returns.index.isocalendar().week]).apply(lambda x: x.index[0])
        elif rebalance_freq == 'quarterly':
            rebalance_dates = returns.groupby([returns.index.year, returns.index.quarter]).apply(lambda x: x.index[0])
        else:
            # daily
            rebalance_dates = returns.index
            
        # Drop the hierarchical index created by groupby and sort
        rebalance_dates = sorted(list(rebalance_dates))
            
        # Simulate portfolio loop
        portfolio_val = self.initial_capital
        equity_series = pd.Series(index=returns.index, dtype=float)
        
        # Initialize target allocations
        current_allocations = {t: 0.0 for t in self.tickers}
        weights_array = np.array([self.weights.get(t, 0.0) for t in prices.columns])
        
        # Simplified vectorised backtest assuming we hold the weights matrix constant, 
        # and pay friction on rebalance days.
        port_returns = pd.Series(0.0, index=returns.index)
        
        # First day setup: we enter the market and pay total cost
        port_returns.iloc[0] = (returns.iloc[0] @ weights_array) - self.total_cost_rate
        
        for i in range(1, len(returns)):
            date = returns.index[i]
            # Standard return before rebalancing
            daily_ret = returns.iloc[i] @ weights_array
            
            if date in rebalance_dates:
                # Pay turnover cost - assuming turnover is roughly proportion of portfolio (simplified approximation)
                daily_ret -= self.total_cost_rate * 0.10 # Assuming 10% average turnover turnover 
                
            port_returns.iloc[i] = daily_ret
            
        # Subtract generic expense ratio (reduction in return continuously)
        # Assuming 0.5% annualized expense ratio / 252
        expense_ratio_daily = 0.005 / 252.0
        port_returns -= expense_ratio_daily
            
        self.metrics = BacktestMetrics.calculate_metrics(port_returns)
        self.drawdown_curve = BacktestMetrics.get_drawdown_curve(port_returns)
        self.equity_curve = self.initial_capital * (1 + port_returns).cumprod()
        self.attribution = PerformanceAttribution.calculate_attribution(self.weights, returns)
        
        return {
            "metrics": self.metrics,
            "equity_curve": {k.strftime("%Y-%m-%d"): v for k, v in self.equity_curve.items()},
            "drawdown_curve": {k.strftime("%Y-%m-%d"): round(v, 4) for k, v in self.drawdown_curve.items()},
            "attribution": self.attribution
        }
