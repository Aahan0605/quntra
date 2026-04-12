import numpy as np
import pandas as pd

class BacktestMetrics:
    """Calculates financial metrics for a portfolio Return series."""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.07, trading_days: int = 252) -> dict:
        """
        Calculate key performance indicators from a series of daily returns.
        """
        if returns.empty or len(returns) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
            
        # Total Return
        cumulative_returns = (1 + returns).cumprod()
        total_ret = cumulative_returns.iloc[-1] - 1
        
        # Annualized Return and Volatility
        years = len(returns) / trading_days
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        ann_vol = returns.std() * np.sqrt(trading_days)
        
        # Sharpe Ratio
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(trading_days)
        sortino = (ann_ret - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Max Drawdown
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win Rate
        win_rate = len(returns[returns > 0]) / len(returns)
        
        return {
            "total_return": float(total_ret),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate)
        }
        
    @staticmethod
    def get_drawdown_curve(returns: pd.Series) -> pd.Series:
        """Calculate the drawdown series over time."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        return (cumulative_returns - running_max) / running_max
