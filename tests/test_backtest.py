import pytest
import pandas as pd
import numpy as np
from src.backtest.metrics import BacktestMetrics

@pytest.fixture
def dummy_returns():
    """Fixture providing a known sequence of daily returns."""
    # 5 days of positive 1% returns followed by 5 days of negative 1% returns
    ret = [0.01]*5 + [-0.01]*5
    dates = pd.date_range(start="2023-01-01", periods=10, freq="B")
    return pd.Series(ret, index=dates)

def test_calculate_metrics_empty():
    """Test metrics calculation with empty series."""
    empty_series = pd.Series([], dtype=float)
    metrics = BacktestMetrics.calculate_metrics(empty_series)
    assert metrics["total_return"] == 0.0
    assert metrics["annualized_return"] == 0.0

def test_calculate_metrics_values(dummy_returns):
    """Test metrics calculation correctness."""
    metrics = BacktestMetrics.calculate_metrics(dummy_returns, risk_free_rate=0.0)
    
    # 5 * 1% gains and 5 * 1% losses ~ total return should be slightly less than 0 due to compounding
    # (1.01^5) * (0.99^5) - 1 = (0.9999^5) - 1 ≈ -0.0005
    assert metrics["total_return"] < 0
    assert -0.01 < metrics["total_return"] < 0.0
    
    # Win rate should be 50%
    assert metrics["win_rate"] == 0.5
    
    # Max drawdown should be roughly 5% (from peak after first 5 days)
    assert metrics["max_drawdown"] < -0.04
    assert metrics["max_drawdown"] > -0.06

def test_get_drawdown_curve(dummy_returns):
    """Test drawdown curve generation."""
    dd_curve = BacktestMetrics.get_drawdown_curve(dummy_returns)
    assert len(dd_curve) == 10
    # First 5 days should have 0 drawdown as it's making new highs
    assert all(dd_curve.iloc[:5] == 0.0)
    # Last 5 days should be negative
    assert all(dd_curve.iloc[5:] < 0.0)
