"""
Markowitz Mean-Variance Portfolio Optimizer.

Fetches historical price data for a given list of tickers via yfinance,
computes the annualised return vector and covariance matrix, then uses
CVXPY to solve the maximum-Sharpe-ratio portfolio via the Sharpe ratio
reformulation (Cornuejols & Tutuncu, 2006):

    Maximise  (mu^T w - r_f) / sqrt(w^T Sigma w)
    ≡ Minimise  y^T Sigma y
      subject to  (mu - r_f)^T y = 1,  y >= 0,  w = y / sum(y)

Also exposes the full efficient frontier by parametric sweep over
target return levels.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional

from src.utils.data_loader import fetch_nifty50_prices, get_returns
from src.utils.visualizer import plot_efficient_frontier


NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "AXISBANK.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS",
]

def compute_stats(prices: pd.DataFrame, trading_days: int = 252) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute annualised mean returns and covariance matrix from daily prices."""
    daily_returns = get_returns(prices)
    mu = daily_returns.mean().values * trading_days
    Sigma = daily_returns.cov().values * trading_days
    return mu, Sigma, list(prices.columns)


def max_sharpe_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    tickers: list[str],
    risk_free_rate: float = 0.065,
) -> dict:
    """
    Solve for the maximum Sharpe ratio portfolio using the
    Cornuejols-Tutuncu reformulation.

    Parameters
    ----------
    mu             : annualised expected return vector (N,)
    Sigma          : annualised covariance matrix (N, N)
    tickers        : asset names corresponding to mu/Sigma columns
    risk_free_rate : annualised risk-free rate (default: RBI repo rate proxy)

    Returns
    -------
    dict with keys:
        weights      : {ticker: weight} optimal allocation
        expected_return : portfolio annualised return
        volatility   : portfolio annualised std dev
        sharpe_ratio : (return - rf) / volatility
    """
    n = len(mu)
    excess_mu = mu - risk_free_rate

    y = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.quad_form(y, Sigma))
    constraints = [excess_mu @ y == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY solver failed with status: {prob.status}")

    y_val = y.value
    weights = y_val / y_val.sum()

    port_return = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ Sigma @ weights))
    sharpe = (port_return - risk_free_rate) / port_vol

    return {
        "weights": dict(zip(tickers, np.round(weights, 6))),
        "expected_return": round(port_return, 6),
        "volatility": round(port_vol, 6),
        "sharpe_ratio": round(sharpe, 6),
    }


def efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by solving minimum-variance portfolios
    across a grid of target return levels.

    Returns
    -------
    DataFrame with columns: target_return, volatility
    """
    n = len(mu)
    min_ret = mu.min()
    max_ret = mu.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vols = []
    for target in target_returns:
        w = cp.Variable(n, nonneg=True)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1, mu @ w >= target]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            frontier_vols.append(float(np.sqrt(w.value @ Sigma @ w.value)))
        else:
            frontier_vols.append(np.nan)

    return pd.DataFrame({"target_return": target_returns, "volatility": frontier_vols}).dropna()




def run(
    tickers: list[str] = NIFTY50_TICKERS,
    start: str = "2021-01-01",
    end: str = "2024-01-01",
    risk_free_rate: float = 0.065,
    cache_path: Optional[str] = None,
    plot: bool = True,
    frontier_save_path: Optional[str] = None,
) -> dict:
    """End-to-end pipeline: fetch data → compute stats → optimise → plot."""
    prices = fetch_nifty50_prices(tickers, start, end, cache_path)
    mu, Sigma, names = compute_stats(prices)
    result = max_sharpe_portfolio(mu, Sigma, names, risk_free_rate)

    if plot:
        frontier_df = efficient_frontier(mu, Sigma)
        plot_efficient_frontier(frontier_df, result, risk_free_rate, frontier_save_path)

    return result
