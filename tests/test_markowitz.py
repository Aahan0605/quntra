"""
Tests for src/portfolio/markowitz.py

Uses synthetically generated price data to avoid network dependency
in CI. The synthetic data is constructed to have known statistical
properties, allowing deterministic assertions on the optimizer output.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.portfolio.markowitz import compute_stats, max_sharpe_portfolio, efficient_frontier

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
           "WIPRO.NS", "AXISBANK.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS"]


def make_synthetic_prices(n_assets: int = 10, n_days: int = 756, seed: int = 0) -> pd.DataFrame:
    """
    Generate synthetic log-normal price paths with controlled drift and vol.
    n_days = 756 ≈ 3 years of trading days.
    """
    rng = np.random.default_rng(seed)
    mu_daily = rng.uniform(0.0003, 0.0012, n_assets)   # ~7–30% annualised
    sigma_daily = rng.uniform(0.010, 0.025, n_assets)  # ~16–40% annualised

    log_returns = rng.normal(mu_daily, sigma_daily, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))

    dates = pd.bdate_range(end="2024-01-01", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=TICKERS[:n_assets])


@pytest.fixture(scope="module")
def synthetic_prices():
    return make_synthetic_prices()


@pytest.fixture(scope="module")
def stats(synthetic_prices):
    return compute_stats(synthetic_prices)


class TestComputeStats:
    def test_mu_shape(self, stats):
        mu, Sigma, names = stats
        assert mu.shape == (10,)

    def test_sigma_shape(self, stats):
        mu, Sigma, names = stats
        assert Sigma.shape == (10, 10)

    def test_sigma_positive_semidefinite(self, stats):
        _, Sigma, _ = stats
        eigenvalues = np.linalg.eigvalsh(Sigma)
        assert np.all(eigenvalues >= -1e-8)

    def test_sigma_symmetric(self, stats):
        _, Sigma, _ = stats
        assert np.allclose(Sigma, Sigma.T)

    def test_names_match_tickers(self, stats):
        _, _, names = stats
        assert names == TICKERS


class TestMaxSharpePortfolio:
    def test_weights_sum_to_one(self, stats):
        mu, Sigma, names = stats
        result = max_sharpe_portfolio(mu, Sigma, names)
        assert abs(sum(result["weights"].values()) - 1.0) < 1e-4

    def test_weights_non_negative(self, stats):
        mu, Sigma, names = stats
        result = max_sharpe_portfolio(mu, Sigma, names)
        assert all(w >= -1e-6 for w in result["weights"].values())

    def test_result_keys(self, stats):
        mu, Sigma, names = stats
        result = max_sharpe_portfolio(mu, Sigma, names)
        assert {"weights", "expected_return", "volatility", "sharpe_ratio"} == set(result.keys())

    def test_sharpe_positive(self, stats):
        mu, Sigma, names = stats
        result = max_sharpe_portfolio(mu, Sigma, names)
        assert result["sharpe_ratio"] > 0

    def test_volatility_positive(self, stats):
        mu, Sigma, names = stats
        result = max_sharpe_portfolio(mu, Sigma, names)
        assert result["volatility"] > 0

    def test_weights_ticker_keys(self, stats):
        mu, Sigma, names = stats
        result = max_sharpe_portfolio(mu, Sigma, names)
        assert set(result["weights"].keys()) == set(TICKERS)


class TestEfficientFrontier:
    def test_frontier_shape(self, stats):
        mu, Sigma, _ = stats
        frontier = efficient_frontier(mu, Sigma, n_points=20)
        assert len(frontier) > 0
        assert "target_return" in frontier.columns
        assert "volatility" in frontier.columns

    def test_frontier_volatility_positive(self, stats):
        mu, Sigma, _ = stats
        frontier = efficient_frontier(mu, Sigma, n_points=20)
        assert (frontier["volatility"] > 0).all()

    def test_frontier_monotone_vol(self, stats):
        """
        Volatility should be roughly U-shaped (non-decreasing after minimum).
        Check that the minimum is not at the extreme ends.
        """
        mu, Sigma, _ = stats
        frontier = efficient_frontier(mu, Sigma, n_points=30)
        min_idx = frontier["volatility"].idxmin()
        assert 0 < min_idx < len(frontier) - 1
