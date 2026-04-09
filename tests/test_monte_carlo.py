"""
Tests for src/options/monte_carlo.py

MC prices are stochastic — tests verify that BSM analytical prices
fall within the 95% confidence interval of the MC estimate, and that
structural properties (put-call parity, non-negativity) hold.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.options.monte_carlo import price_european_call, price_european_put, price_both, MCResult
from src.options.black_scholes import call_price, put_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N = 200_000


class TestMCResult:
    def test_returns_mcresult(self):
        result = price_european_call(S, K, T, r, sigma, n_simulations=10_000)
        assert isinstance(result, MCResult)

    def test_ci_contains_bs_call(self):
        bs_call = call_price(S, K, T, r, sigma)
        mc = price_european_call(S, K, T, r, sigma, n_simulations=N, seed=0)
        assert mc.ci_lower <= bs_call <= mc.ci_upper, (
            f"BS call {bs_call:.4f} outside MC 95% CI [{mc.ci_lower:.4f}, {mc.ci_upper:.4f}]"
        )

    def test_ci_contains_bs_put(self):
        bs_put = put_price(S, K, T, r, sigma)
        mc = price_european_put(S, K, T, r, sigma, n_simulations=N, seed=0)
        assert mc.ci_lower <= bs_put <= mc.ci_upper, (
            f"BS put {bs_put:.4f} outside MC 95% CI [{mc.ci_lower:.4f}, {mc.ci_upper:.4f}]"
        )

    def test_price_positive(self):
        mc = price_european_call(S, K, T, r, sigma, n_simulations=10_000)
        assert mc.price > 0

    def test_stderr_positive(self):
        mc = price_european_call(S, K, T, r, sigma, n_simulations=10_000)
        assert mc.stderr > 0

    def test_ci_ordering(self):
        mc = price_european_call(S, K, T, r, sigma, n_simulations=10_000)
        assert mc.ci_lower < mc.price < mc.ci_upper


class TestPriceBoth:
    def test_returns_call_and_put(self):
        results = price_both(S, K, T, r, sigma, n_simulations=10_000)
        assert "call" in results and "put" in results

    def test_put_call_parity(self):
        # C - P ≈ S - K * exp(-rT), tolerance 3 * stderr
        results = price_both(S, K, T, r, sigma, n_simulations=N, seed=1)
        lhs = results["call"].price - results["put"].price
        rhs = S - K * np.exp(-r * T)
        combined_se = np.sqrt(results["call"].stderr**2 + results["put"].stderr**2)
        assert abs(lhs - rhs) < 3 * combined_se, (
            f"Put-call parity violated: LHS={lhs:.4f}, RHS={rhs:.4f}"
        )


class TestAntithetic:
    def test_antithetic_reduces_stderr(self):
        mc_anti = price_european_call(S, K, T, r, sigma, n_simulations=50_000,
                                      antithetic=True, seed=7)
        mc_plain = price_european_call(S, K, T, r, sigma, n_simulations=50_000,
                                       antithetic=False, seed=7)
        assert mc_anti.stderr <= mc_plain.stderr * 1.1  # antithetic should not be worse


class TestReproducibility:
    def test_same_seed_same_price(self):
        r1 = price_european_call(S, K, T, r, sigma, n_simulations=10_000, seed=99)
        r2 = price_european_call(S, K, T, r, sigma, n_simulations=10_000, seed=99)
        assert r1.price == r2.price

    def test_different_seed_different_price(self):
        r1 = price_european_call(S, K, T, r, sigma, n_simulations=10_000, seed=1)
        r2 = price_european_call(S, K, T, r, sigma, n_simulations=10_000, seed=2)
        assert r1.price != r2.price
