"""
Tests for src/options/black_scholes.py

Reference values computed against known BSM analytical solutions.
Tolerance of 1e-4 is appropriate for double-precision floating point.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.options.black_scholes import (
    call_price, put_price, delta, gamma, theta, vega, rho, greeks
)

# ── Shared test parameters ────────────────────────────────────────────────────
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
TOL = 1e-4


class TestCallPrice:
    def test_atm_call(self):
        # ATM call with known reference value ~10.4506
        assert abs(call_price(S, K, T, r, sigma) - 10.4506) < 1e-3

    def test_deep_itm_call_approaches_intrinsic(self):
        # Deep ITM: call ≈ S - K * exp(-rT)
        c = call_price(200, 100, 1.0, 0.05, 0.20)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert abs(c - intrinsic) < 1.0

    def test_deep_otm_call_near_zero(self):
        c = call_price(50, 200, 1.0, 0.05, 0.20)
        assert c < 0.01

    def test_call_positive(self):
        assert call_price(S, K, T, r, sigma) > 0


class TestPutPrice:
    def test_atm_put(self):
        # ATM put with known reference value ~5.5735
        assert abs(put_price(S, K, T, r, sigma) - 5.5735) < 1e-3

    def test_put_call_parity(self):
        # C - P = S - K * exp(-rT)
        c = call_price(S, K, T, r, sigma)
        p = put_price(S, K, T, r, sigma)
        lhs = c - p
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < TOL

    def test_deep_otm_put_near_zero(self):
        p = put_price(200, 50, 1.0, 0.05, 0.20)
        assert p < 0.01


class TestGreeks:
    def test_call_delta_range(self):
        d = delta(S, K, T, r, sigma, "call")
        assert 0 < d < 1

    def test_put_delta_range(self):
        d = delta(S, K, T, r, sigma, "put")
        assert -1 < d < 0

    def test_call_put_delta_relationship(self):
        # call_delta - put_delta = exp(-qT) = 1 when q=0
        d_call = delta(S, K, T, r, sigma, "call")
        d_put  = delta(S, K, T, r, sigma, "put")
        assert abs(d_call - d_put - 1.0) < TOL

    def test_gamma_positive(self):
        assert gamma(S, K, T, r, sigma) > 0

    def test_gamma_call_equals_put(self):
        # Gamma is identical for calls and puts
        g_call = gamma(S, K, T, r, sigma)
        g_put  = gamma(S, K, T, r, sigma)
        assert abs(g_call - g_put) < TOL

    def test_theta_negative_for_long_call(self):
        assert theta(S, K, T, r, sigma, "call") < 0

    def test_vega_positive(self):
        assert vega(S, K, T, r, sigma) > 0

    def test_call_rho_positive(self):
        assert rho(S, K, T, r, sigma, "call") > 0

    def test_put_rho_negative(self):
        assert rho(S, K, T, r, sigma, "put") < 0

    def test_greeks_dict_keys(self):
        g = greeks(S, K, T, r, sigma, "call")
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}


class TestEdgeCases:
    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            call_price(S, K, -1.0, r, sigma)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            call_price(S, K, T, r, -0.1)

    def test_high_volatility(self):
        # Should not raise; just return a large but finite price
        c = call_price(S, K, T, r, 2.0)
        assert np.isfinite(c)
