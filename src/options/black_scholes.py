"""
Black-Scholes options pricing model.

Implements closed-form pricing for European call and put options,
along with the full set of first-order Greeks (Delta, Gamma, Theta,
Vega, Rho) under the standard GBM assumption.

Parameters follow market convention:
    S     : current underlying price
    K     : strike price
    T     : time to expiry in years
    r     : continuously compounded risk-free rate (annualised)
    sigma : implied / historical volatility (annualised)
    q     : continuous dividend yield (default 0)
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Compute d1 and d2 intermediates. Raises ValueError for non-positive T."""
    if T <= 0:
        raise ValueError("Time to expiry T must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European call price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European put price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ── Greeks ────────────────────────────────────────────────────────────────────

def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call", q: float = 0.0) -> float:
    """
    Delta — sensitivity of option price to underlying price.
    Call delta ∈ (0, 1); put delta ∈ (-1, 0).
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Gamma — rate of change of delta with respect to underlying price.
    Identical for calls and puts under BSM.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call", q: float = 0.0) -> float:
    """
    Theta — time decay, expressed as change per calendar day.
    Typically negative for long positions.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (common - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
    return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Vega — sensitivity to a 1-point (100%) move in volatility.
    Divided by 100 to express per 1% vol move (market convention).
    Identical for calls and puts.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call", q: float = 0.0) -> float:
    """
    Rho — sensitivity to a 1-point (100%) move in the risk-free rate.
    Divided by 100 to express per 1% rate move (market convention).
    """
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           option_type: str = "call", q: float = 0.0) -> dict:
    """
    Returns all five Greeks as a dictionary for a given option.

    Returns
    -------
    dict with keys: delta, gamma, theta, vega, rho
    """
    return {
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "vega":  vega(S, K, T, r, sigma, q),
        "rho":   rho(S, K, T, r, sigma, option_type, q),
    }
