"""
Monte Carlo simulation engine for European options pricing.

Simulates asset price paths under Geometric Brownian Motion (GBM):

    S(T) = S(0) * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
    Z ~ N(0, 1)

Uses antithetic variates by default to reduce variance without
increasing the simulation count.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MCResult:
    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_simulations: int

    def __repr__(self):
        return (
            f"MCResult(price={self.price:.4f}, stderr={self.stderr:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], n={self.n_simulations:,})"
        )


def _simulate_terminal_prices(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int,
    antithetic: bool,
    seed: Optional[int],
) -> np.ndarray:
    """
    Simulate terminal asset prices S(T) under risk-neutral GBM.
    With antithetic variates, generates n_simulations // 2 base draws
    and mirrors them, keeping total paths = n_simulations.
    """
    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma**2) * T
    diffusion_scale = sigma * np.sqrt(T)

    if antithetic:
        half = n_simulations // 2
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = rng.standard_normal(n_simulations)

    return S * np.exp(drift + diffusion_scale * Z_full)


def price_european_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 100_000,
    antithetic: bool = True,
    seed: Optional[int] = 42,
) -> MCResult:
    """
    Price a European call option via Monte Carlo simulation.

    Parameters
    ----------
    S             : current underlying price
    K             : strike price
    T             : time to expiry in years
    r             : continuously compounded risk-free rate (annualised)
    sigma         : volatility (annualised)
    n_simulations : number of GBM paths
    antithetic    : use antithetic variates for variance reduction
    seed          : random seed for reproducibility (None = unseeded)

    Returns
    -------
    MCResult dataclass with price, standard error, and 95% CI
    """
    S_T = _simulate_terminal_prices(S, T, r, sigma, n_simulations, antithetic, seed)
    payoffs = np.maximum(S_T - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_simulations)
    return MCResult(
        price=price,
        stderr=stderr,
        ci_lower=price - 1.96 * stderr,
        ci_upper=price + 1.96 * stderr,
        n_simulations=n_simulations,
    )


def price_european_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 100_000,
    antithetic: bool = True,
    seed: Optional[int] = 42,
) -> MCResult:
    """
    Price a European put option via Monte Carlo simulation.

    Parameters
    ----------
    Same as price_european_call.

    Returns
    -------
    MCResult dataclass with price, standard error, and 95% CI
    """
    S_T = _simulate_terminal_prices(S, T, r, sigma, n_simulations, antithetic, seed)
    payoffs = np.maximum(K - S_T, 0.0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_simulations)
    return MCResult(
        price=price,
        stderr=stderr,
        ci_lower=price - 1.96 * stderr,
        ci_upper=price + 1.96 * stderr,
        n_simulations=n_simulations,
    )


def price_both(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 100_000,
    antithetic: bool = True,
    seed: Optional[int] = 42,
) -> Dict[str, MCResult]:
    """
    Price both European call and put in a single simulation pass.
    More efficient than calling price_european_call and price_european_put
    separately since terminal prices are generated only once.

    Returns
    -------
    dict with keys 'call' and 'put', each an MCResult
    """
    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma**2) * T
    diffusion_scale = sigma * np.sqrt(T)

    if antithetic:
        half = n_simulations // 2
        Z = rng.standard_normal(half)
        Z_full = np.concatenate([Z, -Z])
    else:
        Z_full = rng.standard_normal(n_simulations)

    S_T = S * np.exp(drift + diffusion_scale * Z_full)
    discount = np.exp(-r * T)

    results = {}
    for name, payoffs in [
        ("call", np.maximum(S_T - K, 0.0)),
        ("put",  np.maximum(K - S_T, 0.0)),
    ]:
        disc = discount * payoffs
        p = disc.mean()
        se = disc.std(ddof=1) / np.sqrt(n_simulations)
        results[name] = MCResult(
            price=p,
            stderr=se,
            ci_lower=p - 1.96 * se,
            ci_upper=p + 1.96 * se,
            n_simulations=n_simulations,
        )
    return results
