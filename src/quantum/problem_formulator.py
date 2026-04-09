# src/quantum/problem_formulator.py

"""
Quantum Portfolio Optimization: Problem Formulator

This module converts the classical portfolio optimization problem into a 
Quadratic Unconstrained Binary Optimization (QUBO) problem that a quantum 
computer can solve natively.

Why QUBO?
Quantum computers (and specific algorithms like QAOA or Quantum Annealing) 
solve binary optimization problems naturally because each qubit can be in 
state |0> or |1>. We map our financial problem onto these states.

What each qubit represents:
In our formulation, each qubit 'i' represents a stock.
- State 1: Include stock 'i' in the portfolio.
- State 0: Exclude stock 'i' from the portfolio.

The Objective:
We want to minimize the risk while maximizing the return. In minimization form:
Minimize: Risk - λ * Return
where λ (risk_factor) is the tradeoff parameter between risk and return.
- λ = 0: Pure risk minimization.
- λ = 1: Pure return maximization.

The Constraint:
We want exactly K stocks selected out of N total stocks. (Cardinality constraint).
Because QUBO must be "Unconstrained", we add a penalty term to our objective function
that heavily penalizes any solution where the sum of selected stocks is not exactly K.
"""

import numpy as np
import pandas as pd


class PortfolioQUBO:
    def __init__(self, returns_df: pd.DataFrame, n_assets_to_select: int = 5, risk_factor: float = 0.5):
        """
        returns_df: DataFrame of daily returns for all Nifty 50 stocks in the universe.
        n_assets_to_select: Exactly how many stocks to include in the portfolio (K).
        risk_factor: The lambda tradeoff parameter between risk and return.
                     (0 = pure risk min, 1 = pure return max)
        """
        self.returns_df = returns_df
        self.tickers = list(returns_df.columns)
        self.n_assets = len(self.tickers)
        self.k = n_assets_to_select
        self.risk_factor = risk_factor
        
        # 252 trading days for annualization
        self.trading_days = 252 

    def compute_covariance_matrix(self) -> np.ndarray:
        """Return the annualized covariance matrix as a numpy array."""
        cov_matrix = self.returns_df.cov() * self.trading_days
        return cov_matrix.to_numpy()

    def compute_expected_returns(self) -> np.ndarray:
        """Return the annualized mean returns as a numpy array."""
        expected_returns = self.returns_df.mean() * self.trading_days
        return expected_returns.to_numpy()

    def build_qubo_matrix(self) -> np.ndarray:
        """
        Build the QUBO matrix Q such that the portfolio objective becomes:
        minimize x^T Q x
        where x is a binary vector (1=include stock, 0=exclude).
        
        Formulation:
        1. Objective: Q_base
           Diagonal entries: -expected_return[i] * (1 - risk_factor)
           Off-diagonal: covariance[i][j] * risk_factor
        2. Cardinality Penalty: P * (sum(x) - K)^2
           P = 3 * max(abs(Q_base)) to ensure constraint is strictly enforced.
           Expansion of (sum(x) - K)^2 for binary x:
           = (sum_i x_i^2 + sum_{i!=j} x_i x_j) - 2K sum_i x_i + K^2
           = sum_i (1 - 2K) x_i + sum_{i!=j} x_i x_j + K^2
        """
        expected_returns = self.compute_expected_returns()
        covariance = self.compute_covariance_matrix()
        
        Q_base = np.zeros((self.n_assets, self.n_assets))
        
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i == j:
                    Q_base[i, i] = -expected_returns[i] * (1 - self.risk_factor)
                else:
                    # Risk applies to interactions (variance/covariance)
                    Q_base[i, j] = covariance[i, j] * self.risk_factor
                    
        # Calculate penalty strength
        penalty_strength = 3 * np.max(np.abs(Q_base)) if np.max(np.abs(Q_base)) > 0 else 1.0
        
        # Add penalty to Q matrix
        Q = np.copy(Q_base)
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i == j:
                    Q[i, i] += penalty_strength * (1 - 2 * self.k)
                else:
                    Q[i, j] += penalty_strength
                    
        return Q

    def decode_bitstring(self, bitstring: str) -> dict:
        """
        Convert QAOA output bitstring (e.g. '1010110010') to portfolio metrics.
        The bitstring length must match the number of assets.
        We interpret it left-to-right matching the tickers list.
        """
        if len(bitstring) != self.n_assets:
            raise ValueError(f"Bitstring length {len(bitstring)} does not match n_assets {self.n_assets}")
            
        selected_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
        
        # If no stocks or invalid number selected, return zero metrics
        if len(selected_indices) == 0:
            return {
                "selected_stocks": [],
                "portfolio_weights": {},
                "expected_return": 0.0,
                "expected_risk": 0.0,
                "sharpe_ratio": 0.0
            }
            
        selected_stocks = [self.tickers[i] for i in selected_indices]
        
        # Equal weighting among selected stocks
        weight_per_stock = 1.0 / len(selected_indices)
        weights_array = np.zeros(self.n_assets)
        for i in selected_indices:
            weights_array[i] = weight_per_stock
            
        portfolio_weights = {ticker: weight_per_stock for ticker in selected_stocks}
        
        expected_returns = self.compute_expected_returns()
        covariance = self.compute_covariance_matrix()
        
        port_return = float(np.dot(weights_array, expected_returns))
        port_variance = float(weights_array.T @ covariance @ weights_array)
        port_risk = np.sqrt(port_variance)
        
        # Indian risk-free rate assumption (roughly 10yr G-Sec)
        risk_free_rate = 0.065
        sharpe_ratio = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0.0
        
        return {
            "selected_stocks": selected_stocks,
            "portfolio_weights": portfolio_weights,
            "expected_return": port_return,
            "expected_risk": port_risk,
            "sharpe_ratio": sharpe_ratio
        }
