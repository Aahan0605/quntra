# src/quantum/benchmarker.py

import time
import numpy as np
from src.portfolio.markowitz import max_sharpe_portfolio, compute_stats
from src.quantum.problem_formulator import PortfolioQUBO
from src.quantum.qaoa_circuit import QAOACircuit
from src.quantum.qaoa_optimizer import QAOAOptimizer

class QuantumClassicalBenchmark:
    def __init__(self, returns_df, tickers):
        self.returns_df = returns_df
        self.tickers = tickers
        
        # We need these for the classical formulation
        # We use trading_days=252 for annualisation
        daily_returns = self.returns_df
        self.mu = daily_returns.mean().values * 252
        self.Sigma = daily_returns.cov().values * 252
        
    def run_classical(self) -> dict:
        """
        Run the existing classical optimization pipeline (from markowitz.py).
        """
        result = max_sharpe_portfolio(self.mu, self.Sigma, self.tickers)
        
        # Add a selected_stocks list based on non-zero weights
        selected = [t for t, w in result["weights"].items() if w > 1e-4]
        result["selected_stocks"] = selected
        
        return result
        
    def run_quantum(self, n_assets_to_select=5, p_layers=2, max_iterations=150) -> dict:
        """
        Run the full QAOA pipeline.
        1. Formulate QUBO
        2. Build parameterized QAOA Circuit
        3. Optimize parameters iteratively
        4. Decode resulting best bitstring
        """
        # 1. Formulation
        qubo = PortfolioQUBO(self.returns_df, n_assets_to_select)
        Q = qubo.build_qubo_matrix()
        
        # 2. Circuit Building
        circuit_builder = QAOACircuit(Q, p_layers)
        
        # 3. Optimization
        optimizer = QAOAOptimizer(circuit_builder, Q, n_shots=2048)
        opt_results = optimizer.optimize(max_iterations)
        
        # 4. Decoding
        best_bitstring = opt_results['optimal_bitstring']
        decoded = qubo.decode_bitstring(best_bitstring)
        
        # Attach quantum-specific metadata
        decoded['convergence_history'] = opt_results['convergence_history']
        decoded['circuit_info'] = circuit_builder.get_circuit_info()
        decoded['solution_distribution'] = optimizer.get_solution_distribution(opt_results['all_counts'])
        
        return decoded
        
    def compare(self, n_assets_to_select=5, p_layers=2, max_iterations=150) -> dict:
        """
        Run both pipelines and compute exactly how they compare.
        """
        print("Running Classical Markowitz Optimizer...")
        c_start = time.time()
        c_res = self.run_classical()
        c_time = time.time() - c_start
        
        print("Running QAOA Optimizer...")
        q_start = time.time()
        q_res = self.run_quantum(n_assets_to_select, p_layers, max_iterations)
        q_time = time.time() - q_start
        
        c_sharpe = c_res['sharpe_ratio']
        q_sharpe = q_res['sharpe_ratio']
        
        # The prompt defines 'improvement' based on the quantum metrics vs classical metrics
        sharpe_delta = q_sharpe - c_sharpe
        sharpe_pct = ((q_sharpe / c_sharpe) - 1) * 100 if c_sharpe > 0 else 0
        
        c_weights_sorted = sorted(c_res['weights'].items(), key=lambda x: x[1], reverse=True)
        classical_top_k = [x[0] for x in c_weights_sorted[:n_assets_to_select]]
        
        return {
            'classical': c_res,
            'quantum': q_res,
            'improvement': {
                'sharpe_delta': sharpe_delta,
                'sharpe_pct': sharpe_pct,
                'return_delta': q_res['expected_return'] - c_res['expected_return'],
                'risk_delta': q_res['expected_risk'] - c_res['volatility']
            },
            'quantum_selected_stocks': q_res['selected_stocks'],
            'classical_top_k': classical_top_k,
            'execution_times': {
                'classical': c_time,
                'quantum': q_time
            }
        }
