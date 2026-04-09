# src/quantum/qaoa_optimizer.py

"""
The quantum circuit has parameters (gamma, beta) that need to be tuned. 
We use a classical optimizer (COBYLA — fast, derivative-free) in an outer loop. 
Each iteration: set parameters → run circuit on simulator → measure expectation 
value → optimizer updates parameters. 

This is the 'variational' part of a variational quantum algorithm.
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit_aer import AerSimulator

class QAOAOptimizer:
    def __init__(self, qaoa_circuit, qubo_matrix, n_shots=2048, backend='aer_simulator'):
        """
        n_shots: number of times circuit is measured per iteration.
                 More shots = less statistical noise, more compute time.
        backend: use Qiskit Aer statevector_simulator for noiseless sim (or aer_simulator).
        """
        self.qaoa_circuit = qaoa_circuit
        self.qubo_matrix = qubo_matrix
        self.n_shots = n_shots
        self.simulator = AerSimulator(method="statevector") if backend == 'statevector_simulator' else AerSimulator()
        
        self.iteration = 0
        self.convergence_history = []
        
    def expectation_value(self, params) -> float:
        """
        Given flat params array [gamma_0, gamma_1, ..., beta_0, beta_1, ...]:
        1. Split into gamma and beta lists.
        2. Build circuit with these params.
        3. Run on simulator with n_shots.
        4. Compute weighted average energy from results.
        """
        p = self.qaoa_circuit.p_layers
        gamma = params[:p]
        beta = params[p:]
        
        qc = self.qaoa_circuit.build_circuit(gamma, beta)
        compiled_circuit = transpile(qc, self.simulator)
        
        job = self.simulator.run(compiled_circuit, shots=self.n_shots)
        counts = job.result().get_counts()
        
        expected_energy = 0.0
        total_counts = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Qiskit orders output string as c_{n-1} c_{n-2} ... c_0. 
            # Reverse it so index 0 corresponds to the first character (stock 0)
            reversed_bitstring = bitstring[::-1]
            x = np.array([int(bit) for bit in reversed_bitstring])
            
            # Energy calculation: x^T Q x
            energy = x.T @ self.qubo_matrix @ x
            
            expected_energy += energy * (count / total_counts)
            
        self.iteration += 1
        self.convergence_history.append((self.iteration, expected_energy))
        
        return expected_energy

    def optimize(self, max_iterations=150) -> dict:
        """
        Run COBYLA optimization to find optimal (gamma, beta) angles.
        After finding them, run the circuit once more with high shot count
        to deeply sample the solution distribution.
        """
        self.iteration = 0
        self.convergence_history = []
        
        p = self.qaoa_circuit.p_layers
        initial_params = np.array([0.1] * p + [0.1] * p)
        
        res = minimize(
            self.expectation_value, 
            initial_params, 
            method='COBYLA', 
            options={'maxiter': max_iterations, 'rhobeg': 0.5}
        )
        
        optimal_gamma = res.x[:p]
        optimal_beta = res.x[p:]
        
        # Final high-resolution run
        qc = self.qaoa_circuit.build_circuit(optimal_gamma, optimal_beta)
        compiled_circuit = transpile(qc, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=8192)
        raw_counts = job.result().get_counts()
        
        # Reverse Qiskit bit ordering
        all_counts = {k[::-1]: v for k, v in raw_counts.items()}
        
        # Most frequent solution
        optimal_bitstring = max(all_counts, key=all_counts.get)
        
        return {
            'optimal_bitstring': optimal_bitstring,
            'optimal_params': {'gamma': optimal_gamma.tolist(), 'beta': optimal_beta.tolist()},
            'final_energy': res.fun,
            'n_iterations': self.iteration,
            'convergence_history': self.convergence_history,
            'all_counts': all_counts
        }
        
    def get_solution_distribution(self, counts: dict) -> list:
        """
        From measurement counts, return the top 10 bitstrings sorted by frequency.
        Format: list of (bitstring, count, probability, energy) tuples.
        """
        total_shots = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        top_10 = sorted_counts[:10]
        
        distribution = []
        for bitstring, count in top_10:
            prob = count / total_shots
            x = np.array([int(bit) for bit in bitstring])
            energy = x.T @ self.qubo_matrix @ x
            distribution.append((bitstring, count, prob, energy))
            
        return distribution
