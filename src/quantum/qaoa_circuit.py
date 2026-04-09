# src/quantum/qaoa_circuit.py

"""
QAOA (Quantum Approximate Optimization Algorithm) works in two phases 
repeated p times (p = circuit depth/layers):
1. Problem unitary (Uc): encodes the QUBO objective into phase rotations.
2. Mixer unitary (Um): explores the solution space via X-rotations.

The circuit starts in an equal superposition (all solutions equally likely),
then alternates Uc and Um with parameters gamma and beta.
A classical optimizer tunes gamma and beta to maximize the probability of 
measuring the optimal solution.
"""

from qiskit import QuantumCircuit

class QAOACircuit:
    def __init__(self, qubo_matrix, p_layers=2):
        """
        qubo_matrix: numpy array from PortfolioQUBO.build_qubo_matrix()
        p_layers: QAOA depth — more layers = better solution, but more noise.
                  p=2 is good for simulation on a laptop.
        """
        self.qubo_matrix = qubo_matrix
        self.p_layers = p_layers
        self.n_qubits = qubo_matrix.shape[0]

    def build_circuit(self, gamma, beta) -> QuantumCircuit:
        """
        Build the full QAOA circuit for given parameters.
        
        gamma: list of p_layers parameters for the Problem Unitary.
        beta: list of p_layers parameters for the Mixer Unitary.
        """
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        # 1. Apply H gate to all qubits (equal superposition)
        # Physically, this creates an equal probability of measuring any 
        # combination of stocks before we begin optimizing.
        for i in range(self.n_qubits):
            qc.h(i)
            
        qc.barrier()
        
        # 2. Alternating layers of Problem and Mixer unitaries
        for l in range(self.p_layers):
            
            # --- Problem Unitary Uc(gamma) ---
            # Encodes the objective function into the phases of the quantum state.
            
            # Interactions (Off-diagonals)
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    weight = self.qubo_matrix[i, j]
                    if weight != 0:
                        angle = gamma[l] * weight
                        # RZZ gate: encodes correlations between stock i and stock j.
                        # Implemented via CNOT -> RZ(2*theta) -> CNOT pattern.
                        qc.cx(i, j)
                        qc.rz(2 * angle, j)
                        qc.cx(i, j)
                        
            # Linear terms (Diagonals)
            for i in range(self.n_qubits):
                weight = self.qubo_matrix[i, i]
                if weight != 0:
                    angle = gamma[l] * weight
                    # RZ gate: encodes the individual weight of the stock.
                    qc.rz(angle, i)
                    
            qc.barrier()
            
            # --- Mixer Unitary Um(beta) ---
            # Drives transitions between different classical states,
            # allowing the algorithm to explore the solution landscape.
            for i in range(self.n_qubits):
                # RX gate tilts the qubits along the X-axis.
                qc.rx(2 * beta[l], i)
                
            qc.barrier()
            
        # 3. Measure all qubits
        # Collapses the quantum state into a classical solution bitstring.
        for i in range(self.n_qubits):
            qc.measure(i, i)
            
        return qc

    def get_circuit_info(self) -> dict:
        """Return basic metadata and diagram about a dummy instantiation."""
        # Create a dummy circuit to measure its properties
        dummy_gamma = [0.1] * self.p_layers
        dummy_beta = [0.1] * self.p_layers
        qc = self.build_circuit(dummy_gamma, dummy_beta)
        
        return {
            "n_qubits": self.n_qubits,
            "p_layers": self.p_layers,
            "gate_count": qc.size(),
            "circuit_depth": qc.depth(),
            "circuit_diagram": qc.draw(output='text')
        }
