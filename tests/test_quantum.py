# tests/test_quantum.py

import pytest
import numpy as np
import pandas as pd

from src.quantum.problem_formulator import PortfolioQUBO
from src.quantum.qaoa_circuit import QAOACircuit
from src.quantum.qaoa_optimizer import QAOAOptimizer
from src.quantum.benchmarker import QuantumClassicalBenchmark


@pytest.fixture
def mock_returns_df():
    # Create 5 dummy assets with 10 days of returns
    np.random.seed(42)
    data = np.random.normal(0.001, 0.02, (10, 5))
    tickers = ["A", "B", "C", "D", "E"]
    return pd.DataFrame(data, columns=tickers)


def test_portfolio_qubo_shape_and_symmetry(mock_returns_df):
    qubo = PortfolioQUBO(mock_returns_df, n_assets_to_select=3)
    Q = qubo.build_qubo_matrix()
    
    # 1. Shape should be N x N
    assert Q.shape == (5, 5)
    
    # 2. Symmetry: off-diagonals are derived from covariance which is symmetric.
    # However, depending on matrix representation, we should check if Q == Q.T
    # (Because covariance is symmetric and we added penalty uniformly)
    np.testing.assert_array_almost_equal(Q, Q.T)


def test_qubo_decode_bitstring(mock_returns_df):
    qubo = PortfolioQUBO(mock_returns_df, n_assets_to_select=3)
    
    # Bitstring '11100' -> selecting A, B, C
    result = qubo.decode_bitstring("11100")
    
    assert "selected_stocks" in result
    assert "portfolio_weights" in result
    assert "expected_return" in result
    assert "expected_risk" in result
    assert "sharpe_ratio" in result
    
    assert len(result["selected_stocks"]) == 3
    assert result["selected_stocks"] == ["A", "B", "C"]
    
    # Check bounds error
    with pytest.raises(ValueError):
        qubo.decode_bitstring("111")


def test_qaoa_circuit_creation_and_info(mock_returns_df):
    qubo = PortfolioQUBO(mock_returns_df, n_assets_to_select=3)
    Q = qubo.build_qubo_matrix()
    
    circuit_builder = QAOACircuit(Q, p_layers=2)
    info = circuit_builder.get_circuit_info()
    
    assert info["n_qubits"] == 5
    assert info["p_layers"] == 2
    assert "gate_count" in info
    assert "circuit_diagram" in info


def test_benchmark_output_structure(mock_returns_df):
    # Testing the full pipeline is expensive, so we configure a fast, tiny run
    benchmarker = QuantumClassicalBenchmark(mock_returns_df, list(mock_returns_df.columns))
    
    # Only 1 iteration for test speed
    res = benchmarker.compare(n_assets_to_select=3, p_layers=1, max_iterations=1)
    
    assert "classical" in res
    assert "quantum" in res
    assert "improvement" in res
    assert "execution_times" in res
    
    # Check that it matched the cardinality constraint best-effort
    selected = res["quantum"]["selected_stocks"]
    assert isinstance(selected, list)
    
    # (Because it's 1 iteration, it might randomly select wrong number of stocks, we just test keys)
    assert res["quantum"]["expected_return"] is not None
