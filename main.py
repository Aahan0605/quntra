"""
Quantra Classical Engine — Main Entry Point

Demonstrates the full classical pipeline:
1. European Options Pricing (Black-Scholes vs Monte Carlo) for Nifty 50.
2. Markowitz Portfolio Optimization for a Nifty 50 sub-portfolio.
3. Generation of benchmark plots and results in the 'results/' directory.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict

from src.options.black_scholes import call_price, put_price, greeks
from src.options.monte_carlo import price_both, MCResult
from src.portfolio.markowitz import run as run_optimizer, efficient_frontier
from src.utils.data_loader import fetch_nifty50_prices, get_returns
from src.utils.visualizer import (
    set_plot_style, plot_efficient_frontier, plot_vol_surface,
    plot_convergence, plot_solution_distribution, plot_quantum_vs_classical
)
from src.quantum.benchmarker import QuantumClassicalBenchmark
from src.backtest.engine import BacktestEngine

# --- Configuration ---
RESULTS_DIR = "results"
QUANTUM_RESULTS_DIR = os.path.join(RESULTS_DIR, "quantum")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(QUANTUM_RESULTS_DIR, exist_ok=True)
set_plot_style()

# Nifty 50 constituents for portfolio optimization
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "AXISBANK.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS",
]

# Base parameters for options demo
S_spot = 22000.0   # Approximate Nifty 50 spot
T_expiry = 0.25    # 3-month
r_rate = 0.065     # Risk-free rate
sigma_vol = 0.18   # Annualised volatility


def run_options_demo():
    print("\n--- Phase 1: Options Pricing Benchmark ---")
    print(f"Parameters: Spot={S_spot}, Strike=ATM, T={T_expiry}, r={r_rate*100}%, sigma={sigma_vol*100}%")
    
    # 1. ATM Comparison
    bs_call = call_price(S_spot, S_spot, T_expiry, r_rate, sigma_vol)
    mc_results = price_both(S_spot, S_spot, T_expiry, r_rate, sigma_vol, n_simulations=100_000)
    
    print(f"Black-Scholes ATM Call: ₹{bs_call:.2f}")
    print(f"Monte Carlo ATM Call:   ₹{mc_results['call'].price:.2f} (±{mc_results['call'].stderr:.4f})")
    
    # 2. Generate Volatility Surface Data
    vols = np.linspace(0.10, 0.40, 20)
    times = np.linspace(0.05, 1.0, 20)
    prices = np.zeros((len(times), len(vols)))
    
    for i, t in enumerate(times):
        for j, v in enumerate(vols):
            prices[i, j] = call_price(S_spot, S_spot, t, r_rate, v)
    
    plot_vol_surface(vols, times, prices, 
                     title="ATM Call Price Surface — Vol × Maturity",
                     save_path=os.path.join(RESULTS_DIR, "vol_surface.png"))
    print(f"Volatility surface plot saved to {RESULTS_DIR}/vol_surface.png")


def run_portfolio_demo():
    print("\n--- Phase 2: Markowitz Portfolio Optimization ---")
    print(f"Optimizing for: {', '.join(TICKERS)}")
    
    result = run_optimizer(
        tickers=TICKERS,
        start="2022-01-01",
        end="2024-01-01",
        plot=True,
        frontier_save_path=os.path.join(RESULTS_DIR, "efficient_frontier.png")
    )
    
    print("\nOptimal Portfolio (Max Sharpe Ratio):")
    print(f"  Expected Annual Return: {result['expected_return']*100:.2f}%")
    print(f"  Annual Volatility:      {result['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio:           {result['sharpe_ratio']:.2f}")
    
    # Print Top 5 allocations
    sorted_weights = sorted(result["weights"].items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Allocations:")
    for ticker, weight in sorted_weights[:5]:
        if weight > 0:
            print(f"  {ticker}: {weight*100:.2f}%")
            
    print(f"\nEfficient frontier plot saved to {RESULTS_DIR}/efficient_frontier.png")


def run_quantum_demo():
    print("\n--- Phase 3: Quantum QAOA Benchmark ---")
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    
    print(f"Fetching data for {len(TICKERS)} tickers...")
    prices = fetch_nifty50_prices(TICKERS, start_date, end_date)
    returns_df = get_returns(prices)
    
    print("Initiating Quantum vs Classical Benchmark...")
    benchmarker = QuantumClassicalBenchmark(returns_df, TICKERS)
    results = benchmarker.compare(n_assets_to_select=5, p_layers=2, max_iterations=60)
    
    print("\nBenchmark Complete!")
    print(f"Classical Time: {results['execution_times']['classical']:.2f}s | Quantum Time: {results['execution_times']['quantum']:.2f}s")
    
    # Plotting
    print("Generating Quantum Visualizations...")
    plot_convergence(
        results['quantum']['convergence_history'],
        os.path.join(QUANTUM_RESULTS_DIR, "qaoa_convergence.png")
    )
    
    plot_solution_distribution(
        results['quantum']['solution_distribution'],
        os.path.join(QUANTUM_RESULTS_DIR, "solution_distribution.png")
    )
    
    # Needs frontier for the 4-panel graph, which we can generate quickly
    mu = benchmarker.mu
    Sigma = benchmarker.Sigma
    frontier_df = efficient_frontier(mu, Sigma)
    
    plot_quantum_vs_classical(
        results, frontier_df,
        os.path.join(QUANTUM_RESULTS_DIR, "benchmark_comparison.png")
    )
    print(f"Quantum plots saved to {QUANTUM_RESULTS_DIR}/")

def run_backtest_demo():
    print("\n--- Phase 4: Event-driven Historical Backtesting ---")
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    
    # 1. Fetch classical weights
    print("1. Computing Reference Target Weights (Classical Markowitz)...")
    prices = fetch_nifty50_prices(TICKERS, start_date, end_date)
    returns_df = get_returns(prices)
    mu = returns_df.mean().values * 252
    Sigma = returns_df.cov().values * 252
    
    from src.portfolio.markowitz import max_sharpe_portfolio
    result = max_sharpe_portfolio(mu, Sigma, TICKERS)
    weights = result["weights"]
    
    # 2. Run engine
    print(f"2. Simulating Backtest Engine (Initial Capital = ₹1,000,000, Rebalance = Monthly)...")
    engine = BacktestEngine(
        tickers=TICKERS,
        weights_dict=weights,
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0
    )
    bt_result = engine.run(rebalance_freq='monthly')
    
    # 3. Print Metrics
    metrics = bt_result["metrics"]
    print("\nBacktest Performance Metrics:")
    print(f"  Total Return:           {metrics['total_return']*100:.2f}%")
    print(f"  Annualized Return:      {metrics['annualized_return']*100:.2f}%")
    print(f"  Annualized Volatility:  {metrics['annualized_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:           {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate:               {metrics['win_rate']*100:.2f}%")
    
    # Attribution
    print("\nTop 3 Performance Drivers:")
    sorted_attr = sorted(bt_result["attribution"].items(), key=lambda x: x[1], reverse=True)
    for ticker, contrib in sorted_attr[:3]:
        print(f"  {ticker}: {contrib*100:.2f}% Return Contribution")


def main():
    print("====================================================")
    print("          QUANTRA CLASSICAL QUANT ENGINE            ")
    print("====================================================")
    
    try:
        run_options_demo()
        run_portfolio_demo()
        run_quantum_demo()
        run_backtest_demo()
        print("\nPipeline completed successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nPipeline failed: {e}")


if __name__ == "__main__":
    main()
