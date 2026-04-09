"""
Visualisation utilities for the Quantra Classical Engine.

Provides high-fidelity, premium plots for Options Pricing surfaces
and the Markowitz Efficient Frontier.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm
from typing import Optional


def set_plot_style():
    """Apply consistent, premium styling to all plots."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    max_sharpe_result: Optional[dict] = None,
    risk_free_rate: float = 0.065,
    save_path: Optional[str] = None,
) -> None:
    """Plot the Markowitz Efficient Frontier curve."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        frontier_df["volatility"] * 100,
        frontier_df["target_return"] * 100,
        lw=2.5, color="#1f77b4", label="Efficient Frontier",
    )

    if max_sharpe_result:
        ax.scatter(
            max_sharpe_result["volatility"] * 100,
            max_sharpe_result["expected_return"] * 100,
            marker="*", s=350, color="#d62728", edgecolors="black",
            zorder=5, label=f"Max Sharpe ({max_sharpe_result['sharpe_ratio']:.2f})",
        )

    ax.set_xlabel("Annualised Volatility (%)", fontsize=11)
    ax.set_ylabel("Annualised Expected Return (%)", fontsize=11)
    ax.set_title("Markowitz Efficient Frontier — Nifty 50 Sub-Portfolio", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()


def plot_vol_surface(
    vol_range: np.ndarray,
    T_range: np.ndarray,
    prices: np.ndarray,
    title: str = "Option Price Surface",
    save_path: Optional[str] = None,
) -> None:
    """Plot a 3D surface of option prices vs volatility and maturity."""
    set_plot_style()
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    VOL, MAT = np.meshgrid(vol_range * 100, T_range)
    surf = ax.plot_surface(VOL, MAT, prices, cmap=cm.viridis, alpha=0.9, linewidth=0)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Option Price (₹)")
    ax.set_xlabel("Implied Volatility (%)")
    ax.set_ylabel("Time to Expiry (yr)")
    ax.set_zlabel("Price (₹)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()

# =====================================================================
# MONTH 2 — QUANTUM VISUALIZATIONS
# =====================================================================

def plot_convergence(convergence_history: list, save_path: str) -> None:
    """
    Line plot of QAOA optimization convergence.
    X: iteration number | Y: expectation value (energy)
    Style: dark background (#0a0a0a), accent color line, grid in #1a1a1a
    Title: "QAOA Convergence — Energy vs Iteration"
    Add horizontal dashed line at final value labeled "OPTIMAL"
    Save to results/quantum/qaoa_convergence.png
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    iterations = [x[0] for x in convergence_history]
    energies = [x[1] for x in convergence_history]
    final_energy = energies[-1]
    
    ax.plot(iterations, energies, color="#00ff88", lw=2)
    ax.axhline(final_energy, color="#ff4444", linestyle="--", alpha=0.7)
    
    # Label OPTIMAL
    ax.text(iterations[0], final_energy + 0.1, f"OPTIMAL: {final_energy:.3f}", color="#ff4444", 
            fontsize=10, verticalalignment='bottom')
            
    ax.set_xlabel("Iteration", color="#cccccc", fontsize=11)
    ax.set_ylabel("Expectation Value (Energy)", color="#cccccc", fontsize=11)
    ax.set_title("QAOA Convergence — Energy vs Iteration", color="#cccccc", fontsize=13, fontweight="bold")
    
    ax.tick_params(colors="#cccccc")
    ax.grid(color="#1a1a1a", linestyle="--", alpha=0.5)
    
    for spine in ax.spines.values():
        spine.set_color("#1a1a1a")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def plot_solution_distribution(solution_dist: list, save_path: str) -> None:
    """
    Horizontal bar chart of top 10 QAOA measurement outcomes.
    Y: bitstring labels | X: probability (count/total_shots)
    Highlight the optimal (highest probability) bar in accent color.
    Other bars in #334433. Add probability % labels on each bar.
    Title: "QAOA Solution Distribution — Top 10 Bitstrings"
    Save to results/quantum/solution_distribution.png
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    bitstrings = [x[0] for x in solution_dist]
    probs = [x[2] for x in solution_dist]
    
    y_pos = np.arange(len(bitstrings))
    colors = ["#00ff88" if i == 0 else "#334433" for i in range(len(bitstrings))]
    
    bars = ax.barh(y_pos, probs, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bitstrings, fontfamily='monospace', color="#cccccc")
    ax.invert_yaxis()  # highest prob at the top
    
    ax.set_xlabel("Probability", color="#cccccc", fontsize=11)
    ax.set_title("QAOA Solution Distribution — Top 10 Bitstrings", color="#cccccc", fontsize=13, fontweight="bold")
    
    ax.tick_params(colors="#cccccc")
    ax.grid(axis='x', color="#1a1a1a", linestyle="--", alpha=0.5)
    
    for spine in ax.spines.values():
        spine.set_color("#1a1a1a")
        
    # Add labels
    for bar in bars:
        width = bar.get_width()
        label_x = width + 0.005
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{width * 100:.1f}%", va='center', color="#cccccc", fontsize=9)
                
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def plot_quantum_vs_classical(benchmark_results: dict, frontier_df: pd.DataFrame, save_path: str) -> None:
    """
    4-panel comparison figure (2x2 grid):
    Panel 1 — Portfolio Weights Bar Chart
    Panel 2 — Risk/Return Scatter
    Panel 3 — Metrics Comparison Table
    Panel 4 — QAOA Convergence
    Overall title: "QUANTRA: Quantum vs Classical Portfolio Optimization"
    Subtitle: "QAOA (p=2, 10 qubits) vs Markowitz — Nifty 50 Universe"
    Save to results/quantum/benchmark_comparison.png
    """
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    plt.suptitle("QUANTRA: Quantum vs Classical Portfolio Optimization", 
                 color="#00ff88", fontsize=20, fontweight="bold", y=0.96)
    plt.figtext(0.5, 0.92, "QAOA (p=2, 10 qubits) vs Markowitz — Nifty 50 Universe", 
                ha="center", color="#cccccc", fontsize=14)
                
    c_res = benchmark_results['classical']
    q_res = benchmark_results['quantum']
    
    # -----------------------------
    # Panel 1: Weights Bar Chart
    # -----------------------------
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor('#0a0a0a')
    tickers = list(c_res['weights'].keys())
    c_weights = [c_res['weights'].get(t, 0) * 100 for t in tickers]
    q_weights = [q_res['portfolio_weights'].get(t, 0) * 100 for t in tickers]
    
    x = np.arange(len(tickers))
    width = 0.35
    
    ax1.bar(x - width/2, c_weights, width, label='Classical', color='#555555')
    ax1.bar(x + width/2, q_weights, width, label='QAOA', color='#00ff88')
    
    ax1.set_ylabel('Weight (%)', color="#cccccc")
    ax1.set_title('Portfolio Allocations', color="#cccccc")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, rotation=45, ha="right", color="#cccccc")
    ax1.legend(facecolor="#111111", edgecolor="#333333", labelcolor="#cccccc")
    ax1.tick_params(colors="#cccccc")
    ax1.grid(axis='y', color="#1a1a1a", linestyle="--", alpha=0.5)
    for spine in ax1.spines.values(): spine.set_color("#1a1a1a")

    # -----------------------------
    # Panel 2: Risk/Return Scatter
    # -----------------------------
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor('#0a0a0a')
    
    if frontier_df is not None:
        ax2.plot(frontier_df["volatility"] * 100, frontier_df["target_return"] * 100,
                 color="#555555", linestyle="--", label="Classical Frontier")
                 
    # Classical Point
    c_vol = c_res['volatility'] * 100
    c_ret = c_res['expected_return'] * 100
    ax2.scatter(c_vol, c_ret, marker="o", s=200, color="#555555", label="Classical", zorder=5)
    ax2.annotate(f"Sharpe: {c_res['sharpe_ratio']:.3f}", (c_vol, c_ret), 
                 xytext=(10, 10), textcoords='offset points', color="#cccccc", fontsize=10)
                 
    # Quantum Point
    q_vol = q_res['expected_risk'] * 100
    q_ret = q_res['expected_return'] * 100
    ax2.scatter(q_vol, q_ret, marker="*", s=300, color="#00ff88", label="QAOA Optimal", zorder=5)
    ax2.annotate(f"Sharpe: {q_res['sharpe_ratio']:.3f}", (q_vol, q_ret), 
                 xytext=(10, -20), textcoords='offset points', color="#00ff88", fontsize=10)
                 
    ax2.set_xlabel("Portfolio Risk (σ) %", color="#cccccc")
    ax2.set_ylabel("Expected Return (μ) %", color="#cccccc")
    ax2.set_title("Risk-Return Profile", color="#cccccc")
    ax2.legend(facecolor="#111111", edgecolor="#333333", labelcolor="#cccccc")
    ax2.tick_params(colors="#cccccc")
    ax2.grid(color="#1a1a1a", linestyle="--", alpha=0.5)
    for spine in ax2.spines.values(): spine.set_color("#1a1a1a")

    # -----------------------------
    # Panel 3: Metrics Table
    # -----------------------------
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('#0a0a0a')
    ax3.axis('off')
    
    cell_text = [
        ["Sharpe Ratio", f"{c_res['sharpe_ratio']:.3f}", f"{q_res['sharpe_ratio']:.3f}"],
        ["Expected Return", f"{c_ret:.2f}%", f"{q_ret:.2f}%"],
        ["Portfolio Risk", f"{c_vol:.2f}%", f"{q_vol:.2f}%"],
        ["Selected Stocks", str(len(c_res['selected_stocks'])), str(len(q_res['selected_stocks']))],
        ["Improvement (Sharpe)", "—", f"{benchmark_results['improvement']['sharpe_pct']:+.1f}%"]
    ]
    
    table = ax3.table(cellText=cell_text, colLabels=["Metric", "Classical", "Quantum"],
                      loc='center', cellLoc='center')
                      
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#1a1a1a')
        if key[0] == 0:  # Header row
            cell.set_facecolor('#111111')
            cell.set_text_props(color='#cccccc', fontweight='bold')
        else:
            cell.set_facecolor('#0a0a0a')
            cell.set_text_props(color='#cccccc')
            
        # Color improvement
        if key[0] == 5 and key[1] == 2:
            color = "#00ff88" if benchmark_results['improvement']['sharpe_pct'] > 0 else "#ff4444"
            cell.set_text_props(color=color, fontweight='bold')
            
    ax3.set_title("Performance Metrics", color="#cccccc")

    # -----------------------------
    # Panel 4: Convergence (Reuse)
    # -----------------------------
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('#0a0a0a')
    
    conv = q_res['convergence_history']
    iterations = [x[0] for x in conv]
    energies = [x[1] for x in conv]
    final_energy = energies[-1]
    
    ax4.plot(iterations, energies, color="#00ff88", lw=2)
    ax4.axhline(final_energy, color="#ff4444", linestyle="--", alpha=0.7)
    ax4.text(iterations[0], final_energy + 0.1, f"OPTIMAL: {final_energy:.3f}", color="#ff4444", fontsize=10)
    
    ax4.set_xlabel("Iteration", color="#cccccc")
    ax4.set_ylabel("Expectation Value", color="#cccccc")
    ax4.set_title("QAOA Optimizer Convergence", color="#cccccc")
    ax4.tick_params(colors="#cccccc")
    ax4.grid(color="#1a1a1a", linestyle="--", alpha=0.5)
    for spine in ax4.spines.values(): spine.set_color("#1a1a1a")

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    if save_path:
        plt.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()


def plot_circuit_diagram(circuit, save_path: str) -> None:
    """Save the Qiskit circuit diagram as an image."""
    fig = circuit.draw(output='mpl', style={
        'backgroundcolor': '#0a0a0a',
        'textcolor': '#cccccc', 
        'gatefacecolor': '#1a3a2a',
        'gatetextcolor': '#00ff88', 
        'barrierfacecolor': '#1a1a1a'
    })
    if save_path:
        fig.savefig(save_path, facecolor='#0a0a0a', dpi=150)
    # Circuit draw returns a matplotlib Figure. No plt.close() needed here directly if we don't use pyplot

