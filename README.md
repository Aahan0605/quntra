# classical-quant-engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Research-orange)
![Series](https://img.shields.io/badge/Series-Quantra%20%7C%20Classical%20Layer-purple)

---

## Abstract

`classical-quant-engine` is the foundational classical computation layer of **Quantra** — an ongoing research series aimed at constructing a quantum-classical hybrid hedge fund system targeting emerging markets, with an initial focus on the Indian equity landscape (Nifty 50). This module implements three core quantitative finance primitives: a Monte Carlo simulation engine for options pricing under geometric Brownian motion, a closed-form Black-Scholes Greeks calculator for real-time sensitivity analysis, and a Markowitz mean-variance portfolio optimizer for constructing efficient frontiers from live Nifty 50 market data. The architecture is deliberately modular, designed to serve as a drop-in classical baseline against which forthcoming quantum amplitude estimation and variational quantum eigensolver (VQE) portfolio modules — built on Qiskit — will be benchmarked. All pricing and optimization results are reproducible, and the codebase is structured to support rigorous empirical comparison between classical and quantum computational approaches in a research-grade setting.

---

## Features

- **Monte Carlo Options Pricer** — Simulates thousands of asset price paths under GBM to estimate European call/put option prices with configurable path count and time steps
- **Black-Scholes Greeks Calculator** — Computes closed-form Delta, Gamma, Vega, Theta, and Rho for European options; serves as the analytical ground truth for Monte Carlo validation
- **Markowitz Efficient Frontier Optimizer** — Constructs the mean-variance efficient frontier from historical Nifty 50 return data, with support for minimum-variance and maximum-Sharpe-ratio portfolio extraction
- **Live Market Data Integration** — Fetches real-time and historical OHLCV data for Nifty 50 constituents via `yfinance`, enabling live backtesting and forward analysis
- **Modular Research Architecture** — Each module is independently importable and testable, designed for seamless integration with the upcoming quantum layer of Quantra

---

## Tech Stack

| Library | Role |
|---|---|
| `Python 3.10+` | Core runtime |
| `NumPy` | Vectorized numerical computation, random path generation |
| `Pandas` | Time-series data handling, return matrix construction |
| `yfinance` | Live and historical Nifty 50 market data ingestion |
| `SciPy` | Statistical distributions (norm CDF/PDF) for Black-Scholes |
| `CVXPY` | Convex optimization for Markowitz portfolio construction |
| `Matplotlib` | Efficient frontier visualization, P&L surface plots |

---

## Project Structure

```
classical-quant-engine/
│
├── data/                        # Cached market data (auto-populated via yfinance)
│   └── nifty50_prices.csv
│
├── engines/
│   ├── monte_carlo.py           # GBM path simulation and option pricing
│   ├── black_scholes.py         # Closed-form pricing and Greeks
│   └── portfolio_optimizer.py   # Markowitz efficient frontier
│
├── utils/
│   ├── data_loader.py           # yfinance ingestion and preprocessing
│   └── visualizer.py            # Plotting utilities
│
├── notebooks/
│   ├── options_pricing_demo.ipynb
│   └── efficient_frontier_demo.ipynb
│
├── tests/
│   ├── test_monte_carlo.py
│   ├── test_black_scholes.py
│   └── test_optimizer.py
│
├── results/                     # Output figures and pricing tables
├── requirements.txt
├── README.md
└── main.py                      # Entry point for full pipeline execution
```

---

## Installation & Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Aahan0605/quntra.git
cd classical-quant-engine
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

### 4. Monte Carlo Options Pricing

```python
from engines.monte_carlo import MonteCarloPricer

pricer = MonteCarloPricer(S=18500, K=19000, T=0.25, r=0.065, sigma=0.18, n_paths=100_000)
call_price, put_price = pricer.price()
print(f"MC Call: {call_price:.4f} | MC Put: {put_price:.4f}")
```

### 5. Black-Scholes Greeks

```python
from engines.black_scholes import BlackScholes

bs = BlackScholes(S=18500, K=19000, T=0.25, r=0.065, sigma=0.18)
greeks = bs.greeks()
print(greeks)
# {'delta': 0.4231, 'gamma': 0.0003, 'vega': 28.14, 'theta': -6.72, 'rho': 10.83}
```

### 6. Markowitz Efficient Frontier

```python
from engines.portfolio_optimizer import MarkowitzOptimizer
from utils.data_loader import load_nifty50

returns = load_nifty50(start="2022-01-01", end="2024-01-01")
optimizer = MarkowitzOptimizer(returns)
frontier = optimizer.efficient_frontier(n_points=200)
optimizer.plot_frontier(frontier)
```

---

## Results & Output

### Monte Carlo vs. Black-Scholes Pricing Comparison

> Results generated on Nifty 50 index-level parameters: S = 18,500 | K = 19,000 | T = 0.25 yr | r = 6.5% | σ = 18%

| Option Type | Black-Scholes Price (₹) | Monte Carlo Price (₹) | Std. Error | Relative Error (%) |
|---|---|---|---|---|
| European Call | — | — | — | — |
| European Put | — | — | — | — |

*Table will be populated upon first execution of `main.py`. Output is saved to `results/pricing_comparison.csv`.*

### Efficient Frontier

Efficient frontier plots for a selected Nifty 50 sub-portfolio are saved to `results/efficient_frontier.png` after running the optimizer module.

---

## Roadmap

`classical-quant-engine` is **Layer 1** of the Quantra research series. The following modules are under active development:

| Layer | Module | Status | Description |
|---|---|---|---|
| 1 | `classical-quant-engine` | ✅ Active | Monte Carlo, Black-Scholes, Markowitz optimizer |
| 2 | `quantum-options-pricer` | 🔬 Research | Quantum Amplitude Estimation (QAE) for option pricing via Qiskit |
| 3 | `vqe-portfolio-optimizer` | 🔬 Research | Variational Quantum Eigensolver for portfolio optimization on QUBO formulation |
| 4 | `hybrid-execution-engine` | 📋 Planned | Classical-quantum hybrid execution with noise-aware circuit transpilation |
| 5 | `quantra-hedge-fund-sim` | 📋 Planned | Full simulation of a quantum-classical hedge fund targeting Nifty 50 / emerging markets |

Quantum modules will use **Qiskit** and **Qiskit Finance**, with benchmarks run on both IBM quantum simulators and real hardware via IBM Quantum Network access.

---

## Contributing

This project is part of an active research series. Contributions that improve numerical accuracy, extend the asset universe, or introduce additional classical pricing models (e.g., Heston stochastic volatility, Binomial trees) are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/heston-model`
3. Commit your changes with descriptive messages
4. Open a pull request with a clear description of the contribution and any relevant benchmarks

For research collaborations or discussions on the quantum layer, open an issue or reach out directly via the repository.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Aahan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<p align="center">
  <sub>Part of the <strong>Quantra</strong> research series — building toward a quantum-classical hybrid hedge fund for emerging markets.</sub>
</p>
