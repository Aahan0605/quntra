#!/usr/bin/env python3
"""
Full Phase 0 validation — all 3 targets must pass:

    Sharpe ratio  > 1.00   (after ICICI costs)
    Max drawdown  > -15.0%
    Calmar ratio  > 0.70

Runs a walk-forward backtest over the 25-ticker universe with the
weekly drift-threshold rebalancer and the ICICI cost model, using the
offline data cache (data/cache/). Weights are re-estimated each quarter
from trailing data only (no lookahead): inverse-volatility base tilted
by 6-month momentum, 20% per-ticker cap.

If real cached data is missing, the pipeline runs on SYNTHETIC data to
prove mechanics, but is clearly labeled and NEVER prints PASS.

Results -> data/backtest_results/validation_post_fix.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.universe import UNIVERSE  # noqa: E402
from src.utils.cache_loader import cache_available, load_close_panel  # noqa: E402
from src.utils.costs import CostModel  # noqa: E402
from src.portfolio.rebalancer import Rebalancer  # noqa: E402
from src.backtest.metrics import BacktestMetrics  # noqa: E402

TARGETS = {"sharpe_ratio": 1.00, "max_drawdown": -0.15, "calmar_ratio": 0.70}
OUT = ROOT / "data" / "backtest_results" / "validation_post_fix.json"

ESTIMATION_WINDOW = 252      # 1y trailing window for weight estimation
REESTIMATE_EVERY = 63        # quarterly
WEIGHT_CAP = 0.20


def estimate_weights(returns: pd.DataFrame) -> dict[str, float]:
    """Inverse-volatility base tilted by 6-month momentum. Long-only, capped."""
    vol = returns.std()
    inv_vol = 1.0 / vol.replace(0, np.nan)
    mom = (1 + returns.tail(126)).prod() - 1
    tilt = (1 + mom.clip(-0.5, 0.5)).clip(lower=0.25)
    w = (inv_vol * tilt).fillna(0)
    w = w.clip(upper=w.sum() * WEIGHT_CAP)
    return (w / w.sum()).to_dict()


def walk_forward(panel: pd.DataFrame) -> tuple[pd.Series, float]:
    """Walk-forward simulation. Returns (daily portfolio returns, annual turnover)."""
    rets = panel.pct_change().dropna()
    cols = list(rets.columns)
    cost = CostModel.from_config()
    friction = cost.one_way_cost_rate(delivery=True) + cost.slippage_rate()
    rb = Rebalancer()  # weekly, 3% drift, 20% cap

    start = ESTIMATION_WINDOW
    target = None
    current = None
    out_idx, out_ret = [], []
    total_turnover = 0.0

    for i in range(start, len(rets)):
        # (Re)estimate targets quarterly from trailing data only
        if (i - start) % REESTIMATE_EVERY == 0:
            window = rets.iloc[i - ESTIMATION_WINDOW:i]
            target = estimate_weights(window)
            if current is None:
                current = np.array([target[c] for c in cols])
                out_idx.append(rets.index[i])
                out_ret.append(float(rets.iloc[i] @ current) - friction)
                continue

        row = rets.iloc[i]
        daily = float(row @ current)

        growth = current * (1.0 + row.values)
        s = growth.sum()
        if s > 0:
            current = growth / s

        ts = rets.index[i]
        decision = rb.compute_trades(dict(zip(cols, current)), target, ts.date())
        if decision.should_rebalance:
            daily -= 2.0 * decision.one_way_turnover * friction
            total_turnover += decision.one_way_turnover
            for j, c in enumerate(cols):
                current[j] += decision.trades.get(c, 0.0)
            current = np.clip(current, 0, None)
            current = current / current.sum()

        out_idx.append(ts)
        out_ret.append(daily)

    series = pd.Series(out_ret, index=pd.DatetimeIndex(out_idx))
    years = len(series) / 252.0
    return series, (total_turnover / years if years > 0 else 0.0)


def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 756
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    base = rng.normal(0.0004, 0.009, size=n)  # market factor
    data = {}
    for k, t in enumerate(UNIVERSE):
        idio = rng.normal(0.0002, 0.012, size=n)
        r = 0.7 * base + idio
        data[t] = 100 * np.cumprod(1 + r)
    return pd.DataFrame(data, index=idx)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="print cost drag and turnover diagnostics")
    ap.add_argument("--save-tearsheet", action="store_true",
                    help="save equity/drawdown PNG next to the JSON")
    args = ap.parse_args()

    real_data = cache_available(UNIVERSE)
    if real_data:
        panel = load_close_panel(UNIVERSE)
        source = "cache (real NSE data)"
    else:
        panel = synthetic_panel()
        source = "SYNTHETIC (data/cache missing — run scripts/fetch_data_cache.py)"

    port_returns, annual_turnover = walk_forward(panel)
    m = BacktestMetrics.calculate_metrics(port_returns)
    m["annual_turnover"] = annual_turnover

    if args.verbose:
        cost = CostModel.from_config()
        friction = cost.one_way_cost_rate(delivery=True) + cost.slippage_rate()
        annual_cost_drag = 2.0 * annual_turnover * friction
        print("--- diagnostics ---")
        print(f"one-way friction/trade : {friction:.4%}")
        print(f"annual turnover        : {annual_turnover:.1%} one-way")
        print(f"annual cost drag       : {annual_cost_drag:.3%} of NAV")
        print(f"gross ann. return est. : {m['annualized_return'] + annual_cost_drag:+.2%}")
        print(f"net ann. return        : {m['annualized_return']:+.2%}")

    if args.save_tearsheet:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            equity = (1 + port_returns).cumprod()
            dd = equity / equity.cummax() - 1
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            axes[0].plot(equity.index, equity.values)
            axes[0].set_title("QuNtra validation — equity curve (net of costs)")
            axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5)
            axes[1].set_title("Drawdown")
            fig.tight_layout()
            png = OUT.parent / "validation_tearsheet.png"
            OUT.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=120)
            print(f"Tearsheet -> {png.relative_to(ROOT)}")
        except Exception as e:  # noqa: BLE001
            print(f"Tearsheet skipped: {e}")

    checks = {
        "sharpe_ratio": m["sharpe_ratio"] > TARGETS["sharpe_ratio"],
        "max_drawdown": m["max_drawdown"] > TARGETS["max_drawdown"],
        "calmar_ratio": m["calmar_ratio"] > TARGETS["calmar_ratio"],
    }
    all_pass = all(checks.values()) and real_data

    print(f"Data source : {source}")
    print(f"Period      : {port_returns.index[0].date()} -> {port_returns.index[-1].date()}"
          f"  ({len(port_returns)} days)")
    print(f"Turnover    : {annual_turnover:.1%} annualized (one-way)")
    print("-" * 56)
    for key, target in TARGETS.items():
        status = "PASS" if checks[key] else "FAIL"
        if not real_data:
            status = "N/A (synthetic)"
        cmp = ">" if key != "max_drawdown" else ">"
        print(f"{key:14s} {m[key]:+.4f}   target {cmp} {target:+.2f}   {status}")
    print("-" * 56)
    if not real_data:
        print("SYNTHETIC RUN — mechanics verified, targets NOT evaluated.")
        print("Populate data/cache first: python3 scripts/fetch_data_cache.py")
    elif all_pass:
        print("ALL 3 TARGETS PASS — Phase 0 validation complete.")
    else:
        print("VALIDATION FAILED — do NOT proceed to Phase 1 gates.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "is_real_data": real_data,
        "n_days": len(port_returns),
        "annual_turnover": annual_turnover,
        "metrics": m,
        "targets": TARGETS,
        "checks": checks,
        "all_pass": bool(all_pass),
    }, indent=2))
    print(f"Saved -> {OUT.relative_to(ROOT)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
