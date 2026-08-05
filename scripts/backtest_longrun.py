#!/usr/bin/env python3
"""What actually compounds over 18 years? Tests the levers that matter for
LONG-RUN wealth rather than short-term edge.

Uses NIFTY 2007-2026 (18.4 yrs) because it is the only series long enough
to contain real crashes — 2008 (-60%) and 2020 (-38%). The per-stock cache
only starts 2021 and contains no crash, so any "long run" claim built on it
is untested against the exact events that destroy long-run compounding.

Compares, all on identical data with real costs:
  1. buy & hold                      — the baseline to beat
  2. crash-risk overlay              — cut exposure when src/risk/crash_risk
                                       says CRISIS/HIGH (already built+wired)
  3. volatility targeting            — scale exposure to a constant vol
  4. overlay + vol targeting         — both

Cash earns CASH_RATE while de-risked. Set to 0 by default: if a defensive
overlay wins even when sitting in a zero-yield mattress, the result is not
an artifact of assuming generous cash returns.

The metric that matters here is NOT total return. It is risk-adjusted
compounding and max drawdown, because a -50% drawdown needs +100% just to
get back to even — losses compound against you asymmetrically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk.crash_risk import band, compute_signals, score_row  # noqa: E402
from src.utils.costs import CostModel  # noqa: E402

CASH_RATE = 0.0          # annual; 0 = most conservative for the overlay
VOL_TARGET = 0.15        # 15% annualised — a common institutional target
MAX_LEVERAGE = 1.0       # never lever up; this is a de-risking study


def metrics(eq: pd.Series, label: str, exposure: pd.Series | None = None) -> dict:
    r = eq.pct_change().dropna()
    yrs = len(eq) / 252
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd_series = eq / eq.cummax() - 1
    mdd = float(dd_series.min())
    # Worst 1-year rolling return — "does it bleed for a whole year?"
    roll = eq.pct_change(252).dropna()
    return {
        "strategy": label,
        "total_return_pct": round(100 * total, 2),
        "cagr_pct": round(100 * cagr, 2),
        "sharpe": round(float((r.mean() * 252) / vol), 3) if vol > 0 else 0.0,
        "max_drawdown_pct": round(100 * mdd, 2),
        "calmar": round(float(cagr / abs(mdd)), 3) if mdd else None,
        "worst_1y_pct": round(100 * float(roll.min()), 2),
        "pct_time_invested": (round(100 * float(exposure.mean()), 1)
                              if exposure is not None else 100.0),
        "years": round(yrs, 1),
    }


def run(close: pd.Series) -> list[dict]:
    cost = CostModel.from_config()
    friction = cost.one_way_cost_rate(delivery=True) + cost.slippage_rate()
    rets = close.pct_change().fillna(0.0)
    daily_cash = (1 + CASH_RATE) ** (1 / 252) - 1

    # Crash-risk exposure, computed point-in-time by construction:
    # compute_signals uses only trailing windows.
    sig = compute_signals(close)
    sig["score"] = sig.apply(score_row, axis=1)
    crash_exp = sig["score"].map(lambda s: band(s)[1]).fillna(1.0)

    # Vol-target exposure from TRAILING realised vol only.
    trail_vol = rets.rolling(20).std() * np.sqrt(252)
    vt_exp = (VOL_TARGET / trail_vol).clip(upper=MAX_LEVERAGE).fillna(1.0)

    def simulate(exposure: pd.Series, label: str) -> dict:
        # Lag by one day: today's signal can only be acted on tomorrow.
        e = exposure.shift(1).fillna(1.0).clip(0.0, MAX_LEVERAGE)
        turnover = e.diff().abs().fillna(0.0)
        strat = e * rets + (1 - e) * daily_cash - turnover * friction
        eq = (1 + strat).cumprod()
        return metrics(eq, label, exposure=e)

    ones = pd.Series(1.0, index=close.index)
    return [
        simulate(ones, "1. buy & hold"),
        simulate(crash_exp, "2. crash-risk overlay"),
        simulate(vt_exp, "3. volatility targeting"),
        simulate((crash_exp * vt_exp).clip(0, MAX_LEVERAGE),
                 "4. overlay + vol target"),
    ]


def main() -> int:
    df = pd.read_csv(ROOT / "data" / "cache" / "NIFTY_LONG.csv",
                     index_col=0, parse_dates=True)
    close = df["Close"].dropna()
    print(f"NIFTY {close.index[0].date()} -> {close.index[-1].date()} "
          f"({len(close)/252:.1f} yrs, includes 2008 and 2020 crashes)")
    print(f"cash rate while de-risked: {CASH_RATE:.0%} (conservative) | "
          f"vol target {VOL_TARGET:.0%}\n")

    rows = run(close)
    hdr = (f"{'strategy':26s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>8s} "
           f"{'Calmar':>7s} {'worst1y':>8s} {'invested':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        print(f"{m['strategy']:26s} {m['cagr_pct']:>6.2f}% {m['sharpe']:>7.3f} "
              f"{m['max_drawdown_pct']:>7.2f}% {m['calmar']:>7.3f} "
              f"{m['worst_1y_pct']:>7.2f}% {m['pct_time_invested']:>8.1f}%")

    dest = ROOT / "data" / "backtest_results" / "longrun_backtest.json"
    dest.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
