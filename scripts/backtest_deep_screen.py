#!/usr/bin/env python3
"""Backtest the nightly deep screen — does picking its top names beat
buy-and-hold, or is it the signal-council result all over again?

Replays the REAL scoring function (src.research.deep_screen.score_ticker,
imported, not reimplemented) on point-in-time slices: on each rebalance
date only close[:t] is passed in, so no bar after the decision can leak
into it. Exits use PaperTrader's real rules (-2% stop / +4% target /
5-day time stop) and the real ICICI cost model.

Compared against two honest baselines over the identical window:
  1. buy-and-hold NIFTY
  2. the validated inverse-vol allocator's universe, equal-weighted

A screen that can't beat both isn't worth trading, however good the
individual picks look.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.deep_screen import (MAX_PER_SECTOR, STOP_PCT,  # noqa: E402
                                      TARGET_PCT, MAX_HOLD_DAYS,
                                      score_ticker)
from src.utils.cache_loader import load_ticker  # noqa: E402
from src.utils.costs import CostModel  # noqa: E402
from src.utils.universe_nifty200 import (NIFTY200,  # noqa: E402
                                         NIFTY200_SECTOR_MAP)

WARMUP = 252          # score_ticker needs a full year of history
REBALANCE_EVERY = 5   # trading days — matches the 5-day max hold
TOP_N = 10            # positions held at once; 50 names on 25k is ~500/name
CAPITAL = 25_000.0


def build_panel() -> pd.DataFrame:
    frames = {}
    for t in NIFTY200:
        try:
            s = load_ticker(t)["close"]
        except FileNotFoundError:
            continue
        if int(s.notna().sum()) >= 1000:
            frames[t] = s
    return pd.DataFrame(frames).ffill()


def pick_at(panel: pd.DataFrame, i: int, top_n: int) -> list[str]:
    """Top names as of bar i, using ONLY data up to and including i."""
    rows = []
    for t in panel.columns:
        hist = panel[t].iloc[:i + 1]
        sc = score_ticker(hist)
        if sc is None:
            continue
        rows.append((sc["score"], t))
    rows.sort(reverse=True)

    picked, per_sector = [], {}
    for _, t in rows:
        sec = NIFTY200_SECTOR_MAP.get(t, "UNKNOWN")
        if per_sector.get(sec, 0) >= MAX_PER_SECTOR:
            continue
        per_sector[sec] = per_sector.get(sec, 0) + 1
        picked.append(t)
        if len(picked) >= top_n:
            break
    return picked


def simulate(panel: pd.DataFrame, top_n: int = TOP_N) -> dict:
    cost = CostModel.from_config()
    cash = CAPITAL
    positions: dict[str, dict] = {}
    equity, trades = [], []

    for i in range(WARMUP, len(panel)):
        date = panel.index[i]

        # 1. Manage open positions first (same order as the live engine).
        for t in list(positions):
            pos = positions[t]
            px = panel[t].iloc[i]
            if pd.isna(px):
                continue
            ret = px / pos["entry"] - 1
            age = i - pos["entry_i"]
            reason = None
            if ret <= STOP_PCT:
                reason = "STOP"
            elif ret >= TARGET_PCT:
                reason = "TARGET"
            elif age >= MAX_HOLD_DAYS:
                reason = "TIME"
            if reason:
                notional = px * pos["qty"]
                fees = cost.cost_for_trade(notional, delivery=True)
                cash += notional - fees
                trades.append({"ticker": t, "reason": reason,
                               "pnl": (px - pos["entry"]) * pos["qty"]
                                      - fees - pos["fees"]})
                del positions[t]

        # 2. Rebalance into fresh picks on schedule.
        if (i - WARMUP) % REBALANCE_EVERY == 0 and len(positions) < top_n:
            for t in pick_at(panel, i, top_n):
                if t in positions or len(positions) >= top_n:
                    continue
                px = panel[t].iloc[i]
                if pd.isna(px) or px <= 0:
                    continue
                budget = CAPITAL / top_n
                qty = int(budget / px)
                if qty < 1 or qty * px > cash:
                    continue
                notional = qty * px
                fees = cost.cost_for_trade(notional, delivery=True)
                cash -= notional + fees
                positions[t] = {"entry": px, "qty": qty, "entry_i": i,
                                "fees": fees}

        mtm = sum(p["qty"] * panel[t].iloc[i] for t, p in positions.items()
                  if not pd.isna(panel[t].iloc[i]))
        equity.append({"date": date, "equity": cash + mtm})

    eq = pd.DataFrame(equity).set_index("date")["equity"]
    return _metrics(eq, trades)


def _metrics(eq: pd.Series, trades: list) -> dict:
    daily = eq.pct_change().dropna()
    years = len(eq) / 252
    total = eq.iloc[-1] / CAPITAL - 1
    vol = daily.std() * np.sqrt(252)
    wins = [t for t in trades if t["pnl"] > 0]
    return {
        "n_days": len(eq),
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 4) if trades else None,
        "total_return_pct": round(100 * total, 2),
        "cagr_pct": round(100 * ((eq.iloc[-1] / CAPITAL) ** (1 / years) - 1), 2),
        "sharpe": round(float((daily.mean() * 252) / vol), 4) if vol > 0 else 0.0,
        "max_drawdown_pct": round(100 * float((eq / eq.cummax() - 1).min()), 2),
        "final_equity": round(float(eq.iloc[-1]), 2),
    }


def baseline_buy_hold(series: pd.Series) -> dict:
    s = series.dropna()
    eq = CAPITAL * (s / s.iloc[0])
    return _metrics(eq, [])


def baseline_equal_weight(panel: pd.DataFrame) -> dict:
    """Equal-weight the same investable panel, rebalanced never — the
    'just own everything' control the screen has to beat to justify itself."""
    sub = panel.iloc[WARMUP:].dropna(axis=1, how="any")
    norm = sub / sub.iloc[0]
    eq = CAPITAL * norm.mean(axis=1)
    return _metrics(eq, [])


def main() -> int:
    panel = build_panel()
    print(f"panel: {panel.shape[1]} tickers x {panel.shape[0]} days "
          f"({panel.index[0].date()} -> {panel.index[-1].date()})")
    print(f"scoring point-in-time, rebalance every {REBALANCE_EVERY}d, "
          f"top {TOP_N}, warmup {WARMUP}d…")

    screen = simulate(panel)

    bench = None
    try:
        b = pd.read_csv(ROOT / "data" / "cache" / "NIFTY50_BENCH.csv",
                        index_col=0, parse_dates=True)
        col = "Close" if "Close" in b.columns else "close"
        bench = baseline_buy_hold(b[col].reindex(panel.index).ffill()
                                  .iloc[WARMUP:])
    except Exception as e:  # noqa: BLE001
        print("benchmark unavailable:", e)

    eqw = baseline_equal_weight(panel)

    out = {"deep_screen": screen, "buy_hold_nifty": bench,
           "equal_weight_panel": eqw,
           "config": {"top_n": TOP_N, "rebalance_every": REBALANCE_EVERY,
                      "stop_pct": STOP_PCT, "target_pct": TARGET_PCT,
                      "max_hold_days": MAX_HOLD_DAYS}}

    print("\n" + "=" * 66)
    print("DEEP SCREEN BACKTEST (point-in-time, real exits, real costs)")
    print("=" * 66)
    hdr = f"{'':24s} {'return':>10s} {'CAGR':>8s} {'Sharpe':>8s} {'maxDD':>9s}"
    print(hdr)
    for name, m in (("deep screen", screen), ("buy-and-hold NIFTY", bench),
                    ("equal-weight panel", eqw)):
        if not m:
            continue
        print(f"{name:24s} {m['total_return_pct']:>9.2f}% "
              f"{m['cagr_pct']:>7.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_drawdown_pct']:>8.2f}%")
    print("-" * 66)
    print(f"screen trades: {screen['n_trades']} | win rate: "
          f"{screen['win_rate']}")

    dest = ROOT / "data" / "backtest_results" / "deep_screen_backtest.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwritten -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
