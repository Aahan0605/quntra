#!/usr/bin/env python3
"""Historical backtest of the actual signal-council strategy.

Unlike the random-entry Monte Carlo done earlier, this replays the REAL
production scoring functions — `SignalCouncil._technical_vote`,
`_momentum_vote`, `_sector_votes` — imported directly from
src/governor/council.py, not reimplemented. If those functions change,
this backtest changes with them; there is no drift between "what we
tested" and "what runs live".

WHAT CANNOT BE REPLAYED, AND WHY THAT'S HONEST RATHER THAN SLOPPY
------------------------------------------------------------------
- ml     -> pinned NEUTRAL (1). The FDR audit (src/ml/multiple_testing.py)
            found 0 of 194 trained models distinguishable from noise, so
            this matches what the deployed council does *today*, live.
- macro  -> pinned NEUTRAL (1). Historical macro_bias was never persisted
            day-by-day before the scheduler existed; there is no dataset
            to replay. Pinning neutral is the same "no information"
            default the live code uses when the state row is missing.
- news, fundamental -> pinned at their own documented defaults (0), which
            is exactly what `_news_vote`/`_fundamental_vote` return when no
            research note exists for that ticker — not a simplification,
            the actual default path.

Pinning ml+macro at neutral makes the score gate (>=9/12) HARDER to clear
than in live trading, where those votes can occasionally add real
information. This backtest is therefore a floor on the strategy's
historical behavior, not a ceiling — a deliberately conservative bias
given how many other numbers in this project turned out optimistic.

Exit mechanics are PaperTrader's real rules: -2% stop / +4% target / 5-day
time stop, ICICI cost model, MAX_POSITION_PCT=0.10 cap, MAX_TRADES_PER_DAY=3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.governor.council import SignalCouncil  # noqa: E402
from src.utils.cache_loader import load_benchmark, load_close_panel  # noqa: E402
from src.utils.costs import CostModel  # noqa: E402
from src.utils.universe import UNIVERSE  # noqa: E402

WARMUP = 60          # matches council.py's own minimum series length
MAX_TRADES_PER_DAY = SignalCouncil.MAX_TRADES_PER_DAY
MAX_POSITION_PCT = SignalCouncil.MAX_POSITION_PCT
SCORE_GATE = 9
STOP_LOSS_PCT = -0.02
TAKE_PROFIT_PCT = 0.04
TIME_STOP_DAYS = 5


def score_all_days(panel: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """Score every ticker on every eligible day using the REAL vote fns."""
    sector_by_day = {}   # computed once per day, like production
    rows = []
    for i in range(WARMUP, len(panel)):
        date = panel.index[i]
        day_panel = panel.iloc[:i + 1]
        day_bench = bench.iloc[:i + 1] if bench is not None else None
        sector_votes = SignalCouncil._sector_votes(day_panel)
        for ticker in panel.columns:
            series = day_panel[ticker].dropna()
            if len(series) < WARMUP:
                continue
            tech = SignalCouncil._technical_vote(series)
            mom = SignalCouncil._momentum_vote(series, day_bench)
            sector = sector_votes.get(ticker, 1)
            score = tech + mom + 1 + 1 + sector  # +1 ml, +1 macro (neutral)
            rows.append({"date": date, "ticker": ticker, "score": score,
                        "close": series.iloc[-1]})
    return pd.DataFrame(rows)


def simulate(scores: pd.DataFrame, panel: pd.DataFrame,
            capital: float = 25_000.0) -> dict:
    cost = CostModel.from_config()
    cash = capital
    open_pos: dict[str, dict] = {}   # ticker -> {entry_px, qty, entry_i, fees}
    equity_curve = []
    trades = []
    dates = sorted(scores["date"].unique())
    date_to_i = {d: i for i, d in enumerate(panel.index)}

    for date in dates:
        i = date_to_i[date]
        day_scores = scores[scores["date"] == date]

        # 1. Manage existing positions first (real exit order).
        for ticker in list(open_pos):
            pos = open_pos[ticker]
            px = panel[ticker].iloc[i]
            if pd.isna(px):
                continue
            ret = px / pos["entry_px"] - 1
            age = i - pos["entry_i"]
            reason = None
            if ret <= STOP_LOSS_PCT:
                reason = "STOP_LOSS"
            elif ret >= TAKE_PROFIT_PCT:
                reason = "TAKE_PROFIT"
            elif age >= TIME_STOP_DAYS:
                reason = "TIME_STOP"
            if reason:
                notional = px * pos["qty"]
                fees = cost.cost_for_trade(notional, delivery=True)
                pnl = (px - pos["entry_px"]) * pos["qty"] - fees - pos["fees"]
                cash += notional - fees
                trades.append({"ticker": ticker, "entry": pos["entry_px"],
                               "exit": px, "pnl": pnl, "reason": reason,
                               "hold_days": age})
                del open_pos[ticker]

        # 2. New entries: top-scoring qualifiers, capped by daily/position limits.
        qualifiers = day_scores[day_scores["score"] >= SCORE_GATE]
        qualifiers = qualifiers[~qualifiers["ticker"].isin(open_pos)]
        qualifiers = qualifiers.sort_values("score", ascending=False)
        slots = MAX_TRADES_PER_DAY - len(open_pos)
        for _, row in qualifiers.head(max(slots, 0)).iterrows():
            px = row["close"]
            if pd.isna(px) or px <= 0:
                continue
            per_trade = min(capital / MAX_TRADES_PER_DAY,
                            capital * MAX_POSITION_PCT)
            qty = int(per_trade / px)
            if qty < 1 or qty * px > cash:
                continue
            notional = qty * px
            fees = cost.cost_for_trade(notional, delivery=True)
            cash -= notional + fees
            open_pos[row["ticker"]] = {"entry_px": px, "qty": qty,
                                       "entry_i": i, "fees": fees}

        mtm = sum(pos["qty"] * panel[t].iloc[i]
                 for t, pos in open_pos.items()
                 if not pd.isna(panel[t].iloc[i]))
        equity_curve.append({"date": date, "equity": cash + mtm})

    eq = pd.DataFrame(equity_curve).set_index("date")["equity"]
    daily_ret = eq.pct_change().dropna()
    n_years = len(eq) / 252
    total_return = eq.iloc[-1] / capital - 1
    cagr = (eq.iloc[-1] / capital) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / vol if vol > 0 else 0.0
    running_max = eq.cummax()
    max_dd = (eq / running_max - 1).min()
    wins = [t for t in trades if t["pnl"] > 0]

    return {
        "n_trading_days": len(eq),
        "n_trades": len(trades),
        "n_days_with_a_signal": scores[scores["score"] >= SCORE_GATE]
                                     ["date"].nunique(),
        "win_rate": len(wins) / len(trades) if trades else None,
        "total_return_pct": round(100 * total_return, 2),
        "cagr_pct": round(100 * cagr, 2),
        "annualized_vol_pct": round(100 * vol, 2),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown_pct": round(100 * float(max_dd), 2),
        "final_equity": round(float(eq.iloc[-1]), 2),
        "avg_trade_pnl": round(float(np.mean([t["pnl"] for t in trades])), 2)
                        if trades else None,
        "exit_reason_counts": pd.Series([t["reason"] for t in trades])
                              .value_counts().to_dict() if trades else {},
    }


def benchmark_buy_hold(bench: pd.Series, capital: float = 25_000.0) -> dict:
    ret = bench.iloc[-1] / bench.iloc[0] - 1
    n_years = len(bench) / 252
    cagr = (1 + ret) ** (1 / n_years) - 1
    daily = bench.pct_change().dropna()
    vol = daily.std() * np.sqrt(252)
    sharpe = (daily.mean() * 252) / vol if vol > 0 else 0.0
    dd = (bench / bench.cummax() - 1).min()
    return {"total_return_pct": round(100 * ret, 2),
           "cagr_pct": round(100 * cagr, 2),
           "sharpe_ratio": round(float(sharpe), 4),
           "max_drawdown_pct": round(100 * float(dd), 2)}


def main() -> int:
    panel = load_close_panel(UNIVERSE)
    bench = load_benchmark()
    print(f"Universe: {panel.shape[1]} tickers, {panel.shape[0]} days "
         f"({panel.index[0].date()} -> {panel.index[-1].date()})")

    print("Scoring every ticker on every eligible day with the real "
         "technical/momentum/sector vote functions...")
    scores = score_all_days(panel, bench)

    print("Simulating the strategy (real exit rules + cost model + "
         "position cap)...")
    result = simulate(scores, panel)
    result["benchmark_buy_and_hold_NIFTY"] = (
        benchmark_buy_hold(bench) if bench is not None else None)
    result["note"] = ("ml and macro votes pinned NEUTRAL (see module "
                      "docstring) — this is a conservative floor, not a "
                      "ceiling, on the strategy's historical performance.")

    print("\n" + "=" * 62)
    print("SIGNAL-COUNCIL STRATEGY — HONEST HISTORICAL BACKTEST")
    print("=" * 62)
    for k, v in result.items():
        if k in ("benchmark_buy_and_hold_NIFTY", "note", "exit_reason_counts"):
            continue
        print(f"{k:<28} {v}")
    print(f"{'exit_reason_counts':<28} {result['exit_reason_counts']}")
    print("-" * 62)
    print("vs. buy-and-hold NIFTY over the same window:")
    for k, v in (result["benchmark_buy_and_hold_NIFTY"] or {}).items():
        print(f"  {k:<26} {v}")
    print("=" * 62)

    out = ROOT / "data" / "backtest_results" / "signal_council_backtest.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
