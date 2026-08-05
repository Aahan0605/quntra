"""Nightly deep screen — picks 50 Nifty-200 names and builds tomorrow's plan.

Runs AFTER the day's trading and post-market analysis (20:00 IST), so it
can use the day's realised results rather than guessing ahead of them.
Output is a concrete plan per stock: entry level, stop, target, and how
long to hold — using the SAME numbers PaperTrader actually enforces
(-2% stop / +4% target / 5-day time stop), because a plan quoting
different levels than the engine would be fiction.

WHAT THIS IS NOT
----------------
This does not place trades and is deliberately not wired into the live
allocator. scripts/backtest_signal_council.py measured per-stock selection
on this exact universe at -0.51% over 5 years vs +52.55% for buy-and-hold
NIFTY, and src/ml/multiple_testing.py found 0 of 194 per-ticker models
survive a Benjamini-Hochberg correction. So this is decision support the
operator reads — not an unvalidated strategy quietly given capital.

The 50 are picked by liquidity + sector spread, then scored on measurable
factors only (trend, momentum, volatility, drawdown, valuation). Every
score component is auditable; nothing here is a black box.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger("quntra.deep_screen")

TOP_N = 50
LOOKBACK = 252
# Mirrors PaperTrader's real exit engine — see module docstring.
STOP_PCT = -0.02
TARGET_PCT = 0.04
MAX_HOLD_DAYS = 5
# No more than this many names from one sector, so a single sector's
# factor tilt can't quietly become the whole shortlist.
MAX_PER_SECTOR = 6


def _safe_last(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def score_ticker(close: pd.Series) -> dict | None:
    """Measurable, auditable factors only. None when history is too short
    to compute them honestly — a short series gets excluded, not guessed."""
    s = close.dropna()
    if len(s) < LOOKBACK:
        return None

    sma20 = s.rolling(20).mean()
    sma50 = s.rolling(50).mean()
    sma200 = s.rolling(200).mean()
    px = float(s.iloc[-1])

    ret_20d = px / float(s.iloc[-21]) - 1
    ret_60d = px / float(s.iloc[-61]) - 1
    vol_20d = float(s.pct_change().tail(20).std() * np.sqrt(252))
    peak_252 = float(s.tail(LOOKBACK).max())
    drawdown = px / peak_252 - 1

    above20 = px > (_safe_last(sma20) or px)
    above50 = px > (_safe_last(sma50) or px)
    above200 = px > (_safe_last(sma200) or px)

    # 0-10, each term independently defensible and inspectable.
    score = 0.0
    score += 2.0 * above200          # long-term trend intact
    score += 1.5 * above50
    score += 1.0 * above20
    score += 2.0 * min(max(ret_60d / 0.15, 0), 1)   # medium-term strength
    score += 1.5 * min(max(ret_20d / 0.08, 0), 1)   # short-term confirmation
    # Lower volatility preferred — the validated strategy is inverse-vol.
    score += 2.0 * min(max((0.35 - vol_20d) / 0.25, 0), 1)

    return {
        "price": round(px, 2),
        "ret_20d_pct": round(100 * ret_20d, 2),
        "ret_60d_pct": round(100 * ret_60d, 2),
        "vol_20d_pct": round(100 * vol_20d, 2),
        "drawdown_pct": round(100 * drawdown, 2),
        "above_200dma": bool(above200),
        "score": round(score, 2),
    }


def plan_for(px: float) -> dict:
    """Entry/stop/target/hold using the engine's real thresholds."""
    return {
        "entry_near": round(px, 2),
        "stop_loss": round(px * (1 + STOP_PCT), 2),
        "target": round(px * (1 + TARGET_PCT), 2),
        "max_hold_days": MAX_HOLD_DAYS,
        "exit_rule": (f"whichever first: stop {STOP_PCT:+.0%}, "
                      f"target {TARGET_PCT:+.0%}, or {MAX_HOLD_DAYS} "
                      f"trading days"),
    }


def run_screen(top_n: int = TOP_N) -> dict:
    """Score the Nifty 200, return the top `top_n` with per-stock plans."""
    from src.utils.cache_loader import load_ticker
    from src.utils.universe_nifty200 import NIFTY200, NIFTY200_SECTOR_MAP

    rows, skipped = [], {"no_cache": 0, "short_history": 0}
    for t in NIFTY200:
        try:
            df = load_ticker(t)
        except FileNotFoundError:
            skipped["no_cache"] += 1
            continue
        sc = score_ticker(df["close"])
        if sc is None:
            skipped["short_history"] += 1
            continue
        sc["ticker"] = t
        sc["sector"] = NIFTY200_SECTOR_MAP.get(t, "UNKNOWN")
        sc["as_of"] = str(df.index[-1].date())
        rows.append(sc)

    rows.sort(key=lambda r: -r["score"])

    # Entry/stop/target are quoted off the last cached bar. If that bar is
    # old the levels are fiction, so surface staleness loudly rather than
    # printing confident-looking prices from two weeks ago.
    today = datetime.now(timezone.utc).date()
    for r in rows:
        age = (today - datetime.strptime(r["as_of"], "%Y-%m-%d").date()).days
        r["data_age_days"] = age
        r["stale"] = age > 3

    # Sector cap, applied after ranking so the best name in a sector still
    # wins its slot but one sector cannot dominate the whole list.
    picked, per_sector = [], {}
    for r in rows:
        sec = r["sector"]
        if per_sector.get(sec, 0) >= MAX_PER_SECTOR:
            continue
        per_sector[sec] = per_sector.get(sec, 0) + 1
        r["plan"] = plan_for(r["price"])
        picked.append(r)
        if len(picked) >= top_n:
            break

    n_stale = sum(1 for r in picked if r["stale"])
    if n_stale:
        logger.warning("deep screen: %d/%d picks priced off cache older than "
                       "3 days — refresh with scripts/fetch_data_cache.py",
                       n_stale, len(picked))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_scored": len(rows),
        "n_picked": len(picked),
        "n_stale": n_stale,
        "skipped": skipped,
        "sector_spread": dict(sorted(per_sector.items(), key=lambda x: -x[1])),
        "picks": picked,
    }


def format_report(res: dict, limit: int = 15) -> str:
    """Telegram-friendly summary. Full detail lands in system_state."""
    lines = [
        f"🔬 DEEP SCREEN — top {res['n_picked']} of {res['n_scored']} scored",
        f"sectors: " + ", ".join(f"{k}:{v}" for k, v in
                                 list(res["sector_spread"].items())[:8]),
    ]
    if res.get("n_stale"):
        lines.append(f"🚨 {res['n_stale']}/{res['n_picked']} priced off STALE "
                     f"cache — levels below are NOT current")
    lines.append("")
    for r in res["picks"][:limit]:
        p = r["plan"]
        flag = f" ⚠️{r['data_age_days']}d-old" if r.get("stale") else ""
        lines.append(
            f"{r['ticker']} ({r['sector']}) score {r['score']}{flag}\n"
            f"  ₹{p['entry_near']} · stop ₹{p['stop_loss']} · "
            f"target ₹{p['target']} · ≤{p['max_hold_days']}d\n"
            f"  20d {r['ret_20d_pct']:+.1f}% · 60d {r['ret_60d_pct']:+.1f}% · "
            f"vol {r['vol_20d_pct']:.0f}% · dd {r['drawdown_pct']:+.1f}%")
    if res["n_picked"] > limit:
        lines.append(f"\n…{res['n_picked'] - limit} more in /deep_screen")
    lines.append("\n⚠️ Decision support only — not auto-traded. Per-stock "
                 "selection backtested worse than buy-and-hold on this "
                 "universe (docs/CEO_REVIEW.md).")
    return "\n".join(lines)
