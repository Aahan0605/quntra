"""Market regime classification — was a documented gap, not a config issue.

system_state["regime"] is read in five places (research_writer's daily
report, /regime, daily/weekly/monthly reports, trade tagging) but was
written NOWHERE — nothing ever set it. The pre-market report has said
"MARKET REGIME: UNKNOWN" since the system existed; `/regime`'s own
docstring says "regime defaults to UNKNOWN" pending "a Phase 3 HMM refit."

This is NOT that HMM. A hidden-Markov regime model needs real training
data and validation this project doesn't have time to build honestly right
now — and an unvalidated model would just be a differently-shaped version
of the same overfitting problem as the 194-ticker ML gate
(src/ml/multiple_testing.py). Instead: a small set of transparent rules
over signals `src/risk/crash_risk.py` already computes and has validated
against 2008/2020. Reusing that module's `compute_signals()` rather than
recomputing vol/trend/drawdown a second time.

Six states, matching the taxonomy already wired into
telegram_bot.py:cmd_regime's emoji map (BULL_TRENDING, BULL_VOLATILE,
SIDEWAYS, BEAR_TRENDING, BEAR_VOLATILE, CRISIS) — filling a gap in an
existing design, not introducing a new one.
"""

from __future__ import annotations

import pandas as pd

from src.risk.crash_risk import CRISIS, band, compute_signals, score_row

VOL_ELEVATED = 1.5      # vol_spike above this counts as "volatile"
SIDEWAYS_BAND = 0.02    # |20d return| below this counts as no clear trend


def classify(row: pd.Series) -> str:
    """One row of crash_risk.compute_signals() -> a regime label.

    CRISIS takes priority over everything else — a crisis is a crisis
    regardless of which side of the 200dma price sits on.
    """
    if pd.isna(row.get("score", float("nan"))):
        return "UNKNOWN"
    if band(row["score"])[0] == CRISIS:
        return "CRISIS"

    above_trend = not bool(row["trend_break"])
    volatile = row["vol_spike"] > VOL_ELEVATED if pd.notna(row["vol_spike"]) else False
    sideways = (pd.notna(row["dd_velocity"])
               and abs(row["dd_velocity"]) < SIDEWAYS_BAND)

    if sideways:
        return "SIDEWAYS"
    if above_trend:
        return "BULL_VOLATILE" if volatile else "BULL_TRENDING"
    return "BEAR_VOLATILE" if volatile else "BEAR_TRENDING"


def current_regime(close: pd.Series) -> dict:
    """Today's regime + a short history for /regime's trend display."""
    sig = compute_signals(close)
    sig["score"] = sig.apply(score_row, axis=1)
    sig["state"] = sig.apply(classify, axis=1)
    tail = sig.tail(6)
    history = [{"date": str(d.date()), "regime": r}
              for d, r in tail["state"].items()][:-1]
    return {
        "state": sig["state"].iloc[-1],
        "confidence": 0.6,   # rule-based, not probabilistic — a fixed,
                            # honest mid-confidence rather than a
                            # manufactured number an HMM would produce
        "history": history,
        "as_of": str(sig.index[-1].date()),
    }
