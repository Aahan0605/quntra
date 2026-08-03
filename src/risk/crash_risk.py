"""Crash-risk regime indicator for the Indian market.

WHY THIS IS NOT A MACHINE-LEARNING MODEL
----------------------------------------
India has had roughly ten genuine market crashes since 1992 (1992 Harshad
Mehta, 1996-98, 2000 Ketan Parekh/dot-com, 2004 election, 2006 EM selloff,
2008 GFC, 2015 China, 2018 IL&FS, 2020 COVID, 2024 election). Ten positive
examples cannot train a supervised classifier — any model fitted to them
memorises ten dates and generalises to nothing. The honest instrument is a
small set of transparent, individually-defensible rules whose thresholds
are checked against those episodes.

So this does NOT predict crashes. Nothing does. It measures whether the
market is *behaving the way it behaved entering* past crashes, and outputs
a 0-100 risk score used to cut exposure. Getting out of the way is worth a
great deal; calling the top is not required.

THE FIVE SIGNALS (all computable from index OHLCV alone)
--------------------------------------------------------
1. vol_spike   — 20d realised vol vs its own 1y median. Every crash in the
                 sample was preceded or accompanied by a volatility regime
                 change; calm markets do not crash quietly.
2. drawdown    — distance below the trailing 252d high. States the damage
                 already done.
3. dd_velocity — 20d return. Separates a slow grind from 2008/2020-style
                 collapse; velocity, not depth, is what kills leverage.
4. trend_break — close below the 200d moving average. The single most
                 durable regime filter in the literature.
5. panic_days  — count of <-2% days in the last 20 sessions. Crashes cluster
                 their worst days; calm markets almost never stack them.

Each maps to 0-1, then a weighted sum scales to 0-100. Weights are equal-ish
by design: fitting them to ten events would reintroduce exactly the
overfitting this module exists to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Exposure ladder. Deliberately coarse — false precision on ten events is
# self-deception.
CALM, ELEVATED, HIGH, CRISIS = "CALM", "ELEVATED", "HIGH", "CRISIS"

BANDS = [(75, CRISIS, 0.0), (55, HIGH, 0.25), (35, ELEVATED, 0.60),
         (0, CALM, 1.0)]

WEIGHTS = {"vol_spike": 0.25, "drawdown": 0.20, "dd_velocity": 0.25,
           "trend_break": 0.15, "panic_days": 0.15}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def compute_signals(close: pd.Series) -> pd.DataFrame:
    """Per-day raw signals. Uses only data available on that day."""
    ret = close.pct_change()

    vol20 = ret.rolling(20).std() * np.sqrt(252)
    # min_periods=60: a median is robust well before a full year of data.
    # Demanding 252 observations of a 20d vol blinds the indicator for the
    # first ~13 months of any series — which is precisely when a new
    # deployment is most exposed.
    vol_med = vol20.rolling(252, min_periods=60).median()
    vol_spike = vol20 / vol_med

    peak = close.rolling(252, min_periods=60).max()
    drawdown = close / peak - 1.0
    dd_velocity = close.pct_change(20)
    ma200 = close.rolling(200, min_periods=100).mean()
    trend_break = (close < ma200).astype(float)
    panic_days = (ret < -0.02).rolling(20).sum()

    return pd.DataFrame({
        "close": close, "vol_spike": vol_spike, "drawdown": drawdown,
        "dd_velocity": dd_velocity, "trend_break": trend_break,
        "panic_days": panic_days,
    })


def score_row(row: pd.Series) -> float:
    """Map raw signals to 0-100. Each component saturates at 1.0.

    Missing inputs return NaN, never 0. `max(0.0, nan)` is 0.0 in Python,
    so clipping first would silently report an unknown market as perfectly
    calm — the worst possible failure direction for a risk gauge.
    """
    if any(pd.isna(row.get(k)) for k in WEIGHTS):
        return float("nan")
    parts = {
        # 1.0x = normal vol, 2.5x = full alarm
        "vol_spike": _clip01((row["vol_spike"] - 1.0) / 1.5),
        # -5% = nothing, -25% = full alarm
        "drawdown": _clip01((-row["drawdown"] - 0.05) / 0.20),
        # -3% over 20d = nothing, -18% = full alarm
        "dd_velocity": _clip01((-row["dd_velocity"] - 0.03) / 0.15),
        "trend_break": _clip01(row["trend_break"]),
        # 1 panic day = mild, 5 in 20 sessions = full alarm
        "panic_days": _clip01((row["panic_days"] - 1) / 4),
    }
    if any(pd.isna(v) for v in parts.values()):
        return float("nan")
    return round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()), 1)


def band(score: float) -> tuple[str, float]:
    """(regime name, exposure multiplier) for a score."""
    if pd.isna(score):
        return CALM, 1.0
    for threshold, name, exposure in BANDS:
        if score >= threshold:
            return name, exposure
    return CALM, 1.0


def crash_risk_series(close: pd.Series) -> pd.DataFrame:
    """Full history of scores — used to validate against past crashes."""
    sig = compute_signals(close)
    sig["score"] = sig.apply(score_row, axis=1)
    sig["regime"] = sig["score"].map(lambda s: band(s)[0])
    sig["exposure"] = sig["score"].map(lambda s: band(s)[1])
    return sig


def assess(close: pd.Series) -> dict:
    """Today's verdict. `close` must be a date-indexed price series."""
    sig = compute_signals(close)
    row = sig.iloc[-1]
    score = score_row(row)
    name, exposure = band(score)
    return {
        "as_of": str(sig.index[-1])[:10],
        "score": score,
        "regime": name,
        "exposure_multiplier": exposure,
        "signals": {
            "vol_spike": round(float(row["vol_spike"]), 2)
            if pd.notna(row["vol_spike"]) else None,
            "drawdown_pct": round(100 * float(row["drawdown"]), 1)
            if pd.notna(row["drawdown"]) else None,
            "ret_20d_pct": round(100 * float(row["dd_velocity"]), 1)
            if pd.notna(row["dd_velocity"]) else None,
            "below_200dma": bool(row["trend_break"]),
            "panic_days_20d": int(row["panic_days"])
            if pd.notna(row["panic_days"]) else None,
        },
    }
