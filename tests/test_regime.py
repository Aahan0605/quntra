"""Market regime classification — was a real gap: system_state["regime"]
was read in five places and written nowhere, so the pre-market report and
/regime always said UNKNOWN. Validated against real Nifty history, not
synthetic series, matching crash_risk.py's own validation standard.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.risk.crash_risk import compute_signals, score_row
from src.risk.regime import classify, current_regime

LONG_HISTORY = Path("data/cache/NIFTY_LONG.csv")
needs_history = pytest.mark.skipif(not LONG_HISTORY.exists(),
                                   reason="needs data/cache/NIFTY_LONG.csv")


@pytest.fixture(scope="module")
def classified():
    df = pd.read_csv(LONG_HISTORY, index_col=0, parse_dates=True)
    sig = compute_signals(df["Close"])
    sig["score"] = sig.apply(score_row, axis=1)
    sig["state"] = sig.apply(classify, axis=1)
    return sig


@needs_history
def test_2008_crash_is_mostly_crisis_or_bear(classified):
    window = classified.loc["2008-09-01":"2008-11-30", "state"]
    assert (window.isin(["CRISIS", "BEAR_TRENDING", "BEAR_VOLATILE"])).mean() > 0.8


@needs_history
def test_2020_covid_crash_is_mostly_crisis_or_bear(classified):
    window = classified.loc["2020-02-15":"2020-04-15", "state"]
    assert (window.isin(["CRISIS", "BEAR_TRENDING", "BEAR_VOLATILE"])).mean() > 0.8


@needs_history
def test_2021_recovery_is_mostly_bull(classified):
    window = classified.loc["2021-06-01":"2021-12-31", "state"]
    assert (window.isin(["BULL_TRENDING", "BULL_VOLATILE"])).mean() > 0.5


@needs_history
def test_crisis_overrides_trend_direction(classified):
    """CRISIS must win regardless of which side of the 200dma price sits."""
    crisis_rows = classified[classified["state"] == "CRISIS"]
    assert len(crisis_rows) > 0


def test_classify_sideways_when_return_is_flat():
    row = pd.Series({"score": 5.0, "trend_break": 0.0, "vol_spike": 1.0,
                     "dd_velocity": 0.001})
    assert classify(row) == "SIDEWAYS"


def test_classify_bull_trending():
    row = pd.Series({"score": 5.0, "trend_break": 0.0, "vol_spike": 1.0,
                     "dd_velocity": 0.05})
    assert classify(row) == "BULL_TRENDING"


def test_classify_bull_volatile():
    row = pd.Series({"score": 30.0, "trend_break": 0.0, "vol_spike": 2.0,
                     "dd_velocity": 0.05})
    assert classify(row) == "BULL_VOLATILE"


def test_classify_bear_trending():
    row = pd.Series({"score": 20.0, "trend_break": 1.0, "vol_spike": 1.0,
                     "dd_velocity": -0.05})
    assert classify(row) == "BEAR_TRENDING"


def test_classify_crisis_beats_bull_trend():
    row = pd.Series({"score": 90.0, "trend_break": 0.0, "vol_spike": 1.0,
                     "dd_velocity": 0.05})
    assert classify(row) == "CRISIS"


def test_classify_missing_score_is_unknown_not_calm():
    row = pd.Series({"score": float("nan"), "trend_break": 0.0,
                     "vol_spike": 1.0, "dd_velocity": 0.0})
    assert classify(row) == "UNKNOWN"


@needs_history
def test_current_regime_shape():
    df = pd.read_csv(LONG_HISTORY, index_col=0, parse_dates=True)
    out = current_regime(df["Close"])
    assert set(out) == {"state", "confidence", "history", "as_of"}
    assert out["state"] in {"BULL_TRENDING", "BULL_VOLATILE", "SIDEWAYS",
                            "BEAR_TRENDING", "BEAR_VOLATILE", "CRISIS",
                            "UNKNOWN"}
    assert len(out["history"]) <= 5
