"""The crash indicator must fire on real crashes and stay quiet otherwise.

Validated against actual Nifty history (data/cache/NIFTY_LONG.csv, 2007-2026),
not synthetic data — a crash detector tested only on hand-made series proves
nothing about crashes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.risk.crash_risk import CRISIS, assess, band, crash_risk_series, score_row

LONG_HISTORY = Path("data/cache/NIFTY_LONG.csv")
needs_history = pytest.mark.skipif(
    not LONG_HISTORY.exists(),
    reason="run: yfinance download of ^NSEI -> data/cache/NIFTY_LONG.csv")


@pytest.fixture(scope="module")
def nifty():
    df = pd.read_csv(LONG_HISTORY, index_col=0, parse_dates=True)
    return crash_risk_series(df["Close"])


# ------------------------------------------------------------------ #
# Behaviour on real crashes

@needs_history
@pytest.mark.parametrize("label,start,end", [
    ("2008 GFC", "2008-01-01", "2009-03-31"),
    ("2020 COVID", "2020-02-01", "2020-04-30"),
])
def test_reaches_crisis_during_real_crashes(nifty, label, start, end):
    window = nifty.loc[start:end]
    assert (window["regime"] == CRISIS).sum() >= 10, f"{label} never hit CRISIS"


@needs_history
@pytest.mark.parametrize("label,start,end", [
    ("2017 calm bull", "2017-01-01", "2017-12-31"),
    ("2021 recovery", "2021-04-01", "2021-12-31"),
])
def test_stays_quiet_in_calm_markets(nifty, label, start, end):
    window = nifty.loc[start:end]
    assert (window["regime"] == CRISIS).sum() == 0, f"{label} false alarm"
    assert window["score"].mean() < 10, f"{label} chronically elevated"


@needs_history
def test_crisis_is_rare(nifty):
    """A detector that always shouts is useless. Crashes are rare; so is CRISIS."""
    rate = (nifty["regime"] == CRISIS).mean()
    assert 0 < rate < 0.05, f"CRISIS fired on {rate:.1%} of all days"


@needs_history
@pytest.mark.parametrize("label,peak,trough,min_avoided", [
    ("2008 GFC", "2008-01-08", "2008-10-27", 0.35),
    ("2020 COVID", "2020-01-14", "2020-03-23", 0.20),
])
def test_derisks_before_the_worst_of_the_fall(nifty, label, peak, trough,
                                              min_avoided):
    """The point is avoiding the tail, not calling the top."""
    fall = nifty.loc[peak:trough]
    derisked = fall[fall["exposure"] <= 0.25]
    assert not derisked.empty, f"{label}: never de-risked"
    signal_px = derisked["close"].iloc[0]
    avoided = 1 - fall["close"].min() / signal_px
    assert avoided >= min_avoided, (
        f"{label}: only avoided {avoided:.1%} of downside after signalling")


# ------------------------------------------------------------------ #
# Scoring mechanics

def test_bands_map_to_falling_exposure():
    assert band(90)[1] == 0.0      # CRISIS -> flat
    assert band(60)[1] == 0.25
    assert band(40)[1] == 0.60
    assert band(5)[1] == 1.0
    scores = [5, 40, 60, 90]
    exposures = [band(s)[1] for s in scores]
    assert exposures == sorted(exposures, reverse=True), "must be monotonic"


def test_calm_row_scores_zero():
    calm = pd.Series({"vol_spike": 1.0, "drawdown": -0.01, "dd_velocity": 0.02,
                      "trend_break": 0.0, "panic_days": 0})
    assert score_row(calm) == 0.0


def test_maximally_bad_row_scores_100():
    worst = pd.Series({"vol_spike": 5.0, "drawdown": -0.50, "dd_velocity": -0.40,
                       "trend_break": 1.0, "panic_days": 10})
    assert score_row(worst) == 100.0


def test_score_is_monotonic_in_drawdown():
    def s(dd):
        return score_row(pd.Series({
            "vol_spike": 1.5, "drawdown": dd, "dd_velocity": -0.05,
            "trend_break": 1.0, "panic_days": 2}))
    assert s(-0.05) < s(-0.15) < s(-0.30)


def test_missing_signals_yield_nan_not_a_false_calm():
    """Early history has no 200dma. That must not read as 'safe'."""
    partial = pd.Series({"vol_spike": np.nan, "drawdown": -0.01,
                         "dd_velocity": 0.0, "trend_break": 0.0,
                         "panic_days": 0})
    assert pd.isna(score_row(partial))


def test_band_treats_nan_as_calm_but_full_exposure_is_explicit():
    """NaN must not crash the caller; documented as CALM/1.0."""
    assert band(float("nan")) == ("CALM", 1.0)


def test_assess_returns_a_usable_verdict():
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(np.linspace(100, 150, 400), index=idx)
    out = assess(close)
    assert set(out) == {"as_of", "score", "regime", "exposure_multiplier",
                        "signals"}
    assert 0 <= out["exposure_multiplier"] <= 1


# ---------------------------------------------------------------- #
# Position sizing: concentration cap + crash-scaled exposure

def test_single_position_capped_at_10pct_of_capital():
    """Three positions used to be 88% of capital in three correlated names."""
    from src.governor.council import SignalCouncil
    capital = 25_000.0
    per_trade = min(capital / SignalCouncil.MAX_TRADES_PER_DAY,
                    capital * SignalCouncil.MAX_POSITION_PCT)
    assert per_trade <= capital * 0.10
    # A full book must leave real cash unspent.
    assert per_trade * SignalCouncil.MAX_TRADES_PER_DAY <= capital * 0.35


def test_exposure_multiplier_shrinks_order_size():
    """The crash band must reduce size, not only block at CRISIS."""
    base_qty = 20
    for score, expected_max in [(5, 20), (40, 12), (60, 5)]:
        _, exposure = band(score)
        assert max(1, int(base_qty * exposure)) <= expected_max
