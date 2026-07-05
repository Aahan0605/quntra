"""Tests for the weekly drift-threshold rebalancer (Task P0-3)."""
import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.portfolio.rebalancer import Rebalancer
from src.utils.universe import UNIVERSE


def test_weekly_frequency_gate():
    rb = Rebalancer(frequency="weekly")
    cur = {"A": 0.5, "B": 0.5}
    tgt = {"A": 0.3, "B": 0.7}
    # Monday: rebalances
    d1 = rb.compute_trades(cur, tgt, date(2026, 6, 1))
    assert d1.should_rebalance
    # Wednesday same week: blocked
    d2 = rb.compute_trades(cur, tgt, date(2026, 6, 3))
    assert not d2.should_rebalance
    # Next Monday: allowed again
    d3 = rb.compute_trades(cur, tgt, date(2026, 6, 8))
    assert d3.should_rebalance


def test_drift_threshold_skips_small_deviations():
    rb = Rebalancer()
    cur = {"A": 0.50, "B": 0.29, "C": 0.21}
    tgt = {"A": 0.48, "B": 0.31, "C": 0.21}  # all drifts <= 3%
    d = rb.compute_trades(cur, tgt, date(2026, 6, 1))
    assert not d.should_rebalance
    assert d.reason == "all drifts within threshold"


def test_drift_threshold_trades_large_deviations_only():
    rb = Rebalancer()
    cur = {"A": 0.50, "B": 0.30, "C": 0.20}
    tgt = {"A": 0.40, "B": 0.32, "C": 0.28}  # A: -10%, B: +2%, C: +8%
    d = rb.compute_trades(cur, tgt, date(2026, 6, 1))
    assert d.should_rebalance
    assert "A" in d.trades and "C" in d.trades
    assert "B" not in d.trades  # within threshold


def test_turnover_cap_scales_trades():
    rb = Rebalancer(max_turnover=0.10)
    cur = {"A": 0.90, "B": 0.10}
    tgt = {"A": 0.10, "B": 0.90}  # one-way turnover would be 0.80
    d = rb.compute_trades(cur, tgt, date(2026, 6, 1))
    assert d.should_rebalance
    assert d.one_way_turnover == pytest.approx(0.10)
    assert sum(abs(v) for v in d.trades.values()) * 0.5 == pytest.approx(0.10)


def test_annual_turnover_under_300pct():
    """Acceptance criterion: annual turnover < 300% in simulation."""
    rng = np.random.default_rng(42)
    n_days = 504  # ~2 years
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    # NSE-like daily vol ~1.8%
    rets = pd.DataFrame(
        rng.normal(0.0005, 0.018, size=(n_days, len(UNIVERSE))),
        index=idx, columns=UNIVERSE,
    )
    target = {t: 1.0 / len(UNIVERSE) for t in UNIVERSE}
    rb = Rebalancer()
    annual_turnover = rb.simulate_annual_turnover(rets, target)
    assert annual_turnover < 3.0, f"Annual turnover {annual_turnover:.1%} >= 300%"


def test_daily_rebalancer_produces_more_turnover_than_weekly():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-01", periods=252)
    rets = pd.DataFrame(
        rng.normal(0, 0.02, size=(252, 5)),
        index=idx, columns=list("ABCDE"),
    )
    target = {t: 0.2 for t in "ABCDE"}
    weekly = Rebalancer(frequency="weekly").simulate_annual_turnover(rets, target)
    daily = Rebalancer(frequency="daily", drift_threshold=0.0).simulate_annual_turnover(rets, target)
    assert daily > weekly
