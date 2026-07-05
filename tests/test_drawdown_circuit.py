"""Tests for the recalibrated 3-level circuit breaker (Task P0-4)."""
from datetime import datetime, timedelta

from src.risk.drawdown_circuit import (
    CircuitLevel,
    DrawdownCircuitBreaker,
)

NOW = datetime(2026, 7, 6, 10, 0)


def make_cb():
    return DrawdownCircuitBreaker()


def test_normal_below_thresholds():
    cb = make_cb()
    s = cb.update(intraday_dd=-0.02, rolling_5d_dd=-0.03, now=NOW)
    assert s.level == CircuitLevel.NORMAL
    assert cb.can_enter_new_position(NOW)


def test_level1_at_minus_3pct():
    cb = make_cb()
    s = cb.update(intraday_dd=-0.031, rolling_5d_dd=-0.031, now=NOW)
    assert s.level == CircuitLevel.LEVEL1
    assert s.tighten_stops
    assert not s.square_all
    # In cooldown: no new entries
    assert not cb.can_enter_new_position(NOW + timedelta(minutes=10))
    # After 30-min cooldown: cautious re-entry allowed
    assert cb.can_enter_new_position(NOW + timedelta(minutes=31))


def test_level2_at_minus_4_5pct():
    cb = make_cb()
    s = cb.update(intraday_dd=-0.046, rolling_5d_dd=-0.046, now=NOW)
    assert s.level == CircuitLevel.LEVEL2
    assert s.square_all
    assert s.oms_disabled
    assert not cb.can_enter_new_position(NOW + timedelta(hours=2))


def test_level3_at_minus_7pct_rolling():
    cb = make_cb()
    s = cb.update(intraday_dd=-0.01, rolling_5d_dd=-0.072, now=NOW)
    assert s.level == CircuitLevel.LEVEL3
    assert s.manual_resume_required
    assert not cb.can_enter_new_position(NOW + timedelta(days=1))
    # Daily reset does NOT clear Level 3
    cb.reset_daily()
    assert cb.state.manual_resume_required
    # Only manual resume clears it
    cb.manual_resume()
    assert cb.state.level == CircuitLevel.NORMAL
    assert cb.can_enter_new_position(NOW)


def test_no_deescalation_within_session():
    cb = make_cb()
    cb.update(intraday_dd=-0.046, rolling_5d_dd=-0.046, now=NOW)
    s = cb.update(intraday_dd=-0.01, rolling_5d_dd=-0.01, now=NOW + timedelta(hours=1))
    assert s.level == CircuitLevel.LEVEL2  # recovery doesn't re-enable same day


def test_consecutive_loss_halt():
    cb = make_cb()
    cb.record_trade_result(-100, NOW)
    cb.record_trade_result(-50, NOW)
    assert cb.can_enter_new_position(NOW)
    cb.record_trade_result(-10, NOW)  # third consecutive loss
    assert cb.state.halted_by_losses
    assert not cb.can_enter_new_position(NOW)


def test_win_resets_loss_counter():
    cb = make_cb()
    cb.record_trade_result(-100, NOW)
    cb.record_trade_result(-50, NOW)
    cb.record_trade_result(+200, NOW)
    assert cb.state.consecutive_losses == 0
    cb.record_trade_result(-10, NOW)
    assert not cb.state.halted_by_losses


def test_daily_reset_clears_level1():
    cb = make_cb()
    cb.update(intraday_dd=-0.031, rolling_5d_dd=-0.031, now=NOW)
    cb.reset_daily()
    assert cb.state.level == CircuitLevel.NORMAL
    assert cb.can_enter_new_position(NOW)


def test_invalid_thresholds_rejected():
    import pytest
    with pytest.raises(ValueError):
        DrawdownCircuitBreaker(level1_dd=-0.05, level2_dd=-0.03, level3_dd=-0.07)
