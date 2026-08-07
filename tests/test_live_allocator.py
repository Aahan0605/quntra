"""The live allocator — the strategy this project pivoted to after
scripts/backtest_signal_council.py showed the stock-picker loses to
buy-and-hold NIFTY over 5 real years (see docs/CEO_REVIEW.md).

Covers: PaperTrader.adjust_position (buy more / sell part / close to zero,
and the unique-signal_hash bug that would otherwise crash a second
rebalance), vetoes (exclude-only, never select), and the allocator's own
weight -> quantity -> trade pipeline.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.execution.paper_trader import PaperTrader
from src.portfolio.live_allocator import PassiveAllocator
from src.portfolio.rebalancer import Rebalancer
from src.portfolio.target_weights import inverse_vol_weights


class FakeBrain:
    def __init__(self):
        self.rows: list[dict] = []

    def remember_trade(self, data):
        h = data.get("signal_hash")
        if any(r.get("signal_hash") == h for r in self.rows):
            raise AssertionError(
                f"duplicate insert for signal_hash={h} — the unique "
                f"constraint this test guards against")
        self.rows.append(dict(data))
        return "id-%d" % len(self.rows)

    def close_trade(self, signal_hash, exit_data):
        for row in self.rows:
            if row.get("signal_hash") == signal_hash and not row.get("exit_time"):
                row.update(exit_data)
                return True
        return False

    def update_position_size(self, signal_hash, quantity, entry_price=None):
        for row in self.rows:
            if row.get("signal_hash") == signal_hash and not row.get("exit_time"):
                row["quantity"] = quantity
                if entry_price is not None:
                    row["entry_price"] = entry_price
                return True
        return False

    def get_open_positions(self, is_paper=True):
        return [r for r in self.rows
                if not r.get("exit_time") and r.get("is_paper") is is_paper]


class FakeFetcher:
    def __init__(self, price=100.0):
        self.price = price

    def get_live_quote(self, tickers):
        return pd.DataFrame([{"ticker": tickers[0], "last_price": self.price}])


# ------------------------------------------------------------------ #
# PaperTrader.adjust_position

def test_adjust_position_opens_fresh_from_zero():
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    out = trader.adjust_position("TCS.NS", 5, 100.0, signal_hash="ALLOC-TCS.NS")
    assert out["quantity"] == 5
    assert trader.get_positions()[0]["ticker"] == "TCS.NS"


def test_adjust_position_grows_and_weighted_averages_entry_price():
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    trader.adjust_position("TCS.NS", 4, 100.0, signal_hash="ALLOC-TCS.NS")
    trader.adjust_position("TCS.NS", 8, 120.0, signal_hash="ALLOC-TCS.NS")
    pos = trader._positions["ALLOC-TCS.NS"]
    assert pos["quantity"] == 8
    # (4*100 + 4*120.06) / 8 ~= 110 -- weighted, not the latest fill alone
    assert 108 < pos["entry_price"] < 112


def test_adjust_position_shrinks_and_keeps_remainder_open():
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    trader.adjust_position("TCS.NS", 10, 100.0, signal_hash="ALLOC-TCS.NS")
    trader.adjust_position("TCS.NS", 4, 110.0, signal_hash="ALLOC-TCS.NS")
    pos = trader._positions["ALLOC-TCS.NS"]
    assert pos["quantity"] == 4
    # 100.0 * (1 + SLIPPAGE_PCT) from the original opening fill — unchanged
    # by the later partial sell, which must not touch the remaining shares'
    # cost basis.
    assert pos["entry_price"] == pytest.approx(100.05, abs=0.01)


def test_adjust_position_to_zero_closes_fully():
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    trader.adjust_position("TCS.NS", 5, 100.0, signal_hash="ALLOC-TCS.NS")
    trader.adjust_position("TCS.NS", 0, 105.0, signal_hash="ALLOC-TCS.NS")
    assert trader.get_positions() == []


def test_adjust_position_never_double_inserts_the_same_signal_hash():
    """The regression this whole module exists to prevent: remember_trade()
    inserting on a signal_hash a second time would violate the DB's unique
    constraint on trades.signal_hash."""
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    trader.adjust_position("TCS.NS", 4, 100.0, signal_hash="ALLOC-TCS.NS")
    trader.adjust_position("TCS.NS", 8, 101.0, signal_hash="ALLOC-TCS.NS")
    trader.adjust_position("TCS.NS", 2, 102.0, signal_hash="ALLOC-TCS.NS")
    trader.adjust_position("TCS.NS", 0, 103.0, signal_hash="ALLOC-TCS.NS")
    # no AssertionError from FakeBrain means no duplicate insert occurred


def test_adjust_position_noop_when_already_at_target():
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    trader.adjust_position("TCS.NS", 5, 100.0, signal_hash="ALLOC-TCS.NS")
    out = trader.adjust_position("TCS.NS", 5, 101.0, signal_hash="ALLOC-TCS.NS")
    assert out["status"] == "NOOP"


def test_manage_positions_never_tactically_exits_allocator_positions():
    """A -2% index dip is normal variance for a passive book, not a stop."""
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher(price=100.0))
    trader.adjust_position("TCS.NS", 5, 100.0, signal_hash="ALLOC-TCS.NS")
    trader.fetcher.price = 80.0  # -20%, would have hit STOP_LOSS for a
                                 # signal-council position
    closed = trader.manage_positions()
    assert closed == []
    assert len(trader.get_positions()) == 1


def test_rehydrated_allocator_position_still_skips_tactical_exits():
    """Regression: rehydration must not lose the ALLOC- exemption."""
    brain = FakeBrain()
    brain.rows.append({
        "signal_hash": "ALLOC-TCS.NS", "ticker": "TCS.NS", "direction": "LONG",
        "entry_price": 100.0, "quantity": 5,
        "entry_time": datetime.now(timezone.utc), "exit_time": None,
        "is_paper": True,
    })
    trader = PaperTrader(brain, fetcher=FakeFetcher(price=80.0))
    assert trader.manage_positions() == []


# ------------------------------------------------------------------ #
# Vetoes — exclude only, never select

def test_vetoed_tickers_are_excluded_from_target_weights(monkeypatch):
    import src.portfolio.live_allocator as mod
    monkeypatch.setattr(mod, "vetoed_tickers", lambda db_url: {"BAD.NS"})

    idx = pd.bdate_range("2024-01-01", periods=300)
    panel = pd.DataFrame({
        "GOOD.NS": 100 + pd.Series(range(300), index=idx) * 0.1,
        "BAD.NS": 100 + pd.Series(range(300), index=idx) * 0.1,
    }, index=idx)
    allocator = PassiveAllocator(universe=["GOOD.NS", "BAD.NS"], trader=None)
    weights = allocator.target_weights(panel)
    assert "BAD.NS" not in weights
    assert weights.get("GOOD.NS", 0) > 0


# ------------------------------------------------------------------ #
# target_weights.inverse_vol_weights

def test_inverse_vol_weights_sum_to_one():
    idx = pd.bdate_range("2024-01-01", periods=100)
    rets = pd.DataFrame({"A": [0.01, -0.01] * 50, "B": [0.02, -0.02] * 50},
                        index=idx)
    # weight_cap=1.0: only 2 names can't respect the real 20% production
    # cap and still sum to 1 (that infeasibility is its own test below) —
    # this test is about the raw inverse-vol computation, not the cap.
    w = inverse_vol_weights(rets, weight_cap=1.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_inverse_vol_weights_favor_the_lower_vol_name():
    idx = pd.bdate_range("2024-01-01", periods=100)
    rets = pd.DataFrame({"CALM": [0.001, -0.001] * 50,
                        "WILD": [0.05, -0.05] * 50}, index=idx)
    w = inverse_vol_weights(rets, weight_cap=1.0)
    assert w["CALM"] > w["WILD"]


def test_inverse_vol_weights_cap_infeasible_with_too_few_names():
    """2 names can't each stay <=20% and still sum to 1 — the remainder is
    correctly left unallocated (parked in cash) rather than silently
    breaking the cap to force a sum of 1."""
    idx = pd.bdate_range("2024-01-01", periods=100)
    rets = pd.DataFrame({"A": [0.01, -0.01] * 50, "B": [0.02, -0.02] * 50},
                        index=idx)
    w = inverse_vol_weights(rets, weight_cap=0.20)
    assert all(v <= 0.20 + 1e-9 for v in w.values())
    assert sum(w.values()) <= 0.40 + 1e-9


def test_inverse_vol_weights_respects_cap():
    idx = pd.bdate_range("2024-01-01", periods=100)
    # One near-zero-vol name would dominate without a cap.
    rets = pd.DataFrame({
        "FLAT": [0.0001] * 100,
        "B": [0.02, -0.02] * 50, "C": [0.02, -0.02] * 50,
    }, index=idx)
    w = inverse_vol_weights(rets, weight_cap=0.5)
    assert w["FLAT"] <= 0.5 + 1e-9


def test_inverse_vol_weights_empty_input():
    assert inverse_vol_weights(pd.DataFrame()) == {}


# ------------------------------------------------------------------ #
# Allocator end-to-end rebalance, no vetoes

def test_allocator_rebalance_opens_new_positions(monkeypatch):
    import src.portfolio.live_allocator as mod
    monkeypatch.setattr(mod, "vetoed_tickers", lambda db_url: set())

    idx = pd.bdate_range("2024-01-01", periods=300)
    panel = pd.DataFrame({
        "A.NS": 100 + pd.Series(range(300), index=idx) * 0.05,
        "B.NS": 200 + pd.Series(range(300), index=idx) * 0.05,
    }, index=idx)
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    allocator = PassiveAllocator(universe=["A.NS", "B.NS"], trader=trader,
                                 capital=10_000.0)
    result = allocator.rebalance(panel, exposure_multiplier=1.0)
    assert result["rebalanced"] is True
    assert len(trader.get_positions()) > 0


def test_allocator_exposure_zero_liquidates_book(monkeypatch, tmp_path):
    """CRISIS (exposure=0.0) must empty the book, not just block new buys —
    a passive book has no per-position stop protecting it otherwise."""
    import src.portfolio.live_allocator as mod
    monkeypatch.setattr(mod, "vetoed_tickers", lambda db_url: set())

    idx = pd.bdate_range("2024-01-01", periods=300)
    panel = pd.DataFrame({
        "A.NS": 100 + pd.Series(range(300), index=idx) * 0.05,
        "B.NS": 200 + pd.Series(range(300), index=idx) * 0.05,
    }, index=idx)
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher())
    # Own DB: db_url=None falls back to the real data/quntra.db, so the
    # persisted rebalance date would leak into the developer's local state.
    allocator = _alloc_with_db(tmp_path, monkeypatch, trader)
    allocator.universe = ["A.NS", "B.NS"]
    allocator.capital = 10_000.0
    allocator.rebalance(panel, exposure_multiplier=1.0)
    assert len(trader.get_positions()) > 0

    # No _last_rebalance reset here on purpose: a CRISIS de-risk must
    # bypass the weekly cadence by itself. Otherwise a crash the day after
    # a rebalance would leave the book exposed until the next Monday.
    allocator.rebalance(panel, exposure_multiplier=0.0)
    assert trader.get_positions() == []


# ---------------------------------------------------------------- #
# Weekly cadence must survive process restarts.

def _alloc_with_db(tmp_path, monkeypatch, trader):
    import src.db.session as ds
    from src.db import init_db
    url = f"sqlite:///{tmp_path}/alloc.db"
    monkeypatch.setattr(ds, "_engine", None)
    monkeypatch.setattr(ds, "_SessionLocal", None)
    init_db(url)
    return PassiveAllocator(universe=["A.NS", "B.NS"], trader=trader,
                            capital=10_000.0, db_url=url)


def test_is_due_true_when_never_rebalanced(tmp_path, monkeypatch):
    a = _alloc_with_db(tmp_path, monkeypatch, PaperTrader(FakeBrain(),
                                                          fetcher=FakeFetcher(100.0)))
    assert a.is_due() is True


def test_weekly_cadence_survives_restart(tmp_path, monkeypatch):
    """Rebalancer._last_rebalance is in-memory, so a restart used to reset
    the weekly gate and rebalance again immediately — silent extra turnover
    on a platform that redeploys often."""
    import datetime as dt
    trader = PaperTrader(FakeBrain(), fetcher=FakeFetcher(100.0))
    a = _alloc_with_db(tmp_path, monkeypatch, trader)
    today = dt.datetime.now(dt.timezone.utc).date()
    a._save_last_rebalance(today)

    assert a.is_due(today) is False
    # A brand-new object == a restarted process; must still be gated.
    b = PassiveAllocator(universe=["A.NS"], trader=trader,
                         capital=10_000.0, db_url=a.db_url)
    assert b.is_due(today) is False
    # ...but a new ISO week is genuinely due again.
    assert b.is_due(today + dt.timedelta(days=8)) is True


def test_force_bypasses_only_the_weekly_cadence(monkeypatch):
    """After a /capital change the book must be re-sizable the same day,
    without waiting for Monday — but drift/turnover caps still bind."""
    import pandas as pd
    from src.portfolio.live_allocator import PassiveAllocator

    class T:
        ALLOCATOR_PREFIX = "ALLOC:"
        def get_positions(self): return []
        def adjust_position(self, *a, **k): return {"status": "OK"}

    a = PassiveAllocator(universe=["A", "B"], trader=T(), capital=10_000.0)
    seen = {}

    class Reb:
        _last_rebalance = None
        def compute_trades(self, cur, tgt, today):
            seen["last"] = self._last_rebalance
            class D:
                should_rebalance = False
                reason = "weekly cadence"
            return D()

    a.rebalancer = Reb()
    monkeypatch.setattr(a, "_load_last_rebalance", lambda: __import__(
        "datetime").date.today())
    monkeypatch.setattr(a, "target_weights", lambda *a, **k: {"A": 0.5})
    monkeypatch.setattr(a, "current_weights", lambda: {})
    panel = pd.DataFrame({"A": [100.0], "B": [50.0]})

    a.rebalance(panel)
    assert seen["last"] is not None, "normal run must respect the weekly gate"
    a.rebalance(panel, force=True)
    assert seen["last"] is None, "force must clear the weekly gate"
