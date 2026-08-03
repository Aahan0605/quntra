"""Open positions must survive a process restart.

The 2026-07 paper gate was invalidated by two defects that these tests pin:

1. PaperTrader._positions was in-memory only with no rehydration, so every
   watchdog restart orphaned open positions — no stop, no target, no time
   stop, forever.
2. Exits were INSERTed as a second row keyed "<hash>:exit" instead of
   updating the open row, so the original still read as open and every
   P&L / open-position query double-counted.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.execution.paper_trader import PaperTrader


class FakeBrain:
    """Minimal stand-in for QuNtraBrain's trade persistence."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def remember_trade(self, data):
        self.rows.append(dict(data))
        return "id-%d" % len(self.rows)

    def close_trade(self, signal_hash, exit_data):
        for row in self.rows:
            if row.get("signal_hash") == signal_hash and not row.get("exit_time"):
                row.update(exit_data)
                return True
        return False

    def get_open_positions(self, is_paper=True):
        return [r for r in self.rows
                if not r.get("exit_time") and r.get("is_paper") is is_paper]


class FakeFetcher:
    def __init__(self, price):
        self.price = price

    def get_live_quote(self, tickers):
        return pd.DataFrame([{"ticker": tickers[0], "last_price": self.price}])


def test_open_position_survives_restart():
    brain = FakeBrain()
    first = PaperTrader(brain, fetcher=FakeFetcher(100.0), starting_cash=25_000)
    first.place_order("TCS.NS", "LONG", 3, signal_hash="TCS-1", price=100.0)
    assert len(first.get_positions()) == 1

    # Process dies; a brand-new trader starts against the same database.
    revived = PaperTrader(brain, fetcher=FakeFetcher(100.0), starting_cash=25_000)
    assert [p["ticker"] for p in revived.get_positions()] == ["TCS.NS"]


def test_rehydrated_position_is_exit_managed():
    """The orphaned-position bug: a restart must not disarm the stop-loss."""
    entry = datetime.now(timezone.utc) - timedelta(days=1)
    brain = FakeBrain([{
        "signal_hash": "ICICI-1", "ticker": "ICICIBANK.NS", "direction": "LONG",
        "entry_price": 1000.0, "quantity": 5, "entry_time": entry,
        "exit_time": None, "is_paper": True,
    }])
    # Price is 5% down — well through the -2% stop.
    trader = PaperTrader(brain, fetcher=FakeFetcher(950.0), starting_cash=25_000)
    closed = trader.manage_positions()

    assert [c["exit_reason"] for c in closed] == ["STOP_LOSS"]
    assert trader.get_positions() == []
    assert brain.get_open_positions() == []


def test_exit_updates_the_open_row_not_a_new_one():
    brain = FakeBrain()
    trader = PaperTrader(brain, fetcher=FakeFetcher(100.0), starting_cash=25_000)
    trader.place_order("TCS.NS", "LONG", 3, signal_hash="TCS-1", price=100.0)
    trader.close_position("TCS-1", price=104.0, exit_reason="TAKE_PROFIT")

    assert len(brain.rows) == 1, "exit must not create a second row"
    row = brain.rows[0]
    assert row["signal_hash"] == "TCS-1", "no ':exit' suffix rows"
    assert row["exit_time"] is not None
    assert row["exit_reason"] == "TAKE_PROFIT"
    assert brain.get_open_positions() == []


def test_rehydrate_blocks_duplicate_reentry():
    """A restart must not let today's signal re-open an already-open trade."""
    brain = FakeBrain([{
        "signal_hash": "TCS-1", "ticker": "TCS.NS", "direction": "LONG",
        "entry_price": 100.0, "quantity": 3,
        "entry_time": datetime.now(timezone.utc),
        "exit_time": None, "is_paper": True,
    }])
    trader = PaperTrader(brain, fetcher=FakeFetcher(100.0), starting_cash=25_000)
    res = trader.place_order("TCS.NS", "LONG", 3, signal_hash="TCS-1", price=100.0)

    assert res["status"] == "REJECTED"
    assert len(trader.get_positions()) == 1


def test_rehydrate_survives_a_broken_database():
    """A DB failure at startup must not stop the trader from booting."""
    class BrokenBrain(FakeBrain):
        def get_open_positions(self, is_paper=True):
            raise RuntimeError("connection refused")

    trader = PaperTrader(BrokenBrain(), fetcher=FakeFetcher(100.0),
                         starting_cash=25_000)
    assert trader.get_positions() == []


# ---------------------------------------------------------------- #
# Minimum hold window: 15 minutes -> 5 days, no scalping.

def _open_trade(minutes_ago, entry=100.0):
    return FakeBrain([{
        "signal_hash": "TCS-1", "ticker": "TCS.NS", "direction": "LONG",
        "entry_price": entry, "quantity": 3,
        "entry_time": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        "exit_time": None, "is_paper": True,
    }])


def test_no_take_profit_inside_the_min_hold_window():
    brain = _open_trade(minutes_ago=5)
    trader = PaperTrader(brain, fetcher=FakeFetcher(110.0))  # +10%, way past target
    assert trader.manage_positions() == []
    assert len(trader.get_positions()) == 1


def test_take_profit_fires_once_past_the_window():
    brain = _open_trade(minutes_ago=20)
    trader = PaperTrader(brain, fetcher=FakeFetcher(110.0))
    assert [c["exit_reason"] for c in trader.manage_positions()] == ["TAKE_PROFIT"]


def test_stop_loss_is_exempt_from_the_min_hold_window():
    """Capital preservation overrides the hold window."""
    brain = _open_trade(minutes_ago=2)
    trader = PaperTrader(brain, fetcher=FakeFetcher(90.0))  # -10%
    assert [c["exit_reason"] for c in trader.manage_positions()] == ["STOP_LOSS"]
