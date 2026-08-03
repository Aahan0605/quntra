"""
PaperTrader — live prices, simulated fills.

Interface-identical to KiteOMS so switching paper -> live is one config
change (PAPER_TRADE=false in .env). Slippage 0.05% per side; all fees
from config/costs.env; every trade lands in PostgreSQL with is_paper=True.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from src.utils.costs import CostModel

logger = logging.getLogger("quntra.paper")


class PaperTrader:
    SLIPPAGE_PCT = 0.0005  # 0.05% per side

    def __init__(self, brain, fetcher=None, cost_model: CostModel | None = None,
                 starting_cash: float = 25_000.0, telegram=None):
        self.brain = brain
        self.fetcher = fetcher
        self.costs = cost_model or CostModel.from_config()
        self.cash = starting_cash
        self.enabled = True
        self.telegram = telegram   # trade lifecycle push notifications
        self._positions: dict[str, dict] = {}   # signal_hash -> trade
        self._orders: dict[str, dict] = {}      # order_id -> order
        self._seen_hashes: set[str] = set()
        self._rehydrate()

    def _rehydrate(self) -> None:
        """Reload open positions from the DB on startup.

        Positions used to live only in this process's memory, so every
        restart orphaned them: no stop-loss, no target, no time stop, and
        they sat open forever. The DB is the source of truth now.
        """
        try:
            open_rows = self.brain.get_open_positions(is_paper=True)
        except Exception as e:  # noqa: BLE001 — never block startup
            logger.error("position rehydrate failed: %s", e)
            return
        for pos in open_rows:
            h = pos["signal_hash"]
            pos.setdefault("fees_inr", 0.0)
            pos["status"] = "FILLED"
            self._positions[h] = pos
            self._seen_hashes.add(h)
            if pos["entry_price"] and pos["quantity"]:
                self.cash -= pos["entry_price"] * pos["quantity"]
        if open_rows:
            logger.warning("rehydrated %d open position(s) from DB: %s",
                           len(open_rows),
                           ", ".join(p["ticker"] for p in open_rows))

    # ------------------------------------------------------------------ #
    # OMS interface (identical to KiteOMS)

    def place_order(self, ticker: str, direction: str, qty: int,
                    signal_hash: str | None = None,
                    price: float | None = None, **meta) -> dict:
        """Simulate a fill at live price +/- slippage. Idempotent by hash.

        meta (optional): score, regime, reasoning, agent_votes — carried
        into the DB row and the Telegram notification.
        """
        if not self.enabled:
            return {"status": "REJECTED", "reason": "OMS disabled",
                    "ticker": ticker}
        signal_hash = signal_hash or uuid.uuid4().hex
        if signal_hash in self._seen_hashes:
            return {"status": "REJECTED", "reason": "duplicate signal_hash",
                    "signal_hash": signal_hash, "ticker": ticker}

        if price is None:
            if self.fetcher is None:
                raise ValueError("No fetcher and no price given")
            quote = self.fetcher.get_live_quote([ticker])
            price = float(quote.iloc[0]["last_price"])

        slip = self.SLIPPAGE_PCT if direction == "LONG" else -self.SLIPPAGE_PCT
        fill_price = price * (1 + slip)
        notional = fill_price * qty
        fees = self.costs.cost_for_trade(notional, delivery=True)

        order_id = uuid.uuid4().hex[:12]
        trade = {
            "order_id": order_id,
            "signal_hash": signal_hash,
            "ticker": ticker,
            "direction": direction,
            "quantity": qty,
            "entry_price": round(fill_price, 2),
            "entry_time": datetime.now(timezone.utc),
            "signal_score": meta.get("score"),
            "regime": meta.get("regime"),
            "is_paper": True,
            "status": "FILLED",
            "fees_inr": round(fees, 2),
        }
        self.cash -= notional + fees
        self._positions[signal_hash] = trade
        self._orders[order_id] = trade
        self._seen_hashes.add(signal_hash)

        db_fields = {k: v for k, v in trade.items()
                     if k not in ("order_id", "status", "fees_inr")}
        try:
            self.brain.remember_trade(db_fields)
        except Exception as e:  # noqa: BLE001
            logger.error("DB write failed for paper trade: %s", e)
        logger.info("PAPER FILL %s %s x%d @ %.2f (fees %.2f)",
                    direction, ticker, qty, fill_price, fees)
        if self.telegram is not None:
            try:
                self.telegram.trade_opened(
                    ticker=ticker, direction=direction,
                    entry_price=fill_price, qty=qty,
                    stop_loss=fill_price * (1 + self.STOP_LOSS_PCT),
                    take_profit=fill_price * (1 + self.TAKE_PROFIT_PCT),
                    score=meta.get("score"),
                    agent_votes=meta.get("agent_votes"),
                    regime=meta.get("regime"),
                    reasoning=meta.get("reasoning"),
                )
            except Exception as e:  # noqa: BLE001 — alerts never block fills
                logger.error("trade-open notification failed: %s", e)
        return trade

    def close_position(self, signal_hash: str, price: float | None = None,
                       exit_reason: str = "MANUAL") -> dict | None:
        pos = self._positions.pop(signal_hash, None)
        if pos is None:
            return None
        if price is None:
            quote = self.fetcher.get_live_quote([pos["ticker"]])
            price = float(quote.iloc[0]["last_price"])
        slip = -self.SLIPPAGE_PCT if pos["direction"] == "LONG" else self.SLIPPAGE_PCT
        exit_price = price * (1 + slip)
        notional = exit_price * pos["quantity"]
        fees = self.costs.cost_for_trade(notional, delivery=True)

        sign = 1 if pos["direction"] == "LONG" else -1
        pnl = sign * (exit_price - pos["entry_price"]) * pos["quantity"] - fees \
            - pos.get("fees_inr", 0.0)
        self.cash += notional - fees

        pos.update({
            "exit_price": round(exit_price, 2),
            "exit_time": datetime.now(timezone.utc),
            "pnl": round(pnl, 2),
            "pnl_pct": round(sign * (exit_price / pos["entry_price"] - 1), 4),
            "exit_reason": exit_reason,
            "status": "CLOSED",
        })
        try:
            closed_ok = self.brain.close_trade(pos["signal_hash"], {
                "exit_price": pos["exit_price"], "exit_time": pos["exit_time"],
                "pnl": pos["pnl"], "pnl_pct": pos["pnl_pct"],
                "exit_reason": exit_reason,
            })
            if not closed_ok:
                logger.error("exit for %s matched no open DB row — "
                             "position closed in memory only",
                             pos["signal_hash"])
        except Exception as e:  # noqa: BLE001
            logger.error("DB write failed for paper exit: %s", e)
        if self.telegram is not None:
            try:
                hold_days = int(np.busday_count(
                    pos["entry_time"].date(), pos["exit_time"].date()))
                self.telegram.trade_closed(
                    ticker=pos["ticker"], direction=pos["direction"],
                    entry_price=float(pos["entry_price"]),
                    exit_price=float(pos["exit_price"]),
                    pnl=float(pos["pnl"]), pnl_pct=float(pos["pnl_pct"]),
                    exit_reason=exit_reason, hold_days=hold_days,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("trade-close notification failed: %s", e)
        return pos

    def adjust_position(self, ticker: str, target_qty: int, price: float,
                        signal_hash: str, reason: str = "REBALANCE") -> dict:
        """Move a position toward `target_qty` shares — the passive
        allocator's primitive. Unlike place_order/close_position (whole
        positions, one entry/exit reason), a rebalance changes size:

          target > current -> buy the delta, weighted-average the entry
                              price so unrealized P&L stays consistent
          target < current -> sell the delta, book realized P&L on just
                              those shares, keep the rest open unchanged
          target == 0       -> ordinary close_position

        No position yet and target > 0 opens fresh via place_order, so
        every code path funnels through the existing DB-write/notification
        logic rather than duplicating it.
        """
        pos = self._positions.get(signal_hash)
        current_qty = pos["quantity"] if pos else 0
        delta = target_qty - current_qty

        if delta == 0:
            return {"status": "NOOP", "ticker": ticker, "qty": current_qty}
        if target_qty <= 0:
            closed = self.close_position(signal_hash, price=price,
                                         exit_reason=reason)
            return closed or {"status": "NOOP", "ticker": ticker, "qty": 0}
        if current_qty == 0:
            return self.place_order(ticker, "LONG", delta,
                                    signal_hash=signal_hash, price=price,
                                    regime=reason)

        if delta > 0:
            # Buying more: weighted-average the entry price across old +
            # new shares so unrealized P&L reflects the true cost basis.
            fill_price = price * (1 + self.SLIPPAGE_PCT)
            notional = fill_price * delta
            fees = self.costs.cost_for_trade(notional, delivery=True)
            self.cash -= notional + fees
            new_qty = current_qty + delta
            pos["entry_price"] = round(
                (pos["entry_price"] * current_qty + fill_price * delta)
                / new_qty, 2)
            pos["quantity"] = new_qty
            pos["fees_inr"] = pos.get("fees_inr", 0.0) + round(fees, 2)
        else:
            # Selling part of the position: realize P&L on the sold shares
            # only; the remainder stays open at its existing entry price.
            sell_qty = -delta
            fill_price = price * (1 - self.SLIPPAGE_PCT)
            notional = fill_price * sell_qty
            fees = self.costs.cost_for_trade(notional, delivery=True)
            self.cash += notional - fees
            pnl = (fill_price - pos["entry_price"]) * sell_qty - fees
            pos["quantity"] = current_qty - sell_qty
            try:
                # A genuinely new, distinct trade record for the realized
                # partial exit — this signal_hash has never been inserted,
                # so it cannot collide with the still-open parent row.
                self.brain.remember_trade({
                    "signal_hash": f"{signal_hash}:{uuid.uuid4().hex[:8]}",
                    "ticker": ticker, "direction": "LONG",
                    "entry_price": pos["entry_price"], "exit_price": round(fill_price, 2),
                    "quantity": sell_qty, "entry_time": pos["entry_time"],
                    "exit_time": datetime.now(timezone.utc), "pnl": round(pnl, 2),
                    "pnl_pct": round(fill_price / pos["entry_price"] - 1, 4),
                    "exit_reason": reason, "is_paper": True,
                })
            except Exception as e:  # noqa: BLE001
                logger.error("DB write failed for partial rebalance sell: %s", e)

        # Update the still-open parent row in place — never a second
        # insert on the same signal_hash (unique constraint).
        try:
            updated = self.brain.update_position_size(
                signal_hash, pos["quantity"], pos["entry_price"])
            if not updated:
                logger.error("rebalance for %s matched no open DB row — "
                            "size updated in memory only", signal_hash)
        except Exception as e:  # noqa: BLE001
            logger.error("DB write failed for rebalance size update: %s", e)
        logger.info("REBALANCE %s -> %d shares @ %.2f", ticker,
                   pos["quantity"], price)
        return pos

    def get_positions(self) -> list[dict]:
        return list(self._positions.values())

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.get("status") == "FILLED":
            return False  # already filled — paper fills are instant
        return self._orders.pop(order_id, None) is not None

    STOP_LOSS_PCT = -0.02      # hard stop per position
    TAKE_PROFIT_PCT = 0.04     # 2:1 reward:risk
    TIME_STOP_DAYS = 5         # matches the 5-day signal horizon
    MIN_HOLD_MINUTES = 15      # no scalping: holds run 15 min -> 5 days

    # The stop-loss is deliberately exempt from MIN_HOLD_MINUTES. Holding a
    # position that has already broken its stop, purely to satisfy a minimum
    # holding period, is how a -2% loss becomes -8%. Capital preservation
    # overrides the hold window.

    # Positions opened by the passive allocator (src/portfolio/live_allocator.py)
    # are managed by weekly weight-drift rebalancing, not tactical stops — a
    # -2% index-tracking dip is normal variance, not a signal to sell. Forcing
    # them through this stock-picker exit engine is exactly the mismatch that
    # would wreck a buy-and-hold strategy with panic-sold whipsaws.
    ALLOCATOR_PREFIX = "ALLOC-"

    def manage_positions(self) -> list:
        """Exit engine: stop-loss / take-profit / 5-day time stop.

        Called every session tick by Hermes. Without exits no trade ever
        closes, and the 40-day paper gate has nothing to measure.
        """
        closed = []
        for signal_hash, pos in list(self._positions.items()):
            if signal_hash.startswith(self.ALLOCATOR_PREFIX):
                continue
            ticker = pos["ticker"]
            try:
                quote = self.fetcher.get_live_quote([ticker])
                last = float(quote.iloc[0]["last_price"])
            except Exception as e:  # noqa: BLE001
                logger.warning("manage_positions: no quote for %s (%s)",
                               ticker, e)
                continue
            sign = 1 if pos["direction"] == "LONG" else -1
            ret = sign * (last / pos["entry_price"] - 1)
            age_days = np.busday_count(
                pos["entry_time"].date(),
                datetime.now(timezone.utc).date())

            held_min = (datetime.now(timezone.utc)
                        - pos["entry_time"]).total_seconds() / 60

            reason = None
            if ret <= self.STOP_LOSS_PCT:
                reason = "STOP_LOSS"
            elif held_min < self.MIN_HOLD_MINUTES:
                reason = None       # too young to take profit or time out
            elif ret >= self.TAKE_PROFIT_PCT:
                reason = "TAKE_PROFIT"
            elif age_days >= self.TIME_STOP_DAYS:
                reason = "TIME_STOP"
            if reason:
                out = self.close_position(signal_hash, price=last,
                                          exit_reason=reason)
                if out:
                    closed.append(out)
        return closed

    def reconcile(self) -> dict:
        return {"open_positions": len(self._positions), "cash": round(self.cash, 2)}

    def disable(self) -> None:
        self.enabled = False
        logger.warning("Paper OMS disabled")

    def enable(self) -> None:
        self.enabled = True
        logger.info("Paper OMS enabled")
