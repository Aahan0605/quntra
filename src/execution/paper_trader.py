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

from src.utils.costs import CostModel

logger = logging.getLogger("quntra.paper")


class PaperTrader:
    SLIPPAGE_PCT = 0.0005  # 0.05% per side

    def __init__(self, brain, fetcher=None, cost_model: CostModel | None = None,
                 starting_cash: float = 25_000.0):
        self.brain = brain
        self.fetcher = fetcher
        self.costs = cost_model or CostModel.from_config()
        self.cash = starting_cash
        self.enabled = True
        self._positions: dict[str, dict] = {}   # signal_hash -> trade
        self._orders: dict[str, dict] = {}      # order_id -> order
        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------ #
    # OMS interface (identical to KiteOMS)

    def place_order(self, ticker: str, direction: str, qty: int,
                    signal_hash: str | None = None,
                    price: float | None = None) -> dict:
        """Simulate a fill at live price +/- slippage. Idempotent by hash."""
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
            self.brain.remember_trade({
                "signal_hash": pos["signal_hash"] + ":exit",
                "ticker": pos["ticker"], "direction": pos["direction"],
                "entry_price": pos["entry_price"], "exit_price": pos["exit_price"],
                "quantity": pos["quantity"], "entry_time": pos["entry_time"],
                "exit_time": pos["exit_time"], "pnl": pos["pnl"],
                "pnl_pct": pos["pnl_pct"], "exit_reason": exit_reason,
                "is_paper": True,
            })
        except Exception as e:  # noqa: BLE001
            logger.error("DB write failed for paper exit: %s", e)
        return pos

    def get_positions(self) -> list[dict]:
        return list(self._positions.values())

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.get("status") == "FILLED":
            return False  # already filled — paper fills are instant
        return self._orders.pop(order_id, None) is not None

    def manage_positions(self) -> list:
        """Trailing-stop management hook — extended in Phase 2 live loop."""
        return []

    def reconcile(self) -> dict:
        return {"open_positions": len(self._positions), "cash": round(self.cash, 2)}

    def disable(self) -> None:
        self.enabled = False
        logger.warning("Paper OMS disabled")

    def enable(self) -> None:
        self.enabled = True
        logger.info("Paper OMS enabled")
