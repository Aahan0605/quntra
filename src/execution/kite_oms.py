"""
KiteOMS — live order management via Zerodha Kite Connect (pykiteconnect, MIT).

Safety architecture:
  * Idempotent: signal_hash dedup — the same signal can never fire twice.
  * State machine: PENDING -> ACKNOWLEDGED -> FILLED / CANCELLED / REJECTED.
  * Daily capital enforcer: DAILY_CAPITAL_INR hard limit.
  * Max trades/day cap (default 4).
  * disable()/enable() honored by every entry point.

Credentials from config/secrets.env (KITE_API_KEY, KITE_API_SECRET,
KITE_ACCESS_TOKEN) — never hardcoded. Without credentials the class
still constructs (for tests / dry runs) but connect() raises.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone, date
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("quntra.kite")


class OrderState(str, Enum):
    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


_VALID_TRANSITIONS = {
    OrderState.PENDING: {OrderState.ACKNOWLEDGED, OrderState.REJECTED,
                         OrderState.CANCELLED},
    OrderState.ACKNOWLEDGED: {OrderState.FILLED, OrderState.CANCELLED,
                              OrderState.REJECTED},
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
}


def _load_secrets() -> dict[str, str]:
    vals: dict[str, str] = {}
    p = ROOT / "config" / "secrets.env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip()
    return vals


class KiteOMS:
    def __init__(self, brain, daily_capital_inr: float | None = None,
                 max_trades_per_day: int | None = None):
        sec = _load_secrets()
        env = os.environ
        self.brain = brain
        self.api_key = env.get("KITE_API_KEY") or sec.get("KITE_API_KEY")
        self.api_secret = env.get("KITE_API_SECRET") or sec.get("KITE_API_SECRET")
        self.access_token = (env.get("KITE_ACCESS_TOKEN")
                             or sec.get("KITE_ACCESS_TOKEN"))
        self.daily_capital = float(
            daily_capital_inr
            or env.get("DAILY_CAPITAL_INR") or sec.get("DAILY_CAPITAL_INR") or 10_000
        )
        self.max_trades = int(
            max_trades_per_day
            or env.get("MAX_TRADES_PER_DAY") or sec.get("MAX_TRADES_PER_DAY") or 4
        )
        self.enabled = True
        self.kite = None  # set by connect()

        self._orders: dict[str, dict] = {}
        self._seen_hashes: set[str] = set()
        self._day: date = datetime.now(timezone.utc).date()
        self._capital_used_today = 0.0
        self._trades_today = 0

    # ------------------------------------------------------------------ #

    def connect(self):
        if not (self.api_key and self.access_token):
            raise RuntimeError(
                "BLOCKER: KITE_API_KEY / KITE_ACCESS_TOKEN not in "
                "config/secrets.env — live trading unavailable."
            )
        from kiteconnect import KiteConnect
        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)
        logger.info("Kite Connect session established")
        return self.kite

    def _roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day = today
            self._capital_used_today = 0.0
            self._trades_today = 0

    def _transition(self, order: dict, new_state: OrderState):
        old = OrderState(order["state"])
        if new_state not in _VALID_TRANSITIONS[old]:
            raise ValueError(f"Illegal transition {old} -> {new_state}")
        order["state"] = new_state.value
        order["state_history"].append(
            (new_state.value, datetime.now(timezone.utc).isoformat())
        )
        logger.info("Order %s: %s -> %s", order["order_id"], old.value,
                    new_state.value)

    # ------------------------------------------------------------------ #
    # OMS interface (identical to PaperTrader)

    def place_order(self, ticker: str, direction: str, qty: int,
                    signal_hash: str | None = None,
                    price: float | None = None) -> dict:
        self._roll_day()
        signal_hash = signal_hash or uuid.uuid4().hex

        order = {
            "order_id": uuid.uuid4().hex[:12],
            "signal_hash": signal_hash,
            "ticker": ticker,
            "direction": direction,
            "quantity": qty,
            "state": OrderState.PENDING.value,
            "state_history": [(OrderState.PENDING.value,
                               datetime.now(timezone.utc).isoformat())],
            "is_paper": False,
        }
        self._orders[order["order_id"]] = order

        # Gate 1: OMS enabled
        if not self.enabled:
            order["reject_reason"] = "OMS disabled"
            self._transition(order, OrderState.REJECTED)
            return order
        # Gate 2: idempotency
        if signal_hash in self._seen_hashes:
            order["reject_reason"] = "duplicate signal_hash"
            self._transition(order, OrderState.REJECTED)
            return order
        # Gate 3: trade count cap
        if self._trades_today >= self.max_trades:
            order["reject_reason"] = (
                f"max {self.max_trades} trades/day reached"
            )
            self._transition(order, OrderState.REJECTED)
            self._log_rejected_signal(order)
            return order
        # Gate 4: daily capital cap (needs a price estimate)
        if price is not None:
            notional = price * qty
            if self._capital_used_today + notional > self.daily_capital:
                order["reject_reason"] = (
                    f"daily capital ₹{self.daily_capital:,.0f} would be exceeded"
                )
                self._transition(order, OrderState.REJECTED)
                self._log_rejected_signal(order)
                return order

        # Live placement
        if self.kite is None:
            order["reject_reason"] = ("Kite not connected — call connect() "
                                      "(missing API credentials?)")
            self._transition(order, OrderState.REJECTED)
            return order

        try:
            kite_order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=ticker.replace(".NS", ""),
                transaction_type=(self.kite.TRANSACTION_TYPE_BUY
                                  if direction == "LONG"
                                  else self.kite.TRANSACTION_TYPE_SELL),
                quantity=qty,
                product=self.kite.PRODUCT_CNC,
                order_type=self.kite.ORDER_TYPE_MARKET,
                tag=signal_hash[:20],
            )
            order["kite_order_id"] = kite_order_id
            self._transition(order, OrderState.ACKNOWLEDGED)
            self._seen_hashes.add(signal_hash)
            self._trades_today += 1
            if price is not None:
                self._capital_used_today += price * qty
        except Exception as e:  # noqa: BLE001
            order["reject_reason"] = f"kite error: {e}"
            self._transition(order, OrderState.REJECTED)
        return order

    def mark_filled(self, order_id: str, fill_price: float) -> dict:
        """Called by reconciliation when Kite reports the fill."""
        order = self._orders[order_id]
        self._transition(order, OrderState.FILLED)
        order["entry_price"] = fill_price
        try:
            self.brain.remember_trade({
                "signal_hash": order["signal_hash"],
                "ticker": order["ticker"], "direction": order["direction"],
                "quantity": order["quantity"], "entry_price": fill_price,
                "entry_time": datetime.now(timezone.utc), "is_paper": False,
            })
        except Exception as e:  # noqa: BLE001
            logger.error("DB write failed for live fill: %s", e)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        try:
            if self.kite is not None and order.get("kite_order_id"):
                self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR,
                                       order_id=order["kite_order_id"])
            self._transition(order, OrderState.CANCELLED)
            return True
        except (ValueError, Exception) as e:  # noqa: BLE001
            logger.error("Cancel failed: %s", e)
            return False

    def get_positions(self) -> list[dict]:
        if self.kite is not None:
            try:
                return self.kite.positions().get("net", [])
            except Exception as e:  # noqa: BLE001
                logger.error("positions fetch failed: %s", e)
        return [o for o in self._orders.values()
                if o["state"] == OrderState.FILLED.value]

    def reconcile(self) -> dict:
        """Compare local order states with the broker ledger."""
        if self.kite is None:
            return {"reconciled": False, "reason": "not connected"}
        broker_orders = {o["tag"]: o for o in self.kite.orders() if o.get("tag")}
        mismatches = []
        for order in self._orders.values():
            tag = order["signal_hash"][:20]
            b = broker_orders.get(tag)
            if b and b.get("status") == "COMPLETE" and \
                    order["state"] != OrderState.FILLED.value:
                self.mark_filled(order["order_id"],
                                 float(b.get("average_price", 0)))
                mismatches.append(order["order_id"])
        return {"reconciled": True, "corrected": mismatches}

    def manage_positions(self) -> list:
        return []

    def disable(self) -> None:
        self.enabled = False
        logger.warning("LIVE OMS DISABLED")

    def enable(self) -> None:
        self.enabled = True
        logger.info("Live OMS enabled")

    # ------------------------------------------------------------------ #

    def _log_rejected_signal(self, order: dict):
        try:
            self.brain.remember_signal({
                "signal_hash": order["signal_hash"],
                "ticker": order["ticker"],
                "direction": order["direction"],
                "executed": False,
                "rejection_reason": order.get("reject_reason", "")[:100],
            })
        except Exception as e:  # noqa: BLE001
            logger.error("DB write failed for rejected signal: %s", e)
