"""The passive allocator — the strategy this project is actually pivoting
to, per docs/CEO_REVIEW.md.

Why this exists: the signal-council stock-picker backtests to -0.51% over
5 real years against +52.55% for buy-and-hold NIFTY
(scripts/backtest_signal_council.py). The inverse-vol portfolio
(src/portfolio/target_weights.py) backtests to Sharpe 1.14 over the same
kind of window, with no per-ticker prediction at all. This class is what
runs that strategy live: pick weights, apply vetoes, scale by crash risk,
rebalance through PaperTrader.

Deliberately NOT using SignalCouncil's per-ticker scoring — that machinery
answers "will this go up," which the backtest shows this system cannot
answer profitably. Vetoes only ever answer "is this obviously wrong,"
which is a lower, betterestablished bar (docs/CEO_REVIEW.md Path B).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from src.portfolio.rebalancer import Rebalancer
from src.portfolio.target_weights import inverse_vol_weights
from src.portfolio.vetoes import vetoed_tickers

logger = logging.getLogger("quntra.allocator")

TRAILING_WINDOW_DAYS = 252   # 1y, matches the validated backtest


class PassiveAllocator:
    def __init__(self, universe: list[str], trader, capital: float = 25_000.0,
                db_url: str | None = None, rebalancer: Rebalancer | None = None):
        self.universe = universe
        self.trader = trader
        self.capital = capital
        self.db_url = db_url
        self.rebalancer = rebalancer or Rebalancer()

    def target_weights(self, panel, exposure_multiplier: float = 1.0,
                       today: date | None = None) -> dict[str, float]:
        """Vetoed tickers get 0, never a weight. Exposure scales the whole
        book down together — CRISIS (0.0) empties it into cash without
        singling out any one name, unlike a per-ticker stop would."""
        vetoed = vetoed_tickers(self.db_url)
        investable = [t for t in self.universe
                     if t in panel.columns and t not in vetoed]
        if vetoed:
            logger.info("allocator: vetoed %s", sorted(vetoed))
        if not investable:
            return {}
        returns = panel[investable].pct_change().tail(
            TRAILING_WINDOW_DAYS).dropna(how="all")
        weights = inverse_vol_weights(returns)
        return {t: w * exposure_multiplier for t, w in weights.items()}

    def current_weights(self) -> dict[str, float]:
        positions = [p for h, p in getattr(self.trader, "_positions", {}).items()
                    if h.startswith(self.trader.ALLOCATOR_PREFIX)]
        total = self.capital
        return {p["ticker"]: (p["entry_price"] * p["quantity"]) / total
               for p in positions}

    _STATE_KEY = "allocator_last_rebalance"

    def _load_last_rebalance(self) -> date | None:
        """Rebalancer keeps _last_rebalance in MEMORY, so every process
        restart resets it to None and the next call rebalances immediately.
        On a platform that redeploys often that silently becomes
        rebalance-on-every-restart — real turnover, real costs, invisible.
        Persisting it makes the weekly cadence mean what it says."""
        try:
            from src.db import SystemState, get_session
            with get_session(self.db_url) as s:
                row = s.get(SystemState, self._STATE_KEY)
                v = (row.value or {}).get("date") if row else None
                return date.fromisoformat(v) if v else None
        except Exception as e:  # noqa: BLE001 — absent state = "never ran"
            logger.warning("could not read last-rebalance date: %s", e)
            return None

    def _save_last_rebalance(self, when: date) -> None:
        try:
            from src.db import SystemState, get_session
            with get_session(self.db_url) as s:
                s.merge(SystemState(key=self._STATE_KEY,
                                    value={"date": when.isoformat()}))
        except Exception as e:  # noqa: BLE001
            logger.error("could not persist last-rebalance date: %s", e)

    def is_due(self, today: date | None = None) -> bool:
        """True when a rebalance is owed — used by the startup catch-up so a
        missed Monday self-heals instead of waiting a whole week."""
        today = today or datetime.now(timezone.utc).date()
        last = self._load_last_rebalance()
        if last is None:
            return True
        return today.isocalendar()[:2] != last.isocalendar()[:2]

    def rebalance(self, panel, exposure_multiplier: float = 1.0,
                  force: bool = False) -> dict:
        """One rebalance pass: weekly cadence, 3% drift, 20% turnover cap —
        all enforced by Rebalancer, not reimplemented here.

        force=True drops ONLY the weekly cadence, for a deliberate operator
        run (e.g. re-sizing the book after a capital change). Drift and
        turnover caps still bind.
        """
        today = datetime.now(timezone.utc).date()
        # Seed from the DB so the weekly gate survives restarts.
        self.rebalancer._last_rebalance = self._load_last_rebalance()
        if force:
            logger.warning("forced rebalance — weekly cadence bypassed")
            self.rebalancer._last_rebalance = None
        # ...but a CRISIS de-risk must never wait for the cadence. Without
        # this, a crash on a Wednesday after Monday's rebalance would leave
        # the book fully exposed until the FOLLOWING Monday — the weekly
        # gate silently vetoing the very protection the crash overlay
        # exists to provide. Getting flat is always allowed; only ADDING
        # risk is rate-limited.
        if exposure_multiplier == 0.0:
            logger.warning("CRISIS exposure 0.0 — bypassing weekly cadence "
                           "to de-risk immediately")
            self.rebalancer._last_rebalance = None
        target = self.target_weights(panel, exposure_multiplier, today)
        current = self.current_weights()
        decision = self.rebalancer.compute_trades(current, target, today)
        if not decision.should_rebalance:
            return {"rebalanced": False, "reason": decision.reason}
        if exposure_multiplier > 0.0:
            self._save_last_rebalance(today)

        executed = []
        for ticker, delta_w in decision.trades.items():
            if ticker not in panel.columns:
                continue
            px = float(panel[ticker].dropna().iloc[-1])
            if px <= 0:
                continue
            new_weight = current.get(ticker, 0.0) + delta_w
            target_qty = max(0, int(new_weight * self.capital / px))
            signal_hash = f"{self.trader.ALLOCATOR_PREFIX}{ticker}"
            result = self.trader.adjust_position(
                ticker, target_qty, px, signal_hash=signal_hash,
                reason="REBALANCE")
            executed.append({"ticker": ticker, "target_qty": target_qty,
                            "result": result.get("status", "OK")})
        return {"rebalanced": True, "turnover": decision.one_way_turnover,
               "trades": executed}
