"""
QuNtra Hermes Coordinator — custom orchestrator.

NOT the NousResearch hermes-agent repo (AGPL-3.0 — banned by license
policy). This is QuNtra's own MIT-licensed Python class.

Hermes delegates and coordinates; it does NOT do analysis itself:
  * pre-market sequence   (06:00–08:45 IST)
  * market session loop   (09:30–14:30 IST, every 60s)
  * post-market sequence  (15:30–18:00 IST)
  * overnight batch       (22:00–06:00 IST)

State lives in the system_state table so a crash/restart resumes cleanly.
All collaborators (brain, trader, fetcher, telegram, circuit breaker)
are injected and duck-typed — testable with stubs, swappable
paper -> live with one config change.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from src.db import SystemState, get_session
from src.utils.universe import UNIVERSE

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("quntra.hermes")

MIN_WATCHLIST_SCORE = 9   # out of 12 — pre-market watchlist gate


class HermesCoordinator:
    def __init__(
        self,
        brain,
        trader,                 # PaperTrader (Phase 2) or KiteOMS (Phase 3)
        fetcher=None,           # UnifiedDataFetcher
        telegram=None,
        circuit_breaker=None,
        loss_guard=None,
        council=None,           # 6-agent council (existing intelligence layer)
        db_url: str | None = None,
    ):
        self.brain = brain
        self.trader = trader
        self.fetcher = fetcher
        self.telegram = telegram
        self.circuit = circuit_breaker
        self.loss_guard = loss_guard
        self.council = council
        self.db_url = db_url

    # ------------------------------------------------------------------ #
    # System state (PostgreSQL-backed)

    def get_system_state(self, key: str | None = None):
        with get_session(self.db_url) as s:
            if key is not None:
                row = s.get(SystemState, key)
                return row.value if row else None
            rows = s.query(SystemState).all()
            return {r.key: r.value for r in rows}

    def set_system_state(self, key: str, value) -> None:
        with get_session(self.db_url) as s:
            s.merge(SystemState(key=key, value=value))

    # ------------------------------------------------------------------ #
    # Daily sequences

    def run_pre_market_sequence(self) -> dict:
        """06:00–08:45 IST: research, cues, watchlist, risk limits."""
        now = datetime.now(IST)
        result = {"started_at": now.isoformat(), "steps": [], "watchlist": []}

        # 1. Global cues (yfinance is allowed for non-India indices)
        cues = self._safe(self._fetch_global_cues, "global_cues", result)

        # 2. News sentiment scan (delegated to council/sentiment agent)
        if self.council is not None and hasattr(self.council, "scan_news"):
            self._safe(self.council.scan_news, "news_scan", result)

        # 3. Options flow (OFIE) — delegated
        if self.council is not None and hasattr(self.council, "scan_options_flow"):
            self._safe(self.council.scan_options_flow, "options_flow", result)

        # 4-5. Council pre-market scoring -> watchlist (score >= 9/12)
        watchlist = []
        if self.council is not None and hasattr(self.council, "score_premarket"):
            scores = self._safe(
                lambda: self.council.score_premarket(UNIVERSE), "council_scoring",
                result) or {}
            watchlist = [t for t, sc in scores.items() if sc >= MIN_WATCHLIST_SCORE]
        result["watchlist"] = watchlist

        # 6. Day risk limits + reset daily guards
        if self.circuit is not None:
            self.circuit.reset_daily()
        if self.loss_guard is not None:
            self.loss_guard.reset()

        self.set_system_state("premarket", {
            "date": now.date().isoformat(),
            "watchlist": watchlist,
            "global_cues": cues,
        })
        self.set_system_state("oms", {"enabled": True, "armed_at": now.isoformat()})
        logger.info("Pre-market complete. Watchlist: %s", watchlist)
        return result

    def run_market_session(self) -> dict:
        """09:30–14:30 IST — called every 60 seconds by the scheduler."""
        now = datetime.now(IST)
        actions: dict = {"at": now.isoformat(), "executed": [], "skipped": [],
                         "steps": []}

        oms_state = self.get_system_state("oms") or {}
        if not oms_state.get("enabled", False):
            actions["skipped"].append("oms disabled")
            return actions

        # 1-2. Refresh quotes + update signal scores (delegated)
        signals = []
        if self.council is not None and hasattr(self.council, "live_signals"):
            watch = (self.get_system_state("premarket") or {}).get("watchlist", [])
            signals = self._safe(lambda: self.council.live_signals(watch),
                                 "live_signals", actions) or []

        # 3. Risk gates
        can_trade = True
        if self.circuit is not None and not self.circuit.can_enter_new_position(now):
            can_trade = False
            actions["skipped"].append("circuit breaker gate")
        if self.loss_guard is not None and self.loss_guard.halted:
            can_trade = False
            actions["skipped"].append("loss guard halt")

        # 4. Execute passing signals
        if can_trade:
            for sig in signals:
                self.brain.remember_signal({**sig, "executed": True})
                trade = self.trader.place_order(
                    ticker=sig["ticker"], direction=sig["direction"],
                    qty=sig.get("qty", 1), signal_hash=sig.get("signal_hash"),
                )
                actions["executed"].append(trade)
        else:
            for sig in signals:
                self.brain.remember_signal({
                    **sig, "executed": False,
                    "rejection_reason": "; ".join(actions["skipped"]),
                })

        # 5. Manage open positions (trailing stops) — delegated to trader
        if hasattr(self.trader, "manage_positions"):
            self._safe(self.trader.manage_positions, "manage_positions", actions)
        return actions

    def run_post_market_sequence(self) -> dict:
        """15:30–18:00 IST: reconcile, journal, score agents, report."""
        now = datetime.now(IST)
        result = {"at": now.isoformat(), "steps": []}

        # 1-2. Reconcile fills + compute P&L
        if hasattr(self.trader, "reconcile"):
            self._safe(self.trader.reconcile, "reconcile", result)

        # 3-4. Score agent council accuracy -> update credibility
        if self.council is not None and hasattr(self.council, "score_day"):
            outcomes = self._safe(self.council.score_day, "council_scoring",
                                  result) or []
            for agent_name, correct in outcomes:
                self.brain.update_agent_credibility(agent_name, correct)

        # 5. Journal is already in PostgreSQL via brain.remember_trade

        # 6. Telegram EOD report
        if self.telegram is not None:
            summary = self._build_eod_summary()
            self._safe(lambda: self.telegram.send(summary), "eod_report", result)

        self.set_system_state("post_market", {"date": now.date().isoformat(),
                                              "done": True})
        return result

    def run_overnight_batch(self) -> dict:
        """22:00–06:00 IST: retrain, optimize, refit, calendar."""
        now = datetime.now(IST)
        result = {"at": now.isoformat(), "steps": []}
        for step_name, attr in [
            ("xgb_retrain", "retrain_models"),
            ("qaoa_reoptimize", "run_qaoa"),
            ("genetic_evolve", "evolve_generation"),
            ("hmm_refit", "refit_regime"),
            ("events_calendar", "fetch_calendar"),
        ]:
            target = getattr(self.council, attr, None) if self.council else None
            if callable(target):
                self._safe(target, step_name, result)
            else:
                result["steps"].append(f"{step_name}: no handler (skipped)")
        self.set_system_state("overnight", {"date": now.date().isoformat(),
                                            "result": result["steps"]})
        return result

    # ------------------------------------------------------------------ #
    # Scheduler hook points (thin wrappers so cron jobs stay 1-liners)

    def arm_system(self):
        self.set_system_state("oms", {"enabled": True,
                                      "armed_at": datetime.now(IST).isoformat()})

    def observe_market_open(self):
        self.set_system_state("market_open_observed",
                              {"at": datetime.now(IST).isoformat()})

    def start_trading_session(self):
        self.set_system_state("session", {"active": True,
                                          "since": datetime.now(IST).isoformat()})

    def begin_close_management(self):
        """14:30 IST — tighten stops, no new trades."""
        self.set_system_state("oms", {"enabled": False,
                                      "reason": "close management"})

    def send_eod_report(self):
        if self.telegram is not None:
            self.telegram.send(self._build_eod_summary())

    def start_overnight_research(self):
        self.set_system_state("overnight_research",
                              {"started": datetime.now(IST).isoformat()})

    def health_check(self) -> dict:
        status = {"at": datetime.now(IST).isoformat(), "db": False}
        try:
            self.get_system_state()
            status["db"] = True
        except Exception as e:  # noqa: BLE001
            status["error"] = str(e)
            if self.telegram is not None:
                self.telegram.send(f"🚨 ERROR in health_check: {e}")
        self.set_system_state("health", status)
        return status

    # ------------------------------------------------------------------ #

    def _fetch_global_cues(self) -> dict:
        cues: dict = {}
        if self.fetcher is None:
            return cues
        try:
            import yfinance as yf
            for name, tick in [("sp500", "^GSPC"), ("nasdaq", "^IXIC"),
                               ("nikkei", "^N225")]:
                h = yf.Ticker(tick).history(period="2d")
                if len(h) >= 2:
                    cues[name] = round(
                        float(h["Close"].iloc[-1] / h["Close"].iloc[-2] - 1), 4
                    )
        except Exception as e:  # noqa: BLE001
            cues["error"] = str(e)
        return cues

    def _build_eod_summary(self) -> str:
        trades = []
        if hasattr(self.trader, "get_positions"):
            try:
                trades = self.trader.get_positions()
            except Exception:  # noqa: BLE001
                pass
        state = self.get_system_state("premarket") or {}
        return (
            f"📊 QuNtra EOD {datetime.now(IST).date()}\n"
            f"Open positions: {len(trades)}\n"
            f"Watchlist was: {state.get('watchlist', [])}\n"
        )

    @staticmethod
    def _safe(fn, name: str, result: dict):
        try:
            out = fn()
            result["steps"].append(f"{name}: ok")
            return out
        except Exception as e:  # noqa: BLE001
            logger.exception("step %s failed", name)
            result["steps"].append(f"{name}: ERROR {e}")
            return None
