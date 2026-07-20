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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.db import SystemState, get_session
from src.utils.universe import UNIVERSE

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("quntra.hermes")

MIN_WATCHLIST_SCORE = 9   # out of 12 — pre-market watchlist gate


class HermesCoordinator:
    """QuNtra CEO. Delegates to specialist teams; never analyzes itself.

    Teams: Research (news/macro/company/sector/fundamental/geopolitical +
    writer), Quant (council, DailyTrainer, overnight pipeline), Risk
    (circuit breaker, loss guard), Engineering (brain, knowledge, reports).
    """

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
        research_team: dict | None = None,   # {name: agent} override for tests
        knowledge=None,
    ):
        self.brain = brain
        self.trader = trader
        self.fetcher = fetcher
        self.telegram = telegram
        self.circuit = circuit_breaker
        self.loss_guard = loss_guard
        self.council = council
        self.db_url = db_url
        self._research_team = research_team
        self._knowledge = knowledge

    # ------------------------------------------------------------------ #
    # Lazily-built teams (injectable for tests)

    @property
    def research_team(self) -> dict:
        if self._research_team is None:
            from src.agents.research import (
                CompanyAnalysisAgent,
                FundamentalAgent,
                GeopoliticalAgent,
                MacroAgent,
                NewsAgent,
                SectorAgent,
            )
            self._research_team = {
                "news_agent": NewsAgent(self.db_url),
                "macro_agent": MacroAgent(self.db_url, fetcher=self.fetcher),
                "geopolitical_agent": GeopoliticalAgent(self.db_url),
                "fundamental_agent": FundamentalAgent(self.db_url),
                "sector_agent": SectorAgent(self.db_url),
                "company_analysis_agent": CompanyAnalysisAgent(self.db_url),
            }
        return self._research_team

    @property
    def knowledge(self):
        if self._knowledge is None:
            from src.knowledge import KnowledgeManager
            self._knowledge = KnowledgeManager(self.db_url)
        return self._knowledge

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
        """06:00–08:45 IST: research team, synthesis, watchlist, risk limits."""
        now = datetime.now(IST)
        result = {"started_at": now.isoformat(), "steps": [], "watchlist": []}

        # 1-4. Research team: news, macro, geopolitical, earnings/events
        outputs = {}
        watch = (self.get_system_state("premarket") or {}).get("watchlist", [])
        ctx = {"date": now.date().isoformat(), "watchlist": watch,
               "regime": (self.get_system_state("regime") or {}).get(
                   "state", "UNKNOWN")}
        for name in ("news_agent", "macro_agent", "geopolitical_agent",
                     "fundamental_agent", "sector_agent",
                     "company_analysis_agent"):
            agent = self.research_team.get(name)
            if agent is None:
                continue
            out = agent.safe_run(ctx)
            outputs[name] = out
            agent.store(out)
            result["steps"].append(f"{name}: "
                                   + ("ok" if out.ok else f"ERROR {out.error}"))

        # Legacy global cues (kept for downstream consumers)
        cues = self._safe(self._fetch_global_cues, "global_cues", result)

        # 5. Risk snapshot before arming
        risk_snapshot = {
            "circuit_halted": (not self.circuit.can_enter_new_position(now)
                               if self.circuit is not None else False),
            "loss_guard_halted": (self.loss_guard.halted
                                  if self.loss_guard is not None else False),
        }

        # 6. Synthesize the pre-market intelligence report
        report = self._safe(
            lambda: self._compose_premarket_report(outputs, ctx),
            "research_synthesis", result) or ""

        # 7-8. Council scoring -> watchlist, filtered through risk limits
        watchlist = []
        if self.council is not None and hasattr(self.council, "score_premarket"):
            scores = self._safe(
                lambda: self.council.score_premarket(UNIVERSE), "council_scoring",
                result) or {}
            watchlist = [t for t, sc in scores.items() if sc >= MIN_WATCHLIST_SCORE]
        # Earnings blackout: never trade into a report
        corp = outputs.get("company_analysis_agent")
        blackout = (corp.payload.get("earnings_blackout", [])
                    if corp is not None and corp.ok else [])
        watchlist = [t for t in watchlist if t not in blackout]
        result["watchlist"] = watchlist

        # Risk limits + daily guard reset
        if self.circuit is not None:
            self.circuit.reset_daily()
        if self.loss_guard is not None:
            self.loss_guard.reset()

        # 9. Telegram
        if self.telegram is not None and report:
            self._safe(lambda: self.telegram.send(report), "telegram", result)

        # 10. Persist state
        self.set_system_state("premarket", {
            "date": now.date().isoformat(),
            "watchlist": watchlist,
            "earnings_blackout": blackout,
            "global_cues": cues,
            "risk_snapshot": risk_snapshot,
            "macro_bias": (outputs.get("macro_agent").payload.get("macro_bias")
                           if outputs.get("macro_agent") is not None
                           and outputs["macro_agent"].ok else "UNKNOWN"),
        })
        self.set_system_state("oms", {"enabled": True, "armed_at": now.isoformat()})
        logger.info("Pre-market complete. Watchlist: %s", watchlist)
        return result

    def _compose_premarket_report(self, outputs: dict, ctx: dict) -> str:
        from src.agents.research import ResearchWriter
        return ResearchWriter(self.db_url).compose(outputs, ctx)

    # Vision-v1.0 name; the scheduler's 60s job calls run_market_session()
    def run_market_session_tick(self) -> dict:
        return self.run_market_session()

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
            regime = (self.get_system_state("regime") or {}).get("state")
            for sig in signals:
                self.brain.remember_signal({**sig, "executed": True})
                trade = self.trader.place_order(
                    ticker=sig["ticker"], direction=sig["direction"],
                    qty=sig.get("qty", 1), signal_hash=sig.get("signal_hash"),
                    score=sig.get("score"), regime=sig.get("regime", regime),
                    agent_votes=sig.get("agent_votes"),
                    reasoning=sig.get("reasoning"),
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

        # 5. Lessons learned from today's losing trades -> knowledge base
        self._safe(self._store_daily_lessons, "lessons", result)

        # 6. Journal is already in PostgreSQL via brain.remember_trade

        # 7. Full EOD report (metrics from DB) via Telegram + archive
        self._safe(self._send_daily_report, "eod_report", result)

        self.set_system_state("post_market", {"date": now.date().isoformat(),
                                              "done": True})
        return result

    def _store_daily_lessons(self) -> None:
        """Every losing trade closed today becomes a TRADE_LESSON."""
        from datetime import timezone as _tz
        today = datetime.now(_tz.utc).date()
        regime = (self.get_system_state("regime") or {}).get("state")
        for t in self.brain.get_recent_trades(days=2):
            exit_time = t.get("exit_time")
            if exit_time is None or exit_time.date() != today:
                continue
            pnl = t.get("pnl") or 0
            if pnl >= 0:
                continue
            self.knowledge.store(
                knowledge_type="TRADE_LESSON",
                content=(f"LOSS ₹{pnl:,.0f} on {t['ticker']} "
                         f"({t['direction']}, score {t.get('signal_score')}, "
                         f"regime {t.get('regime')}) — review entry conditions"),
                tickers=[t["ticker"]],
                confidence=0.6,
                source="post_market_review",
                regime=regime or t.get("regime"),
            )

    def _send_daily_report(self) -> None:
        from src.reporting import DailyReport
        DailyReport(self.db_url, telegram=self.telegram).generate()

    def run_overnight_batch(self) -> dict:
        """22:00–06:00 IST: learn from trades, research, optimize, refit."""
        now = datetime.now(IST)
        result = {"at": now.isoformat(), "steps": []}

        # 1. Daily self-learning loop (rolling 90-day trade retrain)
        self._safe(self._run_daily_trainer, "daily_trainer", result)

        # 2-4. Council-owned optimizers (QAOA, genetic, HMM) when wired
        for step_name, attr in [
            ("qaoa_reoptimize", "run_qaoa"),
            ("genetic_evolve", "evolve_generation"),
            ("hmm_refit", "refit_regime"),
        ]:
            target = getattr(self.council, attr, None) if self.council else None
            if callable(target):
                self._safe(target, step_name, result)
            else:
                result["steps"].append(f"{step_name}: no handler (skipped)")

        # 5-10. The nine-task overnight research pipeline (company reads,
        # RBI, macro, papers, maintenance, tomorrow's pre-market draft)
        self._safe(self._run_overnight_pipeline, "research_pipeline", result)

        self.set_system_state("overnight", {"date": now.date().isoformat(),
                                            "result": result["steps"]})
        return result

    def _run_daily_trainer(self) -> None:
        from src.learning import DailyTrainer
        DailyTrainer(self.brain, self.db_url, telegram=self.telegram).run()

    def _run_overnight_pipeline(self) -> None:
        from src.research import OvernightResearchPipeline
        report = OvernightResearchPipeline(
            self.db_url, telegram=self.telegram,
            knowledge=self.knowledge, fetcher=self.fetcher).run()
        logger.info("Overnight pipeline: %s", report.summary())

    # ------------------------------------------------------------------ #
    # Periodic reports

    def generate_weekly_board_report(self) -> str:
        """Sunday 8 PM IST: internal board report."""
        from src.reporting import WeeklyReport
        return WeeklyReport(self.db_url, telegram=self.telegram).generate()

    def generate_monthly_investment_letter(self) -> str:
        """1st of month, 9 AM IST: investment letter."""
        from src.reporting import MonthlyLetter
        return MonthlyLetter(self.db_url, telegram=self.telegram).generate()

    def send_weekly_paper_recap(self) -> str:
        """Friday 6 PM IST — where the 40-day paper gate stands."""
        from src.reporting import metrics as M
        now = datetime.now(timezone.utc)
        week = M.trades_between(self.db_url, now - timedelta(days=7), now)
        stats = M.pnl_stats(week)
        series = M.daily_pnl_series(self.db_url, days=120)
        days_done = len(series)
        sharpe = M.rolling_sharpe(self.db_url)
        dd = M.max_drawdown_from_pnl(self.db_url, days=120)
        sharpe_pass = sharpe is not None and sharpe > 1.0
        dd_pass = dd > -0.15

        lines = [
            "📅 Weekly Paper Trading Recap",
            f"Trades this week: {stats['n_trades']}",
            f"Week P&L: ₹{stats['net_pnl']:+,.2f}",
            f"Paper gate: day {days_done}/40",
        ]
        if days_done >= 40 and sharpe_pass and dd_pass:
            lines += ["", "🎉 PAPER GATE PASSED!",
                      "Ready for live capital deployment.",
                      "Next step: /start_live"]
        else:
            lines += [f"Days remaining: {max(0, 40 - days_done)}",
                      "Stay patient. Capital preservation first."]
        msg = "\n".join(lines)
        if self.telegram is not None:
            self.telegram.send(msg)
        return msg

    # ------------------------------------------------------------------ #
    # Gate completion — the full day-by-day history, sent once the 40-day
    # mark is first reached (pass OR fail; the operator needs to see both).

    def check_gate_completion(self) -> bool:
        """Called daily after EOD. Sends the report exactly once, tracked
        via system_state so a scheduler restart can't duplicate it."""
        from src.reporting import GateCompletionReport
        return GateCompletionReport(self.db_url, telegram=self.telegram) \
            .send_if_gate_reached()

    def generate_gate_report_now(self) -> str:
        """On-demand full history — /gate_report can be run any day,
        including before the 40-day mark, to see progress so far."""
        from src.reporting import GateCompletionReport
        report = GateCompletionReport(self.db_url, telegram=None).generate()
        return report or "No paper trades recorded yet."

    # ------------------------------------------------------------------ #
    # Obsidian export — mirror the DB into a linked markdown vault the
    # operator can browse. Read-only against trading; failures never raise.

    def sync_obsidian(self) -> dict:
        """Regenerate the Obsidian vault from the database."""
        from src.integrations import ObsidianVault
        try:
            return ObsidianVault(db_url=self.db_url).sync()
        except Exception as e:  # noqa: BLE001
            logger.exception("Obsidian sync failed")
            return {"error": str(e)}

    # ------------------------------------------------------------------ #
    # Scheduler hook points (thin wrappers so cron jobs stay 1-liners)

    def arm_system(self):
        """08:45 IST — arm the OMS and push the morning briefing."""
        self.set_system_state("oms", {"enabled": True,
                                      "armed_at": datetime.now(IST).isoformat()})
        try:
            self.send_morning_briefing()
        except Exception:  # noqa: BLE001 — briefing must never block arming
            logger.exception("morning briefing failed")

    def send_morning_briefing(self) -> str:
        """Everything QuNtra found overnight, pushed to the operator."""
        pre = self.get_system_state("premarket") or {}
        regime = (self.get_system_state("regime") or {}).get("state",
                                                             "UNKNOWN")
        watchlist = pre.get("watchlist") or []
        blackout = pre.get("earnings_blackout") or []
        draft = (self.get_system_state("premarket_draft") or {}).get(
            "report", "")

        lines = [
            f"☀️ QuNtra Morning Briefing — "
            f"{datetime.now(IST).strftime('%d %b %Y')}",
            f"🎯 Regime: {regime}",
            f"🌍 Macro: {pre.get('macro_bias', 'UNKNOWN')}",
        ]
        if watchlist:
            shown = ", ".join(watchlist[:5])
            extra = f" +{len(watchlist) - 5} more" if len(watchlist) > 5 else ""
            lines.append(f"📋 Watchlist ({len(watchlist)}): {shown}{extra}")
        else:
            lines.append("📋 Watchlist: empty — no tickers scored ≥ 9/12")
        if blackout:
            lines.append(f"📅 Earnings today: {', '.join(blackout)} "
                         f"(will NOT be traded)")
        risks = [line.strip("• ").strip()
                 for line in draft.splitlines()
                 if line.strip().startswith("•")][:3]
        if risks:
            lines.append("⚠️ Top risks:")
            lines += [f"  • {r}" for r in risks]
        lines.append("Max trades today: 3 · Capital: ₹25,000 · Mode: PAPER")
        lines.append("/watchlist for the full list · /research for the "
                     "full report")
        msg = "\n".join(lines)
        if self.telegram is not None:
            self.telegram.send(msg)
        return msg

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

    def send_eod_report(self) -> str:
        """17:00 IST — compact EOD push; /daily_report has the full story."""
        try:
            from src.reporting import metrics as M
            trades = self.brain.get_todays_trades()
            closed = [t for t in trades if t.get("pnl") is not None]
            day_pnl = sum(t["pnl"] for t in closed)
            winners = sum(1 for t in closed if t["pnl"] > 0)
            losers = sum(1 for t in closed if t["pnl"] <= 0)
            sharpe = M.rolling_sharpe(self.db_url)
            dd = M.max_drawdown_from_pnl(self.db_url)
            series = M.daily_pnl_series(self.db_url, days=120)
            days_done = len(series)

            emoji = "✅" if day_pnl > 0 else ("🔴" if day_pnl < 0 else "⬜")
            lines = [
                f"{emoji} QuNtra EOD Summary — "
                f"{datetime.now(IST).strftime('%d %b %Y')}",
                (f"Today: {len(trades)} trades ({winners}W/{losers}L) · "
                 f"P&L ₹{day_pnl:+,.2f}" if trades
                 else "No trades today."),
                "",
                "📊 Rolling 30d:",
                f"  Sharpe: "
                + (f"{sharpe:.3f}" if sharpe is not None else "n/a"),
                f"  Max DD: {dd:+.2%}",
                "",
                f"🎯 Paper gate: day {days_done}/40 · "
                f"Sharpe {'✓' if sharpe is not None and sharpe > 1 else '⏳'} · "
                f"DD {'✓' if dd > -0.15 else '✗'}",
                "/daily_report for the full breakdown",
            ]
            msg = "\n".join(lines)
        except Exception:  # noqa: BLE001 — fall back to the simple summary
            logger.exception("compact EOD failed — sending basic summary")
            msg = self._build_eod_summary()
        if self.telegram is not None:
            self.telegram.send(msg)
        return msg

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
