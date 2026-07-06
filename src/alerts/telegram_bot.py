"""
QuNtra Telegram command center.

Outbound alerts (system -> operator):
    trade filled, stop hit, kill switch, daily circuit, EOD report, errors

Inbound commands (operator -> system):
    /status /pause /resume /report /override /halt

Secrets come from config/secrets.env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
— never hardcoded. With no token configured, the alerter runs in test mode
and records messages locally so the rest of the system works unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[2]
SECRETS = ROOT / "config" / "secrets.env"
logger = logging.getLogger("quntra.telegram")


def _load_secrets() -> dict[str, str]:
    vals: dict[str, str] = {}
    if SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip()
    return vals


class TelegramAlerter:
    """Outbound-only alert channel. Synchronous interface."""

    def __init__(self, token: str | None = None, chat_id: str | None = None,
                 test_mode: bool = False):
        self.token = token
        self.chat_id = chat_id
        self.test_mode = test_mode or not (token and chat_id)
        self.sent: list[str] = []  # test-mode ledger

    @classmethod
    def from_config(cls) -> "TelegramAlerter":
        import os
        sec = _load_secrets()
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or sec.get("TELEGRAM_BOT_TOKEN")
        chat = os.environ.get("TELEGRAM_CHAT_ID") or sec.get("TELEGRAM_CHAT_ID")
        return cls(token=token, chat_id=chat)

    # ------------------------------------------------------------------ #

    def send(self, text: str) -> bool:
        if self.test_mode:
            self.sent.append(text)
            logger.info("[telegram test-mode] %s", text)
            return True
        try:
            from telegram import Bot

            async def _go():
                await Bot(self.token).send_message(chat_id=self.chat_id,
                                                   text=text)
            asyncio.run(_go())
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Telegram send failed: %s", e)
            self.sent.append(f"[FAILED] {text}")
            return False

    # ------------------------------------------------------------------ #
    # The six alert types

    def trade_filled(self, ticker: str, direction: str, price: float,
                     qty: int, score: int) -> bool:
        return self.send(
            f"✅ {direction} {ticker} · ₹{price:,.2f} · {qty} qty · "
            f"Score {score}/12"
        )

    def stop_hit(self, ticker: str, loss_inr: float, day_dd_pct: float) -> bool:
        return self.send(
            f"🛑 STOP HIT: {ticker} · Loss ₹{abs(loss_inr):,.0f} · "
            f"DD {day_dd_pct:+.1%} today"
        )

    def kill_switch(self, n_losses: int) -> bool:
        return self.send(
            f"⚠️ KILL SWITCH TRIGGERED — {n_losses} consecutive losses. "
            f"OMS halted."
        )

    def daily_circuit(self, level: int, dd_pct: float) -> bool:
        return self.send(
            f"🔴 CIRCUIT LEVEL {level} — drawdown {dd_pct:+.1%}. "
            f"OMS disabled rest of session."
        )

    def eod_report(self, pnl_inr: float, sharpe_30d: float, n_trades: int,
                   regime: str, tomorrow_bias: str) -> bool:
        return self.send(
            f"📊 EOD {datetime.now(IST).date()}\n"
            f"P&L: ₹{pnl_inr:+,.0f}\n"
            f"Sharpe (30d): {sharpe_30d:.2f}\n"
            f"Trades today: {n_trades}\n"
            f"Regime: {regime}\n"
            f"Tomorrow bias: {tomorrow_bias}"
        )

    def error_alert(self, module: str, message: str) -> bool:
        return self.send(f"🚨 ERROR in {module}: {message}")

    # Compatibility aliases (completion-loop prompt uses these names)
    def send_message(self, text: str) -> bool:
        return self.send(text)


# Back-compat name used by the completion-loop prompt:
#   from src.alerts.telegram_bot import TelegramBot
#   TelegramBot().send_message("...")
def TelegramBot() -> "TelegramAlerter":
    return TelegramAlerter.from_config()


class QuNtraTelegramBot:
    """Inbound command center: 22 commands. Wire to a HermesCoordinator.

    Every command: answers fast (no blocking network beyond one quote),
    logs itself to system_state, and never lets an exception reach the
    polling loop.
    """

    def __init__(self, hermes, alerter: TelegramAlerter | None = None,
                 db_url: str | None = None):
        self.hermes = hermes
        self.alerter = alerter or TelegramAlerter.from_config()
        self.db_url = db_url

    def _log_command(self, name: str) -> None:
        try:
            self.hermes.set_system_state("last_command", {
                "command": name,
                "at": datetime.now(IST).isoformat(),
            })
        except Exception as e:  # noqa: BLE001
            logger.error("command log failed: %s", e)

    def dispatch(self, name: str, *args) -> str:
        """Run a command by name; errors become readable replies."""
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            return f"Unknown command /{name}"
        self._log_command(name)
        try:
            return handler(*args)
        except Exception as e:  # noqa: BLE001
            logger.exception("command /%s failed", name)
            return f"⚠️ /{name} failed: {e}"

    # Command implementations (framework-independent, unit-testable) ---- #

    def cmd_status(self) -> str:
        state = self.hermes.get_system_state() or {}
        oms = state.get("oms") or {}
        positions = []
        try:
            positions = self.hermes.trader.get_positions()
        except Exception:  # noqa: BLE001
            pass
        return (
            f"QuNtra status @ {datetime.now(IST).strftime('%H:%M IST')}\n"
            f"OMS enabled: {oms.get('enabled', False)}\n"
            f"Open positions: {len(positions)}\n"
            f"Watchlist: {(state.get('premarket') or {}).get('watchlist', [])}\n"
            f"Regime: {(state.get('regime') or {}).get('state', 'unknown')}"
        )

    def cmd_pause(self) -> str:
        self.hermes.set_system_state("oms", {"enabled": False,
                                             "reason": "manual /pause"})
        return "⏸ OMS paused — no new trades. Existing positions held."

    def cmd_resume(self) -> str:
        if self.hermes.circuit is not None:
            self.hermes.circuit.manual_resume()
        if self.hermes.loss_guard is not None:
            self.hermes.loss_guard.resume()
        self.hermes.set_system_state("oms", {"enabled": True,
                                             "reason": "manual /resume"})
        return "▶️ OMS resumed. Circuit breaker and loss guard reset."

    def cmd_report(self) -> str:
        return self.hermes._build_eod_summary()

    def cmd_override(self, signal_hash: str | None = None) -> str:
        if not signal_hash:
            return "Usage: /override <signal_hash>"
        self.hermes.set_system_state("override",
                                     {"signal_hash": signal_hash,
                                      "at": datetime.now(IST).isoformat()})
        return f"Manual override recorded for signal {signal_hash}."

    def cmd_halt(self) -> str:
        self.hermes.set_system_state("oms", {"enabled": False,
                                             "reason": "EMERGENCY /halt"})
        if hasattr(self.hermes.trader, "disable"):
            self.hermes.trader.disable()
        return "🛑 EMERGENCY HALT — all trading stopped immediately."

    # ---- Expanded command center (vision v1.0) ------------------------ #

    def cmd_portfolio(self) -> str:
        """Positions, weights, sector exposure."""
        from src.agents.research.sector_agent import SECTOR_MAP
        positions = self._positions()
        if not positions:
            return "Portfolio empty — no open positions."
        total = sum(float(p.get("entry_price") or 0) * (p.get("quantity")
                    or p.get("qty") or 0) for p in positions) or 1.0
        by_sector: dict[str, float] = {}
        lines = ["*PORTFOLIO*"]
        for p in positions:
            notional = float(p.get("entry_price") or 0) * (p.get("quantity")
                                                           or p.get("qty") or 0)
            sector = SECTOR_MAP.get(p.get("ticker", ""), "OTHER")
            by_sector[sector] = by_sector.get(sector, 0) + notional
            lines.append(f"{p.get('ticker')}: {p.get('quantity') or p.get('qty')} "
                         f"@ ₹{float(p.get('entry_price') or 0):,.2f} "
                         f"({notional / total:.0%})")
        lines.append("\n*SECTOR EXPOSURE*")
        for s, v in sorted(by_sector.items(), key=lambda x: -x[1]):
            lines.append(f"{s}: {v / total:.0%}")
        return "\n".join(lines)

    def cmd_watchlist(self) -> str:
        state = self.hermes.get_system_state("premarket") or {}
        wl = state.get("watchlist", [])
        blackout = state.get("earnings_blackout", [])
        return (f"*WATCHLIST* ({state.get('date', 'n/a')})\n"
                + ("\n".join(f"• {t}" for t in wl) if wl
                   else "empty — no tickers scored ≥ 9/12")
                + (f"\n\nEarnings blackout: {', '.join(blackout)}"
                   if blackout else ""))

    def cmd_open_positions(self) -> str:
        positions = self._positions()
        if not positions:
            return "No open positions."
        lines = ["*OPEN POSITIONS (mark-to-market)*"]
        for p in positions:
            entry = float(p.get("entry_price") or 0)
            mtm = self._last_price(p.get("ticker"))
            qty = p.get("quantity") or p.get("qty") or 0
            if mtm is not None and entry:
                sign = 1 if p.get("direction") == "LONG" else -1
                pnl = sign * (mtm - entry) * qty
                lines.append(f"{p['ticker']}: {qty} @ ₹{entry:,.2f} → "
                             f"₹{mtm:,.2f} · P&L ₹{pnl:+,.0f}")
            else:
                lines.append(f"{p.get('ticker')}: {qty} @ ₹{entry:,.2f} "
                             f"(no live quote)")
        return "\n".join(lines)

    def cmd_risk(self) -> str:
        from src.reporting import metrics as M
        state = self.hermes.get_system_state() or {}
        circuit = self.hermes.circuit
        guard = self.hermes.loss_guard
        dd90 = M.max_drawdown_from_pnl(self.db_url)
        return (
            "*RISK DASHBOARD*\n"
            f"OMS enabled: {(state.get('oms') or {}).get('enabled', False)}\n"
            f"Circuit breaker: "
            f"{'can trade' if circuit is None or circuit.can_enter_new_position(datetime.now(IST)) else 'HALTED'}\n"
            f"Consecutive losses: "
            f"{getattr(guard, 'consecutive_losses', 'n/a')} / "
            f"{getattr(guard, 'max_losses', 3)}"
            f"{' (HALTED)' if getattr(guard, 'halted', False) else ''}\n"
            f"Max DD (90d): {dd90:+.2%}\n"
            f"Regime: {(state.get('regime') or {}).get('state', 'unknown')}"
        )

    def cmd_performance(self) -> str:
        from src.reporting import metrics as M
        sharpe = M.rolling_sharpe(self.db_url)
        dd = M.max_drawdown_from_pnl(self.db_url)
        series = M.daily_pnl_series(self.db_url, days=30)
        total = sum(series.values())
        wins = sum(1 for v in series.values() if v > 0)
        return (
            "*PERFORMANCE (30d rolling)*\n"
            f"Sharpe: {f'{sharpe:.2f}' if sharpe is not None else 'n/a (<5 days)'}\n"
            f"Max DD: {dd:+.2%}\n"
            f"Net P&L: ₹{total:+,.0f} over {len(series)} trading days\n"
            f"Green days: {wins}/{len(series) or 1}"
        )

    def cmd_health(self) -> str:
        checks = {}
        try:
            self.hermes.get_system_state()
            checks["PostgreSQL/DB"] = "OK"
        except Exception as e:  # noqa: BLE001
            checks["PostgreSQL/DB"] = f"DOWN ({e})"
        import subprocess
        r = subprocess.run(["pgrep", "-f", "scripts/scheduler.py"],
                           capture_output=True, text=True)
        checks["Scheduler"] = "OK" if r.stdout.strip() else "NOT RUNNING"
        checks["PaperTrader"] = ("OK" if getattr(self.hermes.trader,
                                                 "enabled", True)
                                 else "DISABLED")
        try:
            self.hermes.brain.get_agent_credibility("_probe")
            checks["Brain"] = "OK"
        except Exception as e:  # noqa: BLE001
            checks["Brain"] = f"DOWN ({e})"
        checks["DataFetcher"] = ("OK" if self._last_price("RELIANCE.NS")
                                 is not None else "DEGRADED (no quote)")
        return "*SYSTEM HEALTH*\n" + "\n".join(
            f"{'✅' if v == 'OK' else '⚠️'} {k}: {v}"
            for k, v in checks.items())

    def cmd_research(self) -> str:
        from src.reporting import metrics as M
        notes = M.research_notes_since(self.db_url, hours=24)
        draft = self.hermes.get_system_state("premarket_draft") or {}
        if draft.get("report"):
            return draft["report"][:3500]
        if notes:
            return "*LATEST RESEARCH (24h)*\n" + "\n".join(
                f"• [{n['source']}] {n['summary']}" for n in notes[:8])
        return "No research recorded in the last 24h."

    def cmd_daily_report(self) -> str:
        from src.reporting import DailyReport
        return DailyReport(self.db_url).generate()[:3800]

    def cmd_weekly_report(self) -> str:
        return self.hermes.generate_weekly_board_report()[:3800]

    def cmd_monthly_report(self) -> str:
        return self.hermes.generate_monthly_investment_letter()[:3800]

    def cmd_start_live(self) -> str:
        gate = self._paper_gate_status()
        if not gate["passed"]:
            return ("⛔ LIVE TRADING BLOCKED — paper gate not met:\n"
                    + "\n".join(f"• {r}" for r in gate["failures"])
                    + "\nThe 40-day gate cannot be shortcut.")
        import os
        if not os.getenv("KITE_API_KEY"):
            return ("Paper gate PASSED, but Kite keys are missing.\n"
                    "Add KITE_API_KEY / KITE_API_SECRET to "
                    "config/secrets.env, set HUMAN_APPROVAL_REQUIRED=true, "
                    "then /start_live again.")
        self.hermes.set_system_state("live_trading", {
            "requested_at": datetime.now(IST).isoformat(),
            "status": "ARMED_PENDING_HUMAN_APPROVAL",
        })
        return ("✅ Paper gate passed. Live trading ARMED — every trade "
                "still needs human approval for the first 30 days.")

    def cmd_stop_live(self) -> str:
        self.hermes.set_system_state("live_trading",
                                     {"status": "STOPPED",
                                      "at": datetime.now(IST).isoformat()})
        return "Live trading stopped. Paper mode continues."

    def cmd_emergency_stop(self) -> str:
        """Squares all positions AND disables the OMS."""
        closed = []
        trader = self.hermes.trader
        if hasattr(trader, "get_positions") and hasattr(trader,
                                                        "close_position"):
            for p in list(trader.get_positions()):
                sh = p.get("signal_hash")
                if sh:
                    try:
                        out = trader.close_position(
                            sh, exit_reason="EMERGENCY_STOP")
                        if out:
                            closed.append(f"{out['ticker']} "
                                          f"₹{out.get('pnl', 0):+,.0f}")
                    except Exception as e:  # noqa: BLE001
                        closed.append(f"{p.get('ticker')}: close FAILED {e}")
        if hasattr(trader, "disable"):
            trader.disable()
        self.hermes.set_system_state("oms", {
            "enabled": False, "reason": "EMERGENCY_STOP (squared all)"})
        return ("🛑 EMERGENCY STOP\nPositions squared: "
                + ("\n".join(closed) if closed else "none open")
                + "\nOMS disabled. /resume to re-arm after review.")

    def cmd_note(self, *words) -> str:
        text = " ".join(words).strip()
        if not text:
            return "Usage: /note <observation>  e.g. /note Oil rising on sanctions"
        from src.agents.intelligence import NoteProcessor
        resp = NoteProcessor(self.db_url, hermes=self.hermes).process(text)
        return resp.to_message()

    def cmd_context(self) -> str:
        state = self.hermes.get_system_state() or {}
        pre = state.get("premarket") or {}
        return (
            "*WHAT HERMES KNOWS RIGHT NOW*\n"
            f"Regime: {(state.get('regime') or {}).get('state', 'unknown')}\n"
            f"Macro bias: {pre.get('macro_bias', 'unknown')}\n"
            f"Watchlist: {pre.get('watchlist', [])}\n"
            f"OMS: {state.get('oms') or {}}\n"
            f"Last overnight: {(state.get('overnight') or {}).get('date', 'never')}\n"
            f"Last health: {(state.get('health') or {}).get('at', 'never')}\n"
            f"User-note bias: {(state.get('user_note_bias') or {}).get('bias', 'none')}"
        )

    def cmd_chat(self, *words) -> str:
        """Freeform Q&A over organizational memory + system state."""
        question = " ".join(words).strip()
        if not question:
            return ("Ask me anything: /chat what did we learn about "
                    "RELIANCE?")
        from src.knowledge import KnowledgeManager
        hits = KnowledgeManager(self.db_url).recall(question, limit=5)
        if hits:
            return ("*From QuNtra's memory:*\n" + "\n".join(
                f"• [{h['knowledge_type']}] {h['content'][:150]}"
                for h in hits))
        return ("Nothing in memory matches that yet. Current context:\n"
                + self.cmd_context())

    # ---- helpers ------------------------------------------------------ #

    def _positions(self) -> list[dict]:
        try:
            return self.hermes.trader.get_positions()
        except Exception:  # noqa: BLE001
            return []

    def _last_price(self, ticker: str | None) -> float | None:
        if not ticker or self.hermes.fetcher is None:
            return None
        try:
            q = self.hermes.fetcher.get_live_quote([ticker])
            return float(q.iloc[0]["last_price"]) if len(q) else None
        except Exception:  # noqa: BLE001
            return None

    def _paper_gate_status(self) -> dict:
        from src.reporting import metrics as M
        failures = []
        series = M.daily_pnl_series(self.db_url, days=120)
        if len(series) < 40:
            failures.append(f"trading days {len(series)}/40")
        sharpe = M.rolling_sharpe(self.db_url)
        if sharpe is None or sharpe <= 1.0:
            failures.append(f"rolling Sharpe "
                            f"{f'{sharpe:.2f}' if sharpe is not None else 'n/a'} "
                            f"(need > 1.0)")
        dd = M.max_drawdown_from_pnl(self.db_url, days=120)
        if dd <= -0.15:
            failures.append(f"max DD {dd:+.2%} (need > -15%)")
        return {"passed": not failures, "failures": failures}

    # Polling entrypoint (needs a real token) --------------------------- #

    COMMANDS = [
        # original six
        "status", "pause", "resume", "report", "override", "halt",
        # vision v1.0 expansion
        "portfolio", "watchlist", "open_positions", "risk", "performance",
        "health", "research", "daily_report", "weekly_report",
        "monthly_report", "start_live", "stop_live", "emergency_stop",
        "note", "context", "chat",
    ]

    def run_polling(self):
        from telegram.ext import ApplicationBuilder, CommandHandler

        if self.alerter.test_mode:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in "
                "config/secrets.env — cannot start polling."
            )

        def make_handler(name):
            async def handler(update, context):
                args = context.args or []
                # dispatch() logs the call and absorbs all errors
                reply = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.dispatch(name, *args))
                await update.message.reply_text(reply[:4000])
            return handler

        app = ApplicationBuilder().token(self.alerter.token).build()
        for name in self.COMMANDS:
            app.add_handler(CommandHandler(name, make_handler(name)))
        logger.info("Telegram bot polling started (%d commands)",
                    len(self.COMMANDS))
        app.run_polling()
