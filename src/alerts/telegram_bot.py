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
    """Outbound-only alert channel. Synchronous interface.

    chat_id is captured by the bot runner on the operator's first message
    and persisted to config/secrets.env — send() re-reads it lazily so a
    long-running scheduler picks it up without a restart.
    """

    def __init__(self, token: str | None = None, chat_id: str | None = None,
                 test_mode: bool = False):
        self.token = token
        self.chat_id = chat_id
        self.forced_test_mode = test_mode
        self.sent: list[str] = []  # test-mode ledger

    @property
    def test_mode(self) -> bool:
        return self.forced_test_mode or not (self.token and self.chat_id)

    @test_mode.setter
    def test_mode(self, value: bool) -> None:
        self.forced_test_mode = bool(value)

    @classmethod
    def from_config(cls) -> "TelegramAlerter":
        import os
        sec = _load_secrets()
        token = os.environ.get("TELEGRAM_BOT_TOKEN") or sec.get("TELEGRAM_BOT_TOKEN")
        chat = os.environ.get("TELEGRAM_CHAT_ID") or sec.get("TELEGRAM_CHAT_ID")
        return cls(token=token, chat_id=chat)

    def _reload_chat_id(self) -> None:
        """Pick up a chat_id persisted after this process started."""
        if not self.chat_id:
            self.chat_id = _load_secrets().get("TELEGRAM_CHAT_ID") or None

    # ------------------------------------------------------------------ #

    def send(self, text: str, parse_mode: str | None = None) -> bool:
        if not self.forced_test_mode:
            self._reload_chat_id()
        if self.test_mode:
            self.sent.append(text)
            logger.info("[telegram test-mode] %s", text)
            return True
        try:
            from telegram import Bot

            async def _go():
                await Bot(self.token).send_message(chat_id=self.chat_id,
                                                   text=text,
                                                   parse_mode=parse_mode)
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

    # ---- v4.0 trade lifecycle notifications ---------------------------- #

    def trade_opened(self, ticker: str, direction: str, entry_price: float,
                     qty: int, stop_loss: float, take_profit: float,
                     score: int | None = None, agent_votes: dict | None = None,
                     regime: str | None = None,
                     reasoning: str | None = None) -> bool:
        arrow = "📈" if direction == "LONG" else "📉"
        votes = (" | ".join(f"{k}:{v}" for k, v in agent_votes.items())
                 if agent_votes else None)
        lines = [
            f"{arrow} PAPER TRADE OPENED",
            f"{ticker} — {direction} ×{qty}",
            f"Entry:  ₹{entry_price:,.2f}",
            f"Stop:   ₹{stop_loss:,.2f} (-2%)",
            f"Target: ₹{take_profit:,.2f} (+4%)",
        ]
        if score is not None:
            lines.append(f"Score: {score}/12"
                         + (f" · Regime: {regime}" if regime else ""))
        if votes:
            lines.append(f"Votes: {votes}")
        if reasoning:
            lines.append(f"Why: {reasoning[:200]}")
        return self.send("\n".join(lines))

    def trade_closed(self, ticker: str, direction: str, entry_price: float,
                     exit_price: float, pnl: float, pnl_pct: float,
                     exit_reason: str, hold_days: int) -> bool:
        emoji = "✅" if pnl > 0 else "🔴"
        return self.send(
            f"{emoji} PAPER TRADE CLOSED\n"
            f"{ticker} — {direction}\n"
            f"Entry ₹{entry_price:,.2f} → Exit ₹{exit_price:,.2f}\n"
            f"P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2%})\n"
            f"Held: {hold_days} day(s) · Reason: {exit_reason}"
        )

    # Compatibility aliases (completion-loop prompt uses these names)
    def send_message(self, text: str) -> bool:
        return self.send(text)


# Back-compat name used by the completion-loop prompt:
#   from src.alerts.telegram_bot import TelegramBot
#   TelegramBot().send_message("...")
def TelegramBot() -> "TelegramAlerter":
    return TelegramAlerter.from_config()


HELP_TEXT = """🤖 QuNtra Command Guide
Your AI quantitative research organization

━━━━━━━━━━━━━━━━━━━━
📊 PORTFOLIO & POSITIONS
━━━━━━━━━━━━━━━━━━━━
/status — Full system snapshot (P&L, regime, positions, health)
/portfolio — Holdings breakdown with weights and sector exposure
/positions — Open positions with live mark-to-market P&L
/open_positions — Same as /positions
/trades — Today's executed trades with P&L and exit reasons
/signals — All signals generated today (executed + rejected)
/paper_progress — Paper trading gate status (X/40 days)
/gate_report — Full day-by-day trading history (auto-sent at day 40)
/obsidian — Regenerate the Obsidian knowledge vault from the database
/start_trading — Start today's paper session from your phone (no laptop)
/breeze_token — Refresh the daily ICICI Breeze session: /breeze_token <token>

━━━━━━━━━━━━━━━━━━━━
📈 PERFORMANCE
━━━━━━━━━━━━━━━━━━━━
/performance — Rolling Sharpe, win rate, max DD
/daily_report — Full today's report (triggers immediately)
/weekly_report — This week's board report
/monthly_report — This month's investment letter

━━━━━━━━━━━━━━━━━━━━
🔬 RESEARCH & INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━
/research — Latest pre-market research summary
/deep_screen — Tonight's 50-stock shortlist (entry/stop/target/hold); `/deep_screen now` recomputes
/regime — Current market regime + recent history
/watchlist — Today's watchlist (tickers scoring ≥9/12)
/macro — Current macro environment summary
/note <text> — Send an observation for QuNtra to verify and act on
/context — What QuNtra knows right now (regime, bias, key risks)
/chat <question> — Ask QuNtra anything about markets or portfolio

━━━━━━━━━━━━━━━━━━━━
⚠️ RISK MANAGEMENT
━━━━━━━━━━━━━━━━━━━━
/risk — Risk dashboard (DD, consecutive losses, limits)
/health — System health (DB, scheduler, APIs, last job run)
/pause — Pause new signals (keeps existing positions open)
/resume — Resume after pause or kill switch
/override <signal_id> — Manually approve a blocked signal

━━━━━━━━━━━━━━━━━━━━
🚨 EMERGENCY
━━━━━━━━━━━━━━━━━━━━
/halt — Emergency stop all trading (keeps positions open)
/emergency_stop — IMMEDIATE: halt + square ALL positions now
/stop_live — Gracefully stop live trading (paper continues)
/start_live — Initiate live trading (only after 40-day gate)

━━━━━━━━━━━━━━━━━━━━
ℹ️ SYSTEM
━━━━━━━━━━━━━━━━━━━━
/help — This guide
/start — Re-send this guide
/report — Quick EOD-style snapshot (legacy alias)

💡 Example /note usage:
  /note Oil prices spiking on Iran sanctions
  /note RBI likely to cut rates next meeting

QuNtra verifies each note against live data, scores its
relevance to your portfolio, and updates research bias.
━━━━━━━━━━━━━━━━━━━━
Capital preservation is the highest priority.
No live trading until the 40-day paper gate passes."""


def format_trade_details(trades: list[dict]) -> str:
    """Per-trade entry/exit/P&L/reason breakdown — the /trades command's
    formatter, shared with Hermes's automatic market-close push
    (run_post_market_sequence) so both read from one implementation."""
    if not trades:
        return "📭 No trades executed today."
    lines = ["📋 TODAY'S TRADES"]
    for t in trades:
        pnl = t.get("pnl")
        emoji = "⬜" if pnl is None else ("✅" if pnl > 0 else "🔴")
        exit_px = (f"₹{t['exit_price']:,.2f}" if t.get("exit_price")
                   else "open")
        pnl_str = f"₹{pnl:+,.2f}" if pnl is not None else "open"
        lines.append(
            f"{emoji} {t['ticker']} {t['direction']}\n"
            f"   Entry ₹{t.get('entry_price') or 0:,.2f} → {exit_px}\n"
            f"   Score {t.get('signal_score') or '?'}/12 · P&L {pnl_str}"
            + (f" · {t.get('exit_reason')}" if t.get("exit_reason")
               else ""))
    return "\n".join(lines)


class QuNtraTelegramBot:
    """Inbound command center: 22 commands. Wire to a HermesCoordinator.

    Every command: answers fast (no blocking network beyond one quote),
    logs itself to system_state, and never lets an exception reach the
    polling loop.
    """

    AUTHORIZED_USERS_KEY = "telegram_authorized_users"

    def __init__(self, hermes, alerter: TelegramAlerter | None = None,
                 db_url: str | None = None):
        self.hermes = hermes
        self.alerter = alerter or TelegramAlerter.from_config()
        self.db_url = db_url

    # ---- authorization (first-contact capture + whitelist) ------------ #

    def is_authorized(self, chat_id: int) -> bool:
        """True when chat_id is on the whitelist. Empty whitelist means
        nobody is authorized yet — the next message claims the bot."""
        state = self.hermes.get_system_state(self.AUTHORIZED_USERS_KEY) or {}
        return chat_id in [u["chat_id"] for u in state.get("users", [])]

    def handle_first_contact(self, chat_id: int, username: str | None = None,
                             first_name: str | None = None) -> str | None:
        """Process a non-command message.

        Returns the reply text, or None for a silent reject (unknown user
        after the bot is claimed — no reply reveals the bot exists).
        """
        state = self.hermes.get_system_state(self.AUTHORIZED_USERS_KEY) or {}
        users = state.get("users", [])
        if not users:
            users.append({
                "chat_id": chat_id,
                "username": username,
                "first_name": first_name,
                "authorized_at": datetime.now(IST).isoformat(),
            })
            self.hermes.set_system_state(self.AUTHORIZED_USERS_KEY,
                                         {"users": users})
            self._persist_chat_id_to_env(chat_id)
            self.alerter.chat_id = str(chat_id)
            logger.info("Telegram operator authorized: chat_id=%s user=%s",
                        chat_id, username)
            return (f"✅ QuNtra online. Welcome, {first_name or 'operator'}.\n"
                    f"Your chat ID {chat_id} has been authorized.\n\n"
                    + HELP_TEXT)
        if chat_id not in [u["chat_id"] for u in users]:
            logger.warning("Unauthorized Telegram contact from chat_id=%s "
                           "(user=%s) — silently ignored", chat_id, username)
            return None
        return "Already authorized. Use /help for commands."

    @staticmethod
    def _persist_chat_id_to_env(chat_id: int) -> None:
        """Write TELEGRAM_CHAT_ID to secrets.env so it survives restarts."""
        lines = SECRETS.read_text().splitlines() if SECRETS.exists() else []
        lines = [l for l in lines if not l.startswith("TELEGRAM_CHAT_ID=")]
        lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
        SECRETS.write_text("\n".join(lines) + "\n")

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
        last_job = self.hermes.get_system_state("last_job_run") or {}
        job_desc = (f"{last_job.get('name')} @ {last_job.get('at', '?')[:16]}"
                    + (" (ERROR)" if last_job.get("error") else "")
                    if last_job else "none recorded yet")
        lines = ["*SYSTEM HEALTH*"] + [
            f"{'✅' if v == 'OK' else '⚠️'} {k}: {v}"
            for k, v in checks.items()
        ] + [f"🕐 Last job: {job_desc}",
             "📄 Mode: PAPER TRADING (live capital: ₹0)"]
        return "\n".join(lines)

    def cmd_deep_screen(self, *args) -> str:
        """Tonight's 50-stock shortlist with entry/stop/target/hold.
        Reads the stored 20:00 IST run; `/deep_screen now` recomputes."""
        from src.research.deep_screen import format_report, run_screen
        if args and args[0].lower() == "now":
            res = run_screen()
            self.hermes.set_system_state("deep_screen", res)
        else:
            res = self.hermes.get_system_state("deep_screen")
            if not res:
                return ("No screen stored yet — it runs at 20:00 IST after "
                        "the day's trading. Send /deep_screen now to build "
                        "one immediately.")
        return format_report(res, limit=12)[:3800]

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
            return ("Usage: /chat <your question>\n"
                    "Example: /chat what did we learn about RELIANCE?")
        from src.agents.research import ResearchWriter
        return ResearchWriter(self.db_url).answer_question(
            question, {"regime": self.hermes.get_system_state("regime")})

    # ---- discovery + today views (v4.0) ------------------------------- #

    def cmd_help(self) -> str:
        return HELP_TEXT

    def cmd_start(self) -> str:
        """Alias for /help (first-contact welcome handled by the runner)."""
        return HELP_TEXT

    def cmd_trades(self) -> str:
        """Today's executed trades with entry/exit/P&L/reason."""
        return format_trade_details(self.hermes.brain.get_todays_trades())

    def cmd_signals(self) -> str:
        """All signals today: executed vs rejected with reasons."""
        signals = self.hermes.brain.get_todays_signals()
        if not signals:
            return "📡 No signals generated today yet."
        executed = [s for s in signals if s["executed"]]
        rejected = [s for s in signals if not s["executed"]]
        lines = [f"📡 TODAY'S SIGNALS — {len(executed)} executed, "
                 f"{len(rejected)} rejected/scored"]
        for s in signals[:10]:
            icon = "✅" if s["executed"] else "▫️"
            line = f"{icon} {s['ticker']} {s['direction'] or ''} · " \
                   f"score {s['score']}/12"
            if not s["executed"] and s.get("rejection_reason"):
                line += f" · {s['rejection_reason']}"
            lines.append(line)
        if len(signals) > 10:
            lines.append(f"… and {len(signals) - 10} more")
        return "\n".join(lines)

    REGIME_EMOJI = {
        "BULL_TRENDING": "🟢", "BULL_VOLATILE": "🟡", "SIDEWAYS": "⬜",
        "BEAR_TRENDING": "🔴", "BEAR_VOLATILE": "🟠", "CRISIS": "💀",
    }

    def cmd_regime(self) -> str:
        """Current market regime + recent history."""
        state = self.hermes.get_system_state("regime") or {}
        current = state.get("state") or state.get("current", "UNKNOWN")
        conf = state.get("confidence", 0)
        history = (state.get("history") or [])[-5:]
        lines = [f"📊 MARKET REGIME",
                 f"{self.REGIME_EMOJI.get(current, '⬜')} {current}"
                 + (f" (confidence {conf:.0%})" if conf else "")]
        if history:
            lines.append("Recent history:")
            for h in history:
                lines.append(f"  {h.get('date', '?')}: "
                             f"{self.REGIME_EMOJI.get(h.get('regime'), '⬜')} "
                             f"{h.get('regime', '?')}")
        else:
            lines.append("(no regime history yet — HMM refit is a Phase 3 "
                         "upgrade; regime defaults to UNKNOWN)")
        return "\n".join(lines)

    def cmd_paper_progress(self) -> str:
        """Compact paper-gate progress via paper_trading_status --telegram."""
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "scripts/paper_trading_status.py", "--telegram"],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip() or f"status script failed: {r.stderr[:200]}"

    def cmd_gate_report(self) -> str:
        """Full day-by-day paper trading history, on demand — works any
        day (not just after the 40-day mark is reached)."""
        return self.hermes.generate_gate_report_now()

    def cmd_obsidian(self) -> str:
        """Regenerate the Obsidian markdown vault from the database."""
        r = self.hermes.sync_obsidian()
        if "error" in r:
            return f"⚠️ Obsidian sync failed: {r['error']}"
        return ("🗂 Obsidian vault synced\n"
                f"  {r['daily']} daily notes · {r['tickers']} tickers · "
                f"{r['reports']} reports · {r['knowledge']} knowledge sets\n"
                f"  {r['vault']}\n"
                "Open that folder in Obsidian: Open folder as vault.")

    def cmd_start_trading(self) -> str:
        """Kick off today's paper session from the phone: run pre-market
        (build the watchlist) and arm the OMS. The scheduler's minute loop
        then executes — no laptop needed. Normally the 06:00 job does this
        automatically; this is the manual trigger / catch-up."""
        from scripts.scheduler import is_trading_day
        if not is_trading_day():
            return ("📴 Markets are closed today (weekend/holiday) — nothing "
                    "to start. The session begins automatically at 06:00 IST "
                    "on the next trading day.")
        if self.alerter is not None:
            try:
                self.alerter.send("⏳ Starting paper session — running "
                                  "pre-market research and scoring…")
            except Exception:  # noqa: BLE001
                pass
        res = self.hermes.run_pre_market_sequence()
        self.hermes.arm_system()
        watch = (self.hermes.get_system_state("premarket") or {}).get(
            "watchlist", [])
        oms = (self.hermes.get_system_state("oms") or {}).get("enabled")
        if not watch:
            return ("⚠️ Session armed but the watchlist is empty — nothing "
                    "scored ≥9/12 today, so no trades will fire. OMS "
                    f"enabled: {oms}.")
        return (f"✅ Paper session started · OMS armed ({oms})\n"
                f"Watchlist ({len(watch)}): {', '.join(watch[:8])}"
                + ("…" if len(watch) > 8 else "") + "\n"
                "The trading loop will execute up to 3 trades on its next "
                "minute tick. Watch /open_positions.")

    def cmd_breeze_token(self, *args) -> str:
        """Refresh the daily ICICI Breeze session token from the phone.
        Send the apisession value from the login redirect URL:
        /breeze_token <session_token>."""
        token = (args[0] if args else "").strip()
        if not token:
            import os

            from src.integrations.breeze_session import login_url
            api_key = os.getenv("ICICI_BREEZE_API_KEY", "")
            url = login_url(api_key) if api_key else "(set ICICI_BREEZE_API_KEY first)"
            return ("Usage: /breeze_token <session_token>\n\n"
                    "1. Open this login URL, approve:\n" + url + "\n"
                    "2. Copy the apisession= value from the redirect URL\n"
                    "   (e.g. http://127.0.0.1:.../?apisession=12345678)\n"
                    "3. Send: /breeze_token <that value>")
        from src.integrations.breeze_session import set_token, token_status
        try:
            set_token(token)
        except Exception as e:  # noqa: BLE001
            return (f"⚠️ Couldn't use that token: {str(e)[:130]}\n\n"
                    "Send the *apisession* value from the login redirect "
                    "URL, right after logging in. Send /breeze_token with "
                    "no argument to get a fresh login link.")
        status = token_status()
        return (f"✅ Breeze session updated ({token[:6]}…). Status: "
                f"{status}.\nReal-time quotes should now be flowing "
                "(falls back to delayed yfinance automatically if not).")

    def cmd_macro(self) -> str:
        """Latest macro snapshot from the macro agent's stored research."""
        from src.reporting import metrics as M
        pre = self.hermes.get_system_state("premarket") or {}
        notes = [n for n in M.research_notes_since(self.db_url, hours=36)
                 if n["source"] == "macro_agent"]
        lines = ["🌍 MACRO ENVIRONMENT",
                 f"Bias: {pre.get('macro_bias', 'UNKNOWN')}"]
        cues = pre.get("global_cues") or {}
        for name, move in cues.items():
            if isinstance(move, (int, float)):
                lines.append(f"  {name}: {move:+.2%}")
        if notes:
            lines.append(f"Latest: {notes[0]['summary']}")
        return "\n".join(lines)

    def cmd_positions(self) -> str:
        """Alias for /open_positions."""
        return self.cmd_open_positions()

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
        # v4.0: discovery + today views
        "help", "start", "trades", "signals", "regime", "paper_progress",
        "macro", "positions",
        # gate completion: full day-by-day history, on demand or auto-sent
        "gate_report",
        # obsidian: regenerate the markdown vault from the DB
        "obsidian",
        # phone-run controls: start the session, refresh the Breeze session
        "start_trading", "breeze_token",
        # nightly 50-stock Nifty-200 screen -> tomorrow's plan
        "deep_screen",
    ]

    def run_polling(self):
        """Long-poll Telegram. Needs only the TOKEN — the chat_id is
        captured from the operator's first message (handle_first_contact)."""
        from telegram.ext import (ApplicationBuilder, CommandHandler,
                                  MessageHandler, filters)

        if not self.alerter.token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN missing in config/secrets.env — "
                "cannot start polling.")

        def make_handler(name):
            async def handler(update, context):
                chat_id = update.effective_chat.id
                if name == "start" and not self.is_authorized(chat_id):
                    # /start doubles as the claim handshake
                    reply = self.handle_first_contact(
                        chat_id,
                        username=getattr(update.effective_user,
                                         "username", None),
                        first_name=getattr(update.effective_user,
                                           "first_name", None))
                    if reply:
                        await update.message.reply_text(reply[:4000])
                    return
                if not self.is_authorized(chat_id):
                    # Stay silent to the sender (an unknown chat must not
                    # learn the bot exists), but ALWAYS log it. Dropping a
                    # command with no trace made a real outage nearly
                    # undiagnosable: after the Render migration the whitelist
                    # lived in a fresh empty DB, so every operator command
                    # vanished with zero evidence anywhere.
                    logger.warning(
                        "Ignoring /%s from unauthorized chat_id=%s — "
                        "whitelist has %d user(s). Send /start to claim the "
                        "bot if this is the operator on a new database.",
                        name, chat_id,
                        len((self.hermes.get_system_state(
                            self.AUTHORIZED_USERS_KEY) or {}).get("users", [])))
                    return
                args = context.args or []
                # dispatch() logs the call and absorbs all errors
                reply = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.dispatch(name, *args))
                for chunk in (reply[i:i + 4000]
                              for i in range(0, len(reply), 4000)):
                    await update.message.reply_text(chunk)
            return handler

        async def on_message(update, context):
            reply = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.handle_first_contact(
                    update.effective_chat.id,
                    username=getattr(update.effective_user, "username", None),
                    first_name=getattr(update.effective_user,
                                       "first_name", None)))
            if reply:  # None = silent reject
                await update.message.reply_text(reply[:4000])

        app = ApplicationBuilder().token(self.alerter.token).build()
        for name in self.COMMANDS:
            app.add_handler(CommandHandler(name, make_handler(name)))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                       on_message))

        async def on_error(update, context):
            """Without a registered handler, python-telegram-bot logs
            'No error handlers are registered' and the polling loop can
            stop for good while the process stays alive — heartbeats keep
            firing, so nothing looks broken. That is exactly what happened
            on 2026-08-05 01:11 UTC: a transient
            telegram.error.NetworkError 'Bad Gateway' killed polling at
            06:41 IST, so every operator command (including the daily
            /breeze_token) was silently ignored for the rest of the day.
            Swallowing the error here keeps the loop retrying.
            """
            logger.error("Telegram update failed (polling continues): %s",
                         context.error)

        app.add_error_handler(on_error)
        logger.info("Telegram bot polling started (%d commands)",
                    len(self.COMMANDS))
        app.run_polling()
