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


class QuNtraTelegramBot:
    """Inbound command handler. Wire to a HermesCoordinator + guards."""

    def __init__(self, hermes, alerter: TelegramAlerter | None = None):
        self.hermes = hermes
        self.alerter = alerter or TelegramAlerter.from_config()

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

    # Polling entrypoint (needs a real token) --------------------------- #

    def run_polling(self):
        from telegram.ext import ApplicationBuilder, CommandHandler

        if self.alerter.test_mode:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in "
                "config/secrets.env — cannot start polling."
            )

        async def wrap(fn):
            async def handler(update, context):
                args = context.args if context.args else []
                reply = fn(*args) if args else fn()
                await update.message.reply_text(reply)
            return handler

        app = ApplicationBuilder().token(self.alerter.token).build()
        loop = asyncio.new_event_loop()
        for name, fn in [("status", self.cmd_status), ("pause", self.cmd_pause),
                         ("resume", self.cmd_resume), ("report", self.cmd_report),
                         ("override", self.cmd_override), ("halt", self.cmd_halt)]:
            app.add_handler(CommandHandler(name, loop.run_until_complete(wrap(fn))))
        logger.info("Telegram bot polling started")
        app.run_polling()
