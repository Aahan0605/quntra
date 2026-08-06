#!/usr/bin/env python3
"""
Paper trading status dashboard — run anytime.

    python3 scripts/paper_trading_status.py             # full terminal view
    python3 scripts/paper_trading_status.py --telegram  # compact for the bot

Shows: days elapsed vs 40-day gate, P&L, win rate, rolling Sharpe (rf=0,
same convention as the Phase-0 gate), max drawdown, scheduler health,
agent credibility bars, and the gate checklist.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import select  # noqa: E402

load_dotenv(ROOT / ".env")
from src.db import Trade, get_session, init_db  # noqa: E402
from src.governor.brain import QuNtraBrain  # noqa: E402

GATE_DAYS = 40
CAPITAL = float(os.environ.get("DAILY_CAPITAL_INR", "25000"))
AGENTS = ["valuation", "technical", "sentiment", "risk", "macro", "quantum"]


def check_scheduler_health() -> str:
    """Is the scheduler process actually alive?"""
    pid_file = ROOT / "quntra.pid"
    if not pid_file.exists():
        return "⚠️ SCHEDULER NOT RUNNING (quntra.pid missing)"
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return f"✅ Scheduler running (PID {pid})"
    except (ValueError, ProcessLookupError, PermissionError):
        return "🔴 SCHEDULER DEAD (stale quntra.pid) — run watchdog.py"


def gather_stats() -> dict | None:
    """All dashboard numbers from the DB. None when no trades exist."""
    with get_session() as s:
        rows = s.execute(
            select(Trade).where(Trade.is_paper.is_(True))
            .order_by(Trade.entry_time)
        ).scalars().all()
    if not rows:
        return None

    df = pd.DataFrame([{
        "entry_time": t.entry_time or t.created_at,
        "pnl": float(t.pnl) if t.pnl is not None else np.nan,
    } for t in rows])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    closed = df.dropna(subset=["pnl"])

    daily_pnl = closed.groupby(closed["entry_time"].dt.date)["pnl"].sum()
    daily_ret = daily_pnl / CAPITAL
    sharpe = float("nan")
    r30 = daily_ret.tail(30)
    if len(r30) > 5 and r30.std() > 0:
        # rf=0 — identical convention to the Phase-0 validation gate
        sharpe = (r30.mean() * 252) / (r30.std() * np.sqrt(252))
    cum = (1 + daily_ret).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min()) if len(cum) \
        else 0.0

    winning = int((closed["pnl"] > 0).sum())
    losing = int((closed["pnl"] < 0).sum())
    days = df["entry_time"].dt.date.nunique()
    return {
        "days": days,
        "total_pnl": float(closed["pnl"].sum()),
        "wins": winning,
        "losses": losing,
        "win_rate": winning / len(closed) if len(closed) else 0.0,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_closed": len(closed),
        "n_entered": len(df),
        "gate_days": days >= GATE_DAYS,
        "gate_sharpe": (not np.isnan(sharpe)) and sharpe > 1.0,
        "gate_dd": max_dd > -0.15,
    }


def telegram_output(st: dict | None) -> str:
    if st is None:
        return ("📊 Paper Trading Progress\n"
                "No trades yet — day 0/40.\n"
                + check_scheduler_health())
    sharpe_str = ("n/a ⏳" if np.isnan(st["sharpe"]) else
                  f"{st['sharpe']:.3f} "
                  + ("✅" if st["gate_sharpe"] else "⏳"))
    all_pass = st["gate_days"] and st["gate_sharpe"] and st["gate_dd"]
    return (
        f"📊 Paper Trading Progress\n"
        f"Day {st['days']}/{GATE_DAYS} | "
        f"{max(0, GATE_DAYS - st['days'])} remaining\n\n"
        f"Sharpe (30d, rf=0): {sharpe_str}\n"
        f"Max DD: {st['max_dd']:+.2%} "
        f"{'✅' if st['gate_dd'] else '⏳'}\n"
        f"Win rate: {st['win_rate']:.1%} "
        f"({st['wins']}W / {st['losses']}L)\n"
        f"P&L: ₹{st['total_pnl']:+,.0f} · "
        f"{st['n_closed']} closed / {st['n_entered']} entered\n\n"
        f"{check_scheduler_health()}\n"
        f"Gate: {'🟢 ALL PASS' if all_pass else '⏳ IN PROGRESS'}"
    )


def terminal_output(st: dict | None, brain: QuNtraBrain) -> int:
    print("QuNtra Paper Trading Status")
    print("=" * 50)
    print(check_scheduler_health())
    if st is None:
        print("No paper trades yet. Scheduler may not have started.")
        return 0

    print(f"Trading days elapsed:  {st['days']} / {GATE_DAYS} required")
    print(f"Total paper P&L:       ₹{st['total_pnl']:+,.0f}")
    print(f"Win rate:              {st['win_rate']:.1%} "
          f"({st['wins']}W / {st['losses']}L)")
    print(f"Rolling 30d Sharpe:    "
          + (f"{st['sharpe']:.4f}  (rf=0)" if not np.isnan(st["sharpe"])
             else "n/a (need > 5 days)"))
    print(f"Max drawdown:          {st['max_dd']:+.4f}  (target > -0.15)")
    print(f"Total trades:          {st['n_closed']} closed / "
          f"{st['n_entered']} entered")

    print("\nAgent credibility:")
    for agent in AGENTS:
        w = brain.get_agent_credibility(agent)
        filled = max(0, min(20, int(w * 10)))
        print(f"  {agent:12s}: {w:.4f} [{'▓' * filled}{'░' * (20 - filled)}]")

    print("\n" + "=" * 50)
    print("PAPER TRADING GATE STATUS:")
    print(f"  {GATE_DAYS} trading days:  "
          + ("PASS ✓" if st["gate_days"]
             else f"PENDING ({st['days']}/{GATE_DAYS})"))
    print(f"  Sharpe > 1.0:     "
          + ("PASS ✓" if st["gate_sharpe"] else "PENDING"))
    print(f"  Max DD > -15%:    "
          + ("PASS ✓" if st["gate_dd"]
             else f"PENDING ({st['max_dd']:+.4f})"))

    if st["gate_days"] and st["gate_sharpe"] and st["gate_dd"]:
        print("\n✅ ALL GATES PASSED — authorized for live capital deployment")
        print("(Also verify: zero unrecovered crashes + kill-switch fired and")
        print(" recovered correctly at least once — check logs and lessons.)")
        return 0
    print(f"\n⏳ Continue paper trading "
          f"({max(0, GATE_DAYS - st['days'])} days remaining)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true",
                    help="compact Telegram-formatted output")
    args = ap.parse_args()

    init_db()
    st = gather_stats()
    if args.telegram:
        print(telegram_output(st))
        return 0
    return terminal_output(st, QuNtraBrain())


if __name__ == "__main__":
    sys.exit(main())
