#!/usr/bin/env python3
"""
Weekly paper-trading gate report (Step 8 / 40-day gate).

Reads paper trades + signals from the DB and prints progress against
the gate: 40 trading days, Sharpe > 1.0, MaxDD better than -15%,
kill-switch handling, uptime proxy.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from src.db import Signal, Trade, ResearchNote, get_session, init_db  # noqa: E402

GATE_DAYS = 40
GATE_SHARPE = 1.0
GATE_MAX_DD = -0.15
RISK_FREE = 0.07


def main() -> int:
    init_db()
    with get_session() as s:
        trades = s.execute(
            select(Trade).where(Trade.is_paper.is_(True))
            .order_by(Trade.created_at)
        ).scalars().all()
        n_signals = len(s.execute(select(Signal)).scalars().all())
        n_executed = len(s.execute(
            select(Signal).where(Signal.executed.is_(True))).scalars().all())
        halts = s.execute(
            select(ResearchNote).where(ResearchNote.note_type == "lesson")
        ).scalars().all()
        halt_events = [h for h in halts if "consecutive losses" in (h.summary or "")]

    closed = [t for t in trades if t.pnl is not None]
    if not closed:
        print("No closed paper trades yet — system may have just started.")
        print(f"Signals generated so far: {n_signals} ({n_executed} executed)")
        return 0

    df = pd.DataFrame({
        "date": [t.exit_time or t.created_at for t in closed],
        "pnl": [float(t.pnl) for t in closed],
    })
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    daily = df.groupby("date")["pnl"].sum()

    capital = 25_000.0
    daily_ret = daily / capital
    days = len(daily)
    equity = (1 + daily_ret).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min())
    if len(daily_ret) >= 2 and daily_ret.std() > 0:
        ann_ret = (1 + daily_ret.mean()) ** 252 - 1
        sharpe = (ann_ret - RISK_FREE) / (daily_ret.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    rolling30 = daily_ret.tail(30)
    sharpe30 = 0.0
    if len(rolling30) >= 10 and rolling30.std() > 0:
        sharpe30 = (((1 + rolling30.mean()) ** 252 - 1) - RISK_FREE) / \
                   (rolling30.std() * np.sqrt(252))

    print("=" * 52)
    print(f"QuNtra paper report — {datetime.now(timezone.utc).date()}")
    print("=" * 52)
    print(f"Days elapsed          : {days}/{GATE_DAYS}")
    print(f"Total paper P&L       : ₹{daily.sum():+,.0f}")
    print(f"Sharpe (all days)     : {sharpe:.2f}")
    print(f"Sharpe (rolling 30d)  : {sharpe30:.2f}")
    print(f"Max DD reached        : {max_dd:+.1%}")
    print(f"Signals generated     : {n_signals}")
    print(f"Trades executed       : {len(closed)}")
    print(f"Consecutive-loss halts: {len(halt_events)}")
    print("-" * 52)

    checks = {
        f"{GATE_DAYS} trading days": days >= GATE_DAYS,
        "Sharpe > 1.0": sharpe > GATE_SHARPE,
        "Max DD better than -15%": max_dd > GATE_MAX_DD,
    }
    for name, ok in checks.items():
        print(f"[{'x' if ok else ' '}] {name}")
    print("[?] Zero unrecovered crashes — verify logs/quntra_paper.log")
    print("[?] Kill-switch handled correctly — review halt lessons above")

    if all(checks.values()):
        print("\nPAPER GATE PASSED — authorized for ₹10,000 live capital deployment")
        return 0
    print(f"\nGate not yet passed — keep running ({days}/{GATE_DAYS} days).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
