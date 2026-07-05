#!/usr/bin/env python3
"""
Paper trading status dashboard (Task R9) — run anytime.

Shows: days elapsed vs 40-day gate, P&L, win rate, rolling Sharpe,
max drawdown, agent credibility bars, and a clear gate checklist.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from src.db import Trade, get_session, init_db  # noqa: E402
from src.governor.brain import QuNtraBrain  # noqa: E402

GATE_DAYS = 40
CAPITAL = 25_000.0
RISK_FREE = 0.07
AGENTS = ["valuation", "technical", "sentiment", "risk", "macro", "quantum"]


def main() -> int:
    init_db()
    brain = QuNtraBrain()

    print("QuNtra Paper Trading Status")
    print("=" * 50)

    with get_session() as s:
        rows = s.execute(
            select(Trade).where(Trade.is_paper.is_(True))
            .order_by(Trade.entry_time)
        ).scalars().all()

    if not rows:
        print("No paper trades yet. Scheduler may not have started.")
        return 0

    df = pd.DataFrame([{
        "entry_time": t.entry_time or t.created_at,
        "pnl": float(t.pnl) if t.pnl is not None else np.nan,
    } for t in rows])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    closed = df.dropna(subset=["pnl"])

    trading_days = df["entry_time"].dt.date.nunique()
    total_pnl = closed["pnl"].sum()
    winning = int((closed["pnl"] > 0).sum())
    losing = int((closed["pnl"] < 0).sum())
    win_rate = winning / len(closed) if len(closed) else 0.0

    daily_pnl = closed.groupby(closed["entry_time"].dt.date)["pnl"].sum()
    daily_ret = daily_pnl / CAPITAL
    sharpe_30d = float("nan")
    r30 = daily_ret.tail(30)
    if len(r30) > 5 and r30.std() > 0:
        sharpe_30d = (((1 + r30.mean()) ** 252 - 1) - RISK_FREE) / \
                     (r30.std() * np.sqrt(252))
    cum = (1 + daily_ret).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min()) if len(cum) else 0.0

    print(f"Trading days elapsed:  {trading_days} / {GATE_DAYS} required")
    print(f"Total paper P&L:       ₹{total_pnl:+,.0f}")
    print(f"Win rate:              {win_rate:.1%} ({winning}W / {losing}L)")
    print(f"Rolling 30d Sharpe:    "
          f"{sharpe_30d:.4f}" if not np.isnan(sharpe_30d)
          else "Rolling 30d Sharpe:    n/a (need > 5 days)")
    print(f"Max drawdown:          {max_dd:+.4f}  (target > -0.15)")
    print(f"Total trades:          {len(closed)} closed / {len(df)} entered")

    print("\nAgent credibility:")
    for agent in AGENTS:
        w = brain.get_agent_credibility(agent)
        filled = max(0, min(20, int(w * 10)))
        bar = "▓" * filled + "░" * (20 - filled)
        print(f"  {agent:12s}: {w:.4f} [{bar}]")

    print("\n" + "=" * 50)
    print("PAPER TRADING GATE STATUS:")
    gate_days = trading_days >= GATE_DAYS
    gate_sharpe = (not np.isnan(sharpe_30d)) and sharpe_30d > 1.0
    gate_dd = max_dd > -0.15
    print(f"  {GATE_DAYS} trading days:  "
          f"{'PASS ✓' if gate_days else f'PENDING ({trading_days}/{GATE_DAYS})'}")
    print(f"  Sharpe > 1.0:     "
          f"{'PASS ✓' if gate_sharpe else 'PENDING'}")
    print(f"  Max DD > -15%:    "
          f"{'PASS ✓' if gate_dd else f'PENDING ({max_dd:+.4f})'}")

    if gate_days and gate_sharpe and gate_dd:
        print("\n✅ ALL GATES PASSED — authorized for live capital deployment")
        print("(Also verify: zero unrecovered crashes + kill-switch fired and")
        print(" recovered correctly at least once — check logs and lessons.)")
        return 0
    print(f"\n⏳ Continue paper trading "
          f"({max(0, GATE_DAYS - trading_days)} days remaining)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
