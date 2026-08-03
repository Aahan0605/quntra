#!/usr/bin/env python3
"""Generate STATUS.md from the database. Never hand-maintain it again.

The old STATUS.md claimed the knowledge base was "9+ and growing nightly"
while `knowledge_items` held 0 rows, and claimed price data was in Postgres
while `price_data` held 0 rows. A status file that cannot be trusted is
worse than none: it is why a wedged scheduler read as healthy for five days.

Every number below is read live. If a table is empty, it says so.

    ./venv/bin/python scripts/generate_status.py          # print
    ./venv/bin/python scripts/generate_status.py --write  # overwrite STATUS.md
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEARTBEAT_MAX_AGE = 300


def _pid_state(name: str) -> str:
    pid_file = ROOT / name
    if not pid_file.exists():
        return "DOWN (no pidfile)"
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        return "DOWN (corrupt pidfile)"
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                          capture_output=True, text=True).stdout.strip()
    if not stat or stat.startswith("Z"):
        return f"DOWN (pid {pid} gone)"
    return f"UP (pid {pid})"


def _heartbeat() -> str:
    hb = ROOT / "quntra.heartbeat"
    if not hb.exists():
        return "absent — scheduler predates heartbeat, or never started"
    try:
        age = int(time.time()) - int(hb.read_text().strip())
    except (ValueError, OSError):
        return "unreadable"
    verdict = "OK" if age <= HEARTBEAT_MAX_AGE else "**STALE — SCHEDULER HUNG**"
    return f"{age}s old — {verdict}"


def _table_counts() -> list[tuple[str, int]]:
    from sqlalchemy import func, select

    from src.db import (AgentCredibility, KnowledgeItem, ResearchNote, Signal,
                        Trade, get_session)
    tables = [("trades", Trade), ("signals", Signal),
              ("research_notes", ResearchNote),
              ("knowledge_items", KnowledgeItem),
              ("agent_credibility", AgentCredibility)]
    out = []
    with get_session() as s:
        for label, model in tables:
            try:
                out.append((label, s.execute(
                    select(func.count()).select_from(model)).scalar_one()))
            except Exception as e:  # noqa: BLE001
                out.append((label, -1))
    return out


def _trade_stats() -> dict:
    from sqlalchemy import select

    from src.db import Trade, get_session
    with get_session() as s:
        rows = s.execute(select(Trade)).scalars().all()
    closed = [r for r in rows if r.exit_time is not None]
    open_ = [r for r in rows if r.exit_time is None]
    wins = [r for r in closed if r.pnl is not None and float(r.pnl) > 0]
    pnl = sum(float(r.pnl) for r in closed if r.pnl is not None)
    days = {r.entry_time.date() for r in rows if r.entry_time}
    return {"total": len(rows), "closed": len(closed), "open": len(open_),
            "wins": len(wins), "pnl": pnl, "trading_days": len(days)}


def build() -> str:
    now = datetime.now(timezone.utc).astimezone()
    L = [f"# QuNtra — live status",
         "",
         f"Generated {now:%Y-%m-%d %H:%M %Z} by `scripts/generate_status.py`.",
         "**Do not edit by hand** — regenerate it.",
         "", "## Processes", ""]
    for label, f in [("scheduler", "quntra.pid"),
                     ("telegram bot", "telegram_bot.pid"),
                     ("watchdog", "watchdog.pid")]:
        L.append(f"- {label}: {_pid_state(f)}")
    L += [f"- scheduler heartbeat: {_heartbeat()}", "", "## Database", ""]
    try:
        for name, n in _table_counts():
            note = " — **EMPTY**" if n == 0 else (" — unreadable" if n < 0 else "")
            L.append(f"- `{name}`: {n if n >= 0 else '?'} rows{note}")
    except Exception as e:  # noqa: BLE001
        L.append(f"- database unreachable: {e}")
    L += ["", "## Paper trading", ""]
    try:
        st = _trade_stats()
        wr = f"{st['wins'] / st['closed']:.0%}" if st["closed"] else "n/a"
        L += [f"- trading days with activity: {st['trading_days']} / 40",
              f"- trades: {st['total']} total, {st['closed']} closed, "
              f"{st['open']} open",
              f"- win rate: {wr} ({st['wins']}W / "
              f"{st['closed'] - st['wins']}L)",
              f"- realised P&L: Rs {st['pnl']:,.2f}"]
    except Exception as e:  # noqa: BLE001
        L.append(f"- unavailable: {e}")

    audit = ROOT / "data" / "models" / "multiple_testing.json"
    L += ["", "## Model validity", ""]
    if audit.exists():
        import json
        r = json.loads(audit.read_text())
        L += [f"- trials: {r['n_trials']}, naive gate passes: "
              f"{r['n_passed_naive_gate']}, "
              f"**survive FDR: {r['n_survives_fdr']}**"]
        if r["n_survives_fdr"] == 0:
            L.append("- no model's edge is distinguishable from luck; "
                     "all ML votes are NEUTRAL")
    else:
        L.append("- not audited — run `python -m src.ml.multiple_testing`")
    L += ["", "---", "",
          "Trading mode: "
          + ("PAPER" if "PAPER_TRADE=true" in (ROOT / ".env").read_text()
             else "**CHECK .env**"),
          ""]
    return "\n".join(L)


def main() -> int:
    text = build()
    if "--write" in sys.argv:
        (ROOT / "STATUS.md").write_text(text)
        print("wrote STATUS.md")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
