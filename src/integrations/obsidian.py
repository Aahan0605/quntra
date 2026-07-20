"""
Obsidian export — mirror QuNtra's database into a linked markdown vault.

QuNtra keeps everything (trades, signals, research notes, knowledge,
reports) in PostgreSQL/SQLite. This module renders that into an Obsidian
vault of plain markdown files, cross-linked with [[wikilinks]] so the
operator can browse the whole organisation's memory — daily notes,
per-ticker histories, and the report archive — in Obsidian's graph view.

Design notes:
  * Read-only against the DB; never writes back. A missing DB just yields
    an empty vault, never an exception that could disturb trading.
  * sync() is idempotent — it regenerates the derived notes each run, so
    it's safe to call from a daily scheduler job or on demand.
  * No new dependencies: stdlib + the SQLAlchemy models already in use.
  * Obsidian itself is free (local vault, filesystem-synced) — the vault
    is just a folder of .md files; point Obsidian at OBSIDIAN_VAULT_DIR.

Vault layout:
    <vault>/
      Home.md                     index + live gate status
      Daily/YYYY-MM-DD.md         one note per active trading day
      Tickers/<TICKER>.md         per-ticker trade + research history
      Reports/<report>.md         archived daily/weekly/monthly reports
      Knowledge/<TYPE>.md         organisational lessons by type
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from src.db import (
    KnowledgeItem,
    ResearchNote,
    Signal,
    SystemState,
    Trade,
    get_session,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAULT = ROOT / "data" / "obsidian_vault"
REPORTS_DIR = ROOT / "data" / "reports"

_SAFE = re.compile(r"[^A-Za-z0-9._ -]")


def _slug(name: str) -> str:
    """Filename-safe note title (Obsidian links resolve on the basename)."""
    return _SAFE.sub("_", name).strip() or "untitled"


def _day(dt: datetime | None) -> str:
    if dt is None:
        return "unknown-date"
    return dt.date().isoformat()


class ObsidianVault:
    def __init__(self, vault_dir: str | Path | None = None,
                 db_url: str | None = None):
        self.vault = Path(vault_dir or os.getenv("OBSIDIAN_VAULT_DIR")
                          or DEFAULT_VAULT)
        self.db_url = db_url

    # ------------------------------------------------------------------ #

    def sync(self) -> dict:
        """Regenerate the whole vault from the DB. Returns file counts."""
        for sub in ("Daily", "Tickers", "Reports", "Knowledge"):
            (self.vault / sub).mkdir(parents=True, exist_ok=True)

        trades = self._load(Trade, Trade.is_paper.is_(True))
        signals = self._load(Signal)
        notes = self._load(ResearchNote)
        knowledge = self._load(KnowledgeItem)

        counts = {
            "daily": self._write_daily(trades, signals, notes),
            "tickers": self._write_tickers(trades, notes, knowledge),
            "reports": self._write_reports(),
            "knowledge": self._write_knowledge(knowledge),
        }
        self._write_home(counts, len(trades))
        counts["vault"] = str(self.vault)
        return counts

    def _load(self, model, *where):
        try:
            with get_session(self.db_url) as s:
                stmt = select(model)
                for w in where:
                    stmt = stmt.where(w)
                rows = s.execute(stmt).scalars().all()
                s.expunge_all()
                return rows
        except Exception:  # noqa: BLE001 — a DB hiccup must not raise here
            return []

    # ------------------------------------------------------------------ #
    # Daily notes

    def _write_daily(self, trades, signals, notes) -> int:
        by_day: dict[str, dict] = defaultdict(
            lambda: {"trades": [], "signals": [], "notes": []})
        for t in trades:
            by_day[_day(t.entry_time or t.created_at)]["trades"].append(t)
        for sg in signals:
            by_day[_day(sg.signal_time or sg.created_at)]["signals"].append(sg)
        for n in notes:
            by_day[_day(n.created_at)]["notes"].append(n)

        for day, data in by_day.items():
            if day == "unknown-date":
                continue
            self._daily_note(day, data)
        return len([d for d in by_day if d != "unknown-date"])

    def _daily_note(self, day: str, data: dict) -> None:
        trades, signals, notes = (data["trades"], data["signals"],
                                  data["notes"])
        closed = [t for t in trades if t.pnl is not None]
        day_pnl = sum(float(t.pnl) for t in closed)
        wins = sum(1 for t in closed if float(t.pnl) > 0)
        regime = next((t.regime for t in trades if t.regime), None) \
            or next((sg.regime for sg in signals if sg.regime), "UNKNOWN")

        lines = [
            "---",
            f"date: {day}",
            "type: daily-note",
            f"regime: {regime}",
            f"trades: {len(trades)}",
            f"net_pnl: {day_pnl:.2f}",
            "tags: [quntra, daily]",
            "---",
            f"# {day} — Trading Day",
            "",
            f"**Regime:** {regime}  ·  **Trades:** {len(trades)}  ·  "
            f"**Net P&L:** ₹{day_pnl:+,.0f}  ·  **Win rate:** "
            + (f"{wins}/{len(closed)}" if closed else "n/a"),
            "",
        ]
        if trades:
            lines += ["## Trades", "",
                      "| Ticker | Dir | Entry | Exit | P&L | Reason |",
                      "|---|---|--:|--:|--:|---|"]
            for t in trades:
                exit_px = f"₹{float(t.exit_price):,.2f}" if t.exit_price \
                    else "open"
                pnl = f"₹{float(t.pnl):+,.0f}" if t.pnl is not None else "—"
                lines.append(
                    f"| [[{_slug(t.ticker)}]] | {t.direction} | "
                    f"₹{float(t.entry_price or 0):,.2f} | {exit_px} | "
                    f"{pnl} | {t.exit_reason or 'open'} |")
            lines.append("")
        if notes:
            lines += ["## Research", ""]
            for n in sorted(notes, key=lambda x: x.source or ""):
                tick = f" [[{_slug(n.ticker)}]]" if n.ticker else ""
                lines.append(f"- **[{n.source or '?'}]**{tick} "
                             f"{(n.summary or n.content or '')[:200]}")
            lines.append("")
        if signals and not trades:
            lines += ["## Signals (none executed)", ""]
            for sg in signals[:15]:
                lines.append(f"- {sg.ticker} {sg.direction or ''} "
                             f"score {sg.score}/12"
                             + (f" — {sg.rejection_reason}"
                                if sg.rejection_reason else ""))
            lines.append("")
        (self.vault / "Daily" / f"{day}.md").write_text("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Per-ticker notes

    def _write_tickers(self, trades, notes, knowledge) -> int:
        tickers: dict[str, dict] = defaultdict(
            lambda: {"trades": [], "notes": [], "knowledge": []})
        for t in trades:
            tickers[t.ticker]["trades"].append(t)
        for n in notes:
            if n.ticker:
                tickers[n.ticker]["notes"].append(n)
        for k in knowledge:
            for tk in (k.tickers or []):
                tickers[tk]["knowledge"].append(k)

        for ticker, data in tickers.items():
            self._ticker_note(ticker, data)
        return len(tickers)

    def _ticker_note(self, ticker: str, data: dict) -> None:
        trades = data["trades"]
        closed = [t for t in trades if t.pnl is not None]
        total_pnl = sum(float(t.pnl) for t in closed)
        wins = sum(1 for t in closed if float(t.pnl) > 0)
        lines = [
            "---",
            f"ticker: {ticker}",
            "type: ticker",
            f"total_trades: {len(trades)}",
            f"net_pnl: {total_pnl:.2f}",
            "tags: [quntra, ticker]",
            "---",
            f"# {ticker}",
            "",
            f"**Trades:** {len(trades)}  ·  **Net P&L:** ₹{total_pnl:+,.0f}"
            + (f"  ·  **Win rate:** {wins}/{len(closed)}" if closed else ""),
            "",
        ]
        if trades:
            lines += ["## Trade history", "",
                      "| Day | Dir | Entry | Exit | P&L | Reason |",
                      "|---|---|--:|--:|--:|---|"]
            for t in sorted(trades, key=lambda x: _day(x.entry_time)):
                d = _day(t.entry_time or t.created_at)
                exit_px = f"₹{float(t.exit_price):,.2f}" if t.exit_price \
                    else "open"
                pnl = f"₹{float(t.pnl):+,.0f}" if t.pnl is not None else "—"
                lines.append(f"| [[{d}]] | {t.direction} | "
                             f"₹{float(t.entry_price or 0):,.2f} | {exit_px} | "
                             f"{pnl} | {t.exit_reason or 'open'} |")
            lines.append("")
        if data["knowledge"]:
            lines += ["## Lessons learned", ""]
            for k in data["knowledge"]:
                lines.append(f"- *[{k.knowledge_type}]* {k.content[:220]}")
            lines.append("")
        if data["notes"]:
            lines += ["## Research mentions", ""]
            for n in sorted(data["notes"], key=lambda x: x.created_at,
                            reverse=True)[:20]:
                lines.append(f"- [[{_day(n.created_at)}]] **[{n.source}]** "
                             f"{(n.summary or n.content or '')[:180]}")
            lines.append("")
        (self.vault / "Tickers" / f"{_slug(ticker)}.md").write_text(
            "\n".join(lines))

    # ------------------------------------------------------------------ #
    # Report archive (mirrors data/reports/*.txt as markdown)

    def _write_reports(self) -> int:
        n = 0
        if not REPORTS_DIR.exists():
            return 0
        for txt in sorted(REPORTS_DIR.rglob("*.txt")):
            rel = txt.relative_to(REPORTS_DIR)
            kind = rel.parts[0] if len(rel.parts) > 1 else "report"
            title = f"{kind}-{txt.stem}"
            body = txt.read_text()
            md = (f"---\ntype: report\nkind: {kind}\n"
                  f"tags: [quntra, report]\n---\n"
                  f"# {title}\n\n```\n{body}\n```\n")
            (self.vault / "Reports" / f"{_slug(title)}.md").write_text(md)
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # Knowledge notes grouped by type

    def _write_knowledge(self, knowledge) -> int:
        by_type: dict[str, list] = defaultdict(list)
        for k in knowledge:
            by_type[k.knowledge_type].append(k)
        for ktype, items in by_type.items():
            lines = [
                "---", f"type: knowledge", f"knowledge_type: {ktype}",
                "tags: [quntra, knowledge]", "---",
                f"# {ktype}", "",
                f"{len(items)} item(s).", "",
            ]
            for k in sorted(items, key=lambda x: x.created_at, reverse=True):
                ticks = " ".join(f"[[{_slug(t)}]]" for t in (k.tickers or []))
                lines.append(f"- {k.content[:260]} {ticks}".rstrip())
            (self.vault / "Knowledge" / f"{_slug(ktype)}.md").write_text(
                "\n".join(lines))
        return len(by_type)

    # ------------------------------------------------------------------ #

    def _write_home(self, counts: dict, n_trades: int) -> None:
        gate = self._gate_line()
        lines = [
            "---", "type: home", "tags: [quntra]", "---",
            "# QuNtra Vault", "",
            f"Auto-generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
            "from QuNtra's database. Do not edit derived notes by hand — "
            "they are regenerated on each sync.", "",
            "## Status", "", gate, "",
            "## Contents", "",
            f"- **[[Daily]]** — {counts['daily']} trading-day notes",
            f"- **Tickers** — {counts['tickers']} instrument histories",
            f"- **Reports** — {counts['reports']} archived reports",
            f"- **Knowledge** — {counts['knowledge']} lesson categories",
            f"- Total paper trades recorded: {n_trades}", "",
            "> Point Obsidian at this folder: *Open folder as vault* → "
            "select the `obsidian_vault` directory.",
        ]
        (self.vault / "Home.md").write_text("\n".join(lines))

    def _gate_line(self) -> str:
        try:
            from scripts.paper_trading_status import gather_stats
            st = gather_stats()
            if st is None:
                return "Paper gate: **day 0/40** — no trades yet."
            return (f"Paper gate: **day {st['days']}/40** · "
                    f"Sharpe " + (f"{st['sharpe']:.2f}"
                                  if not (st['sharpe'] != st['sharpe'])
                                  else "n/a")
                    + f" · Max DD {st['max_dd']:+.2%} · "
                    f"P&L ₹{st['total_pnl']:+,.0f}")
        except Exception:  # noqa: BLE001
            return "Paper gate: status unavailable."
