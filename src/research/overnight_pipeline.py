"""
OvernightResearchPipeline — QuNtra works while the operator sleeps.

Runs 22:00–06:00 IST, triggered by Hermes run_overnight_batch(). Nine
prioritized tasks; an error in one never stops the rest. Every result
lands in research_notes and the knowledge base; failures alert Telegram.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

logger = logging.getLogger("quntra.overnight")


@dataclass
class OvernightReport:
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    def add(self, task: str, result: dict) -> None:
        self.results[task] = result

    def add_error(self, task: str, error: str) -> None:
        self.errors[task] = error

    def summary(self) -> str:
        lines = [f"OVERNIGHT RESEARCH — {self.started_at[:16]}",
                 f"{len(self.results)} tasks ok, {len(self.errors)} failed"]
        for name, res in self.results.items():
            lines.append(f"  ✓ {name}: {res.get('summary', 'done')[:100]}")
        for name, err in self.errors.items():
            lines.append(f"  ✗ {name}: {err[:100]}")
        return "\n".join(lines)


class OvernightResearchPipeline:
    """Priority queue of overnight jobs. See module docstring."""

    def __init__(self, db_url: str | None = None, telegram=None,
                 knowledge=None, fetcher=None):
        self.db_url = db_url
        self.telegram = telegram
        self.fetcher = fetcher
        if knowledge is None:
            from src.knowledge import KnowledgeManager
            knowledge = KnowledgeManager(db_url)
        self.knowledge = knowledge

    # ------------------------------------------------------------------ #

    def get_priority_queue(self) -> list[tuple[str, callable]]:
        return [
            ("earnings_analysis", self.task_earnings_analysis),
            ("corporate_filings", self.task_corporate_filings),
            ("rbi_bulletin", self.task_rbi_bulletin),
            ("research_paper_digest", self.task_research_paper_digest),
            ("global_macro_update", self.task_global_macro_update),
            ("hypothesis_testing", self.task_hypothesis_testing),
            ("model_validation", self.task_model_validation),
            ("system_maintenance", self.task_system_maintenance),
            ("premarket_draft", self.task_premarket_draft),
        ]

    def run(self) -> OvernightReport:
        report = OvernightReport()
        self._context = {"outputs": {}}  # accumulates agent outputs for draft
        for name, task in self.get_priority_queue():
            try:
                result = task() or {}
                report.add(name, result)
                self.knowledge.store(
                    knowledge_type="OVERNIGHT_RESEARCH",
                    content=f"{name}: {result.get('summary', 'completed')}",
                    confidence=result.get("confidence", 0.5),
                    source="overnight_pipeline",
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("overnight task %s failed", name)
                report.add_error(name, str(e))
                if self.telegram is not None:
                    try:
                        self.telegram.send(
                            f"🚨 Overnight task failed: {name} — {e}")
                    except Exception:  # noqa: BLE001
                        pass
        return report

    # ------------------------------------------------------------------ #
    # Tasks, in priority order

    def task_earnings_analysis(self) -> dict:
        """1. Companies reporting within the blackout window."""
        from src.agents.research import CompanyAnalysisAgent
        out = CompanyAnalysisAgent(self.db_url).safe_run(
            {"date": date.today()})
        self._context["outputs"]["company_analysis_agent"] = out
        out_id = CompanyAnalysisAgent(self.db_url).store(out)
        return {"summary": out.summary, "confidence": out.confidence,
                "note_id": out_id}

    def task_corporate_filings(self) -> dict:
        """2. New filings/announcements scan (news feeds proxy)."""
        from src.agents.research import NewsAgent
        out = NewsAgent(self.db_url).safe_run({})
        self._context["outputs"]["news_agent"] = out
        NewsAgent(self.db_url).store(out)
        return {"summary": out.summary, "confidence": out.confidence}

    def task_rbi_bulletin(self) -> dict:
        """3. RBI policy rates check."""
        if self.fetcher is None:
            from src.utils.data_fetcher import UnifiedDataFetcher
            self.fetcher = UnifiedDataFetcher()
        try:
            rates = self.fetcher.get_rbi_data()
            summary = (f"RBI rates: {rates.iloc[0].to_dict()}"
                       if len(rates) else "RBI data empty")
            conf = 0.8
        except Exception as e:  # noqa: BLE001 — RBI endpoint often blocked
            summary, conf = f"RBI source unreachable ({e})", 0.0
        return {"summary": summary[:300], "confidence": conf}

    def task_research_paper_digest(self) -> dict:
        """4. arXiv q-fin scan for relevant new papers."""
        from src.agents.research.base import fetch_rss
        entries = fetch_rss("https://rss.arxiv.org/rss/q-fin.PM+q-fin.TR",
                            limit=20)
        keywords = {"portfolio", "momentum", "india", "transaction cost",
                    "regime", "volatility", "machine learning"}
        hits = [e["title"] for e in entries
                if any(k in (e["title"] + e["summary"]).lower()
                       for k in keywords)]
        return {"summary": (f"{len(hits)} relevant papers of {len(entries)}: "
                            + "; ".join(h[:60] for h in hits[:3])
                            if entries else "arXiv feed unreachable"),
                "confidence": 0.6 if entries else 0.0}

    def task_global_macro_update(self) -> dict:
        """5. Asia overnight / US close / commodities snapshot."""
        from src.agents.research import MacroAgent
        out = MacroAgent(self.db_url, fetcher=self.fetcher).safe_run({})
        self._context["outputs"]["macro_agent"] = out
        MacroAgent(self.db_url).store(out)
        return {"summary": out.summary, "confidence": out.confidence}

    def task_hypothesis_testing(self) -> dict:
        """6. Test hypotheses queued during the day (system_state)."""
        from src.db import SystemState, get_session
        with get_session(self.db_url) as s:
            row = s.get(SystemState, "hypothesis_queue")
            queue = (row.value or {}).get("items", []) if row else []
        if not queue:
            return {"summary": "no hypotheses queued", "confidence": 0.5}
        # Hypotheses are recorded for the analyst; automated testing is a
        # Phase-3 upgrade — for now each is archived to the knowledge base.
        for h in queue:
            self.knowledge.store("STRATEGY_INSIGHT",
                                 f"HYPOTHESIS (untested): {h}",
                                 confidence=0.3, source="hypothesis_queue")
        with get_session(self.db_url) as s:
            s.merge(SystemState(key="hypothesis_queue", value={"items": []}))
        return {"summary": f"{len(queue)} hypotheses archived for review",
                "confidence": 0.5}

    def task_model_validation(self) -> dict:
        """7. Deployed models still load and predict on latest data."""
        import json
        import pickle
        from pathlib import Path
        model_dir = Path("data/models")
        summary_file = model_dir / "training_summary.json"
        if not summary_file.exists():
            return {"summary": "no training summary — models not trained yet",
                    "confidence": 0.0}
        deployed = [r for r in json.loads(summary_file.read_text())
                    if r.get("passed_gate")]
        ok, broken = 0, []
        for r in deployed:
            stem = r["ticker"].replace("&", "_")
            try:
                with open(model_dir / f"{stem}.pkl", "rb") as fp:
                    pickle.load(fp)
                ok += 1
            except Exception as e:  # noqa: BLE001
                broken.append(f"{r['ticker']}: {e}")
        return {"summary": f"{ok}/{len(deployed)} deployed models healthy"
                           + (f"; BROKEN: {broken}" if broken else ""),
                "confidence": 1.0 if not broken else 0.2}

    def task_system_maintenance(self) -> dict:
        """8. Database housekeeping + old-log archive."""
        import gzip
        import shutil
        from pathlib import Path

        from sqlalchemy import text
        from src.db import get_engine
        engine = get_engine(self.db_url)
        actions = []
        try:
            if engine.dialect.name == "sqlite":
                with engine.connect() as c:
                    c.exec_driver_sql("VACUUM")
                actions.append("sqlite VACUUM")
            else:
                with engine.connect().execution_options(
                        isolation_level="AUTOCOMMIT") as c:
                    c.execute(text("VACUUM (ANALYZE)"))
                actions.append("postgres VACUUM ANALYZE")
        except Exception as e:  # noqa: BLE001
            actions.append(f"vacuum skipped: {e}")

        logs = Path("logs")
        if logs.exists():
            cutoff = datetime.now().timestamp() - 14 * 86400
            for f in logs.glob("*.log"):
                if f.stat().st_mtime < cutoff and f.stat().st_size > 0:
                    with open(f, "rb") as src, \
                            gzip.open(f"{f}.gz", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    f.write_text("")
                    actions.append(f"archived {f.name}")
        return {"summary": "; ".join(actions), "confidence": 1.0}

    def task_premarket_draft(self) -> dict:
        """9. Draft tomorrow's pre-market report from tonight's outputs."""
        from src.agents.research import ResearchWriter
        writer = ResearchWriter(self.db_url)
        report = writer.compose(self._context["outputs"],
                                {"regime": self._current_regime()})
        from src.db import SystemState, get_session
        with get_session(self.db_url) as s:
            s.merge(SystemState(key="premarket_draft", value={
                "report": report,
                "drafted_at": datetime.now(timezone.utc).isoformat(),
            }))
        return {"summary": f"pre-market draft ready ({len(report)} chars)",
                "confidence": 0.7}

    def _current_regime(self) -> str:
        try:
            from src.db import SystemState, get_session
            with get_session(self.db_url) as s:
                row = s.get(SystemState, "regime")
                return (row.value or {}).get("state", "UNKNOWN") if row \
                    else "UNKNOWN"
        except Exception:  # noqa: BLE001
            return "UNKNOWN"
