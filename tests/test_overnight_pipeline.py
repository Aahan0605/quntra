"""Tests for src/research/overnight_pipeline.py — offline, agents mocked."""

from unittest.mock import patch

import pytest

from src.agents.research.base import ResearchOutput
from src.db import Base, get_engine
from src.knowledge import KnowledgeManager
from src.research import OvernightResearchPipeline


class FakeTelegram:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return True


@pytest.fixture
def db_url(tmp_path):
    url = f"sqlite:///{tmp_path / 'overnight.db'}"
    Base.metadata.create_all(get_engine(url))
    return url


@pytest.fixture
def pipeline(db_url):
    return OvernightResearchPipeline(
        db_url=db_url,
        telegram=FakeTelegram(),
        knowledge=KnowledgeManager(db_url),
    )


def _fake_output(name):
    return ResearchOutput(agent=name, summary=f"{name} ok", confidence=0.7)


def test_pipeline_runs_all_nine_tasks(pipeline):
    with patch("src.agents.research.company_analysis_agent."
               "CompanyAnalysisAgent.run",
               return_value=_fake_output("company_analysis_agent")), \
         patch("src.agents.research.news_agent.NewsAgent.run",
               return_value=_fake_output("news_agent")), \
         patch("src.agents.research.macro_agent.MacroAgent.run",
               return_value=_fake_output("macro_agent")), \
         patch("src.research.overnight_pipeline.OvernightResearchPipeline."
               "task_rbi_bulletin",
               return_value={"summary": "rbi ok", "confidence": 0.5}), \
         patch("src.agents.research.base.fetch_rss", return_value=[]):
        report = pipeline.run()

    assert len(report.results) + len(report.errors) == 9
    assert len(report.errors) == 0
    assert "premarket_draft" in report.results


def test_pipeline_isolates_task_errors(pipeline):
    with patch("src.agents.research.company_analysis_agent."
               "CompanyAnalysisAgent.run",
               side_effect=RuntimeError("agent exploded")), \
         patch("src.agents.research.news_agent.NewsAgent.run",
               return_value=_fake_output("news_agent")), \
         patch("src.agents.research.macro_agent.MacroAgent.run",
               return_value=_fake_output("macro_agent")), \
         patch("src.research.overnight_pipeline.OvernightResearchPipeline."
               "task_rbi_bulletin", side_effect=RuntimeError("rbi down")), \
         patch("src.agents.research.base.fetch_rss", return_value=[]):
        report = pipeline.run()

    # safe_run absorbs the agent error; the rbi task raises and is isolated
    assert "rbi_bulletin" in report.errors
    assert "premarket_draft" in report.results  # later tasks still ran
    assert any("rbi_bulletin" in m for m in pipeline.telegram.sent)


def test_pipeline_stores_knowledge(pipeline, db_url):
    with patch("src.agents.research.company_analysis_agent."
               "CompanyAnalysisAgent.run",
               return_value=_fake_output("company_analysis_agent")), \
         patch("src.agents.research.news_agent.NewsAgent.run",
               return_value=_fake_output("news_agent")), \
         patch("src.agents.research.macro_agent.MacroAgent.run",
               return_value=_fake_output("macro_agent")), \
         patch("src.research.overnight_pipeline.OvernightResearchPipeline."
               "task_rbi_bulletin",
               return_value={"summary": "rbi ok", "confidence": 0.5}), \
         patch("src.agents.research.base.fetch_rss", return_value=[]):
        pipeline.run()

    km = KnowledgeManager(db_url)
    assert km.count() >= 9  # one OVERNIGHT_RESEARCH item per task


def test_premarket_draft_persisted(pipeline, db_url):
    with patch("src.agents.research.company_analysis_agent."
               "CompanyAnalysisAgent.run",
               return_value=_fake_output("company_analysis_agent")), \
         patch("src.agents.research.news_agent.NewsAgent.run",
               return_value=_fake_output("news_agent")), \
         patch("src.agents.research.macro_agent.MacroAgent.run",
               return_value=_fake_output("macro_agent")), \
         patch("src.research.overnight_pipeline.OvernightResearchPipeline."
               "task_rbi_bulletin",
               return_value={"summary": "rbi ok", "confidence": 0.5}), \
         patch("src.agents.research.base.fetch_rss", return_value=[]):
        pipeline.run()

    from src.db import SystemState, get_session
    with get_session(db_url) as s:
        draft = s.get(SystemState, "premarket_draft")
        assert draft is not None
        assert "QUNTRA PRE-MARKET INTELLIGENCE" in draft.value["report"]


def test_report_summary_readable(pipeline):
    report_cls = type(pipeline).__mro__  # noqa: F841 — smoke reference
    from src.research import OvernightReport
    r = OvernightReport()
    r.add("taskA", {"summary": "did things"})
    r.add_error("taskB", "broke")
    text = r.summary()
    assert "1 tasks ok, 1 failed" in text
    assert "✓ taskA" in text and "✗ taskB" in text
