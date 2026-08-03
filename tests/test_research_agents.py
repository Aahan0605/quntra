"""Tests for src/agents/research/ — offline, all network mocked."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.agents.research import (
    CompanyAnalysisAgent,
    GeopoliticalAgent,
    MacroAgent,
    NewsAgent,
    ResearchWriter,
    SectorAgent,
)
from src.agents.research.base import BaseResearchAgent, ResearchOutput
from src.agents.research.news_agent import score_relevance, score_sentiment
from src.db import Base, get_engine


@pytest.fixture
def db_url(tmp_path):
    url = f"sqlite:///{tmp_path / 'research.db'}"
    Base.metadata.create_all(get_engine(url))
    return url


# --------------------------------------------------------------------- #
# Base

def test_safe_run_never_raises(db_url):
    class Exploder(BaseResearchAgent):
        name = "exploder"

        def run(self, context):
            raise RuntimeError("boom")

    out = Exploder(db_url).safe_run({})
    assert not out.ok
    assert out.confidence == 0.0
    assert "boom" in out.error


def test_store_writes_research_note(db_url):
    class Simple(BaseResearchAgent):
        name = "simple"
        note_type = "test_note"

        def run(self, context):
            return ResearchOutput(agent=self.name, summary="hello",
                                  confidence=0.9)

    agent = Simple(db_url)
    note_id = agent.store(agent.run({}))
    assert note_id is not None

    from sqlalchemy import select
    from src.db import ResearchNote, get_session
    with get_session(db_url) as s:
        row = s.execute(select(ResearchNote)).scalar_one()
        assert row.source == "simple"
        assert row.summary == "hello"


# --------------------------------------------------------------------- #
# NewsAgent

FAKE_FEED = [
    {"title": "Reliance surges on record profit", "summary": "strong growth",
     "link": "http://x", "published": "today"},
    {"title": "Nifty falls as FII selling continues", "summary": "weak tape",
     "link": "http://y", "published": "today"},
    {"title": "Celebrity gossip roundup", "summary": "irrelevant",
     "link": "http://z", "published": "today"},
]


def test_news_agent_filters_and_scores(db_url):
    with patch("src.agents.research.news_agent.fetch_rss",
               return_value=FAKE_FEED), \
         patch("src.agents.research.news_agent.fetch_newsapi",
              return_value=[]):
        out = NewsAgent(db_url).run({"watchlist": ["RELIANCE.NS"]})
    assert out.ok
    titles = [f["title"] for f in out.findings]
    assert any("Reliance" in t for t in titles)
    assert not any("gossip" in t for t in titles)  # relevance gate
    rel_item = next(f for f in out.findings if "Reliance" in f["title"])
    assert rel_item["sentiment"] > 0
    assert "RELIANCE.NS" in rel_item["tickers"]
    assert rel_item["relevance"] == 0.9  # watchlist boost


def test_news_agent_no_feeds(db_url):
    # fetch_newsapi is also mocked: NEWSAPI_KEY can leak into os.environ
    # from src/utils/data_fetcher.py's load_dotenv() side effect in an
    # earlier test, which would otherwise make this a live network call.
    with patch("src.agents.research.news_agent.fetch_rss", return_value=[]), \
         patch("src.agents.research.news_agent.fetch_newsapi", return_value=[]):
        out = NewsAgent(db_url).run({})
    assert out.confidence == 0.0
    assert "No news feeds reachable" in out.summary


def test_relevance_does_not_substring_match_ordinary_words():
    """'nse' is a substring of 'response'/'sense'/'expense'; 'dii' of
    'radii'; 'itc' of 'kitchen' — naive `kw in text` flagged all of these
    as market-relevant. Must require whole-word matches."""
    for text in ["it made no sense to sell", "measured in radii",
                "clean the kitchen", "in response to the crisis",
                "at great expense to shareholders"]:
        rel, tickers = score_relevance(text)
        assert rel == 0.0 and tickers == [], f"false positive on: {text!r}"


def test_relevance_still_matches_real_company_and_market_terms():
    rel, tickers = score_relevance("Reliance Industries posts record profit")
    assert rel == 0.7 and tickers == ["RELIANCE.NS"]
    rel2, _ = score_relevance("NSE extends trading hours amid volatility")
    assert rel2 == 0.55


def test_sentiment_and_relevance_helpers():
    assert score_sentiment("record profit growth strong") > 0
    assert score_sentiment("crash plunge loss fraud") < 0
    rel, tickers = score_relevance("tata steel expands capacity")
    assert rel > 0.5 and tickers == ["TATASTEEL.NS"]
    # phrases outweigh single words
    assert score_sentiment("company beat estimates despite weak quarter") > 0


def test_news_agent_freshness_and_dedup(db_url):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    stale = now - timedelta(hours=30)
    feed = [
        {"title": "Reliance surges on record profit", "summary": "",
         "link": "a", "published": "", "published_dt": now},
        {"title": "Reliance surges on record profit", "summary": "",
         "link": "b", "published": "", "published_dt": now},   # duplicate
        {"title": "Infosys wins large deal, shares jump", "summary": "",
         "link": "c", "published": "", "published_dt": stale},  # too old
    ]
    with patch("src.agents.research.news_agent.fetch_rss",
               return_value=feed), \
         patch("src.agents.research.news_agent.fetch_newsapi",
              return_value=[]):
        out = NewsAgent(db_url).run({})
    titles = [f["title"] for f in out.findings]
    # duplicate collapsed to one, stale item dropped
    assert titles.count("Reliance surges on record profit") == 1
    assert not any("Infosys" in t for t in titles)


def test_company_agent_persists_blacklist(db_url):
    from datetime import date
    with patch.object(CompanyAnalysisAgent, "_events_for",
                      side_effect=lambda t, d: (
                          {"earnings_blackout": True,
                           "earnings_date": "2026-07-09"}
                          if t == "INFY.NS" else {})):
        CompanyAnalysisAgent(db_url).run(
            {"date": date(2026, 7, 7), "watchlist": ["INFY.NS", "TCS.NS"]})
    from src.db import SystemState, get_session
    with get_session(db_url) as s:
        row = s.get(SystemState, "earnings_blacklist")
        assert row is not None
        assert row.value["tickers"] == ["INFY.NS"]


# --------------------------------------------------------------------- #
# MacroAgent

def test_macro_agent_positive_bias(db_url):
    moves = {"sp500": 0.8, "nasdaq": 1.1, "dow": 0.5, "crude_oil": -2.0,
             "gold": 0.1, "usdinr": -0.4, "us_10y": 0.0,
             "nikkei": 0.9, "hang_seng": 0.7, "vix": 0.0}
    from src.agents.research.macro_agent import MACRO_TICKERS
    symbol_to_name = {v: k for k, v in MACRO_TICKERS.items()}
    with patch("src.agents.research.macro_agent.yf_pct_change",
               side_effect=lambda t, period="5d":
               moves.get(symbol_to_name.get(t), 0.0)), \
         patch.object(MacroAgent, "_vix_level", return_value=13.5):
        out = MacroAgent(db_url).run({})
    assert out.payload["macro_bias"] == "POSITIVE"
    assert "US equities up" in out.reasoning
    assert out.payload["asia_direction"] == "UP"
    assert "VIX calm" in out.reasoning


def test_macro_agent_vix_fear_drags_bias(db_url):
    with patch("src.agents.research.macro_agent.yf_pct_change",
               return_value=None), \
         patch.object(MacroAgent, "_vix_level", return_value=31.0):
        out = MacroAgent(db_url).run({})
    assert out.payload["macro_bias"] == "NEGATIVE"
    assert "VIX elevated" in out.reasoning


def test_macro_agent_all_sources_down(db_url):
    with patch("src.agents.research.macro_agent.yf_pct_change",
               return_value=None), \
         patch.object(MacroAgent, "_vix_level", return_value=None):
        out = MacroAgent(db_url).run({})
    assert out.payload["macro_bias"] == "NEUTRAL"
    assert out.confidence == 0.0


# --------------------------------------------------------------------- #
# SectorAgent

def test_sector_agent_ranks_sectors(db_url):
    idx = pd.bdate_range("2026-01-01", periods=30)
    panel = pd.DataFrame({
        "TCS.NS": [100 + i for i in range(30)],          # IT rising
        "INFY.NS": [100 + i for i in range(30)],
        "SBIN.NS": [100 - i for i in range(30)],         # BANKS falling
        "HDFCBANK.NS": [100 - i * 0.5 for i in range(30)],
    }, index=idx)
    with patch.object(SectorAgent, "_load_panel", return_value=panel):
        out = SectorAgent(db_url).run({})
    assert out.ok
    assert out.payload["leaders"][0] == "IT"
    assert "BANKS" in out.payload["laggards"]


def test_sector_agent_no_cache(db_url):
    with patch.object(SectorAgent, "_load_panel", return_value=None):
        out = SectorAgent(db_url).run({})
    assert not out.ok


# --------------------------------------------------------------------- #
# GeopoliticalAgent

GEO_FEED = [
    {"title": "Missile strike escalates conflict near Strait of Hormuz",
     "summary": "oil supply disruption feared", "link": "", "published": ""},
    {"title": "New sanctions on crude exports announced",
     "summary": "embargo widens", "link": "", "published": ""},
    {"title": "Local sports team wins", "summary": "", "link": "",
     "published": ""},
]


def test_geopolitical_agent_scores_risk(db_url):
    with patch("src.agents.research.geopolitical_agent.fetch_rss",
               return_value=GEO_FEED):
        out = GeopoliticalAgent(db_url).run({})
    assert out.ok
    score = out.payload["geopolitical_risk_score"]
    assert 0 < score <= 10
    assert len(out.payload["top_events"]) >= 1
    assert out.payload["top_events"][0]["escalating"]


def test_geopolitical_agent_quiet_world(db_url):
    with patch("src.agents.research.geopolitical_agent.fetch_rss",
               return_value=[{"title": "flower show opens", "summary": "",
                              "link": "", "published": ""}]):
        out = GeopoliticalAgent(db_url).run({})
    assert out.payload["geopolitical_risk_score"] == 0.0


# --------------------------------------------------------------------- #
# CompanyAnalysisAgent

def test_company_agent_earnings_blackout(db_url):
    from datetime import date, timedelta
    today = date(2026, 7, 6)
    with patch.object(CompanyAnalysisAgent, "_events_for",
                      side_effect=lambda t, d: (
                          {"earnings_date": (today + timedelta(days=2)).isoformat(),
                           "earnings_blackout": True}
                          if t == "TCS.NS" else {})):
        out = CompanyAnalysisAgent(db_url).run(
            {"date": today, "watchlist": ["TCS.NS", "RELIANCE.NS"]})
    assert out.payload["earnings_blackout"] == ["TCS.NS"]


# --------------------------------------------------------------------- #
# ResearchWriter

def _fake_outputs():
    return {
        "macro_agent": ResearchOutput(
            agent="macro_agent", summary="Macro bias: POSITIVE. US up",
            payload={"macro_bias": "POSITIVE",
                     "moves": {"crude_oil": 2.0, "usdinr": 0.5}}),
        "news_agent": ResearchOutput(
            agent="news_agent", summary="3 items",
            payload={"avg_sentiment": -0.5, "n_items": 3}),
        "sector_agent": ResearchOutput(
            agent="sector_agent", summary="IT leads",
            payload={"leaders": ["IT", "PHARMA"], "laggards": ["BANKS"]}),
        "company_analysis_agent": ResearchOutput(
            agent="company_analysis_agent", summary="1 flagged",
            payload={"earnings_blackout": ["TCS.NS"]}),
        "geopolitical_agent": ResearchOutput(
            agent="geopolitical_agent", summary="risk 3.0",
            payload={"geopolitical_risk_score": 3.0, "top_events": [
                {"title": "sanctions news", "weight": 2.0}]}),
    }


def test_writer_composes_full_report(db_url):
    report = ResearchWriter(db_url).compose(_fake_outputs(),
                                            {"regime": "BULL_TRENDING"})
    for fragment in ["MARKET REGIME: BULL_TRENDING", "MACRO BIAS: POSITIVE",
                     "TOP RISKS:", "SECTOR LEADERS: IT, PHARMA",
                     "earnings blackout: TCS.NS", "GEOPOLITICAL: 3.0/10",
                     "RECOMMENDATION: TRADE"]:
        assert fragment in report


def test_writer_no_trade_on_geo_risk(db_url):
    outputs = _fake_outputs()
    outputs["geopolitical_agent"].payload["geopolitical_risk_score"] = 9.0
    report = ResearchWriter(db_url).compose(outputs, {})
    assert "NO TRADE" in report


def test_writer_no_trade_when_agents_down(db_url):
    outputs = _fake_outputs()
    for name in ["macro_agent", "news_agent", "sector_agent"]:
        outputs[name] = ResearchOutput(agent=name, summary="down",
                                       confidence=0, error="dead")
    report = ResearchWriter(db_url).compose(outputs, {})
    assert "NO TRADE" in report
    assert "degraded" in report
