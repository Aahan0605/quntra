"""fetch_newsapi() and its merge into NewsAgent — all HTTP mocked, no live
network calls. NEWSAPI_KEY was supplied by the user directly in chat, so
this also pins the "no key -> no crash" contract that keeps the research
pipeline running when the key is absent, revoked, or rate-limited."""

from unittest.mock import MagicMock, patch

from src.agents.research.base import fetch_newsapi
from src.agents.research.news_agent import NewsAgent

FAKE_RESPONSE = {
    "status": "ok",
    "articles": [
        {"title": "Nifty hits record high on FII inflows",
         "description": "strong rally", "url": "http://a",
         "publishedAt": "2026-07-28T10:00:00Z"},
        {"title": "No date article", "description": "x", "url": "http://b",
         "publishedAt": None},
    ],
}


def test_fetch_newsapi_without_key_returns_empty_no_network(monkeypatch):
    # src/utils/data_fetcher.py load_dotenv()s config/secrets.env as a side
    # effect; earlier tests in the same pytest process can leak the real
    # NEWSAPI_KEY into os.environ. Explicit delenv, not ambient absence.
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    with patch("requests.get") as mock_get:
        out = fetch_newsapi("nifty", api_key=None)
    assert out == []
    mock_get.assert_not_called()


def test_fetch_newsapi_parses_articles():
    resp = MagicMock(json=lambda: FAKE_RESPONSE)
    resp.raise_for_status = lambda: None
    with patch("requests.get", return_value=resp) as mock_get:
        out = fetch_newsapi("nifty", api_key="fake-key")
    assert mock_get.call_args.kwargs["params"]["apiKey"] == "fake-key"
    assert len(out) == 2
    assert out[0]["title"] == "Nifty hits record high on FII inflows"
    assert out[0]["published_dt"] is not None
    assert out[1]["published_dt"] is None  # missing date -> None, not a crash


def test_fetch_newsapi_degrades_on_http_error():
    with patch("requests.get", side_effect=ConnectionError("down")):
        out = fetch_newsapi("nifty", api_key="fake-key")
    assert out == []


def test_fetch_newsapi_degrades_on_api_error_status():
    resp = MagicMock(json=lambda: {"status": "error", "message": "rate limited"})
    resp.raise_for_status = lambda: None
    with patch("requests.get", return_value=resp):
        out = fetch_newsapi("nifty", api_key="fake-key")
    assert out == []


def test_news_agent_merges_newsapi_alongside_rss(db_url_factory=None):
    from src.db import Base, get_engine
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        db_url = f"sqlite:///{d}/research.db"
        Base.metadata.create_all(get_engine(db_url))

        with patch("src.agents.research.news_agent.fetch_rss", return_value=[]), \
             patch("src.agents.research.news_agent.fetch_newsapi",
                  return_value=[{"title": "Sensex surges on record profit",
                                "summary": "strong growth", "link": "http://a",
                                "published": "", "published_dt": None}]):
            out = NewsAgent(db_url, newsapi_key="k").run({})
        assert out.ok
        assert "newsapi" in out.sources
        assert any("Sensex" in f["title"] for f in out.findings)


def test_news_agent_dedups_across_rss_and_newsapi():
    from src.db import Base, get_engine
    import tempfile
    same_item = {"title": "Nifty falls as FII selling continues",
                "summary": "weak tape", "link": "http://x", "published": "",
                "published_dt": None}
    with tempfile.TemporaryDirectory() as d:
        db_url = f"sqlite:///{d}/research.db"
        Base.metadata.create_all(get_engine(db_url))
        with patch("src.agents.research.news_agent.fetch_rss",
                  return_value=[same_item]), \
             patch("src.agents.research.news_agent.fetch_newsapi",
                  return_value=[same_item]):
            out = NewsAgent(db_url, newsapi_key="k").run({})
        titles = [f["title"] for f in out.findings]
        assert titles.count("Nifty falls as FII selling continues") == 1


def test_news_agent_still_works_with_no_newsapi_key(monkeypatch):
    """The default, real-world case: NEWSAPI_KEY unset -> RSS-only, same
    behavior as before this feature existed.

    newsapi_key=None means "fall back to the environment" by design (see
    NewsAgent's docstring) — so proving "no key" requires actually clearing
    the environment, not just passing None to the constructor.
    """
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    from src.db import Base, get_engine
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        db_url = f"sqlite:///{d}/research.db"
        Base.metadata.create_all(get_engine(db_url))
        with patch("src.agents.research.news_agent.fetch_rss", return_value=[]):
            out = NewsAgent(db_url, newsapi_key=None).run({})
        assert out.confidence == 0.0
        assert "newsapi" not in out.sources
