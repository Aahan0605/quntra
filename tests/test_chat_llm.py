"""Tests for /chat → ResearchWriter.answer_question — Claude mocked, offline."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agents.research import ResearchWriter
from src.db import Base, get_engine
from src.knowledge import KnowledgeManager


@pytest.fixture
def db_url(tmp_path):
    url = f"sqlite:///{tmp_path / 'chat.db'}"
    Base.metadata.create_all(get_engine(url))
    return url


def _fake_anthropic(reply_text):
    """A fake `anthropic` module whose Anthropic().messages.create returns
    a response with one text block."""
    block = SimpleNamespace(type="text", text=reply_text)
    resp = SimpleNamespace(content=[block])
    client = MagicMock()
    client.messages.create.return_value = resp
    module = MagicMock()
    module.Anthropic.return_value = client
    return module, client


def test_chat_uses_claude_when_key_present(db_url, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    module, client = _fake_anthropic("The regime is BULL_TRENDING.")
    with patch.dict("sys.modules", {"anthropic": module}):
        out = ResearchWriter(db_url).answer_question(
            "what is the current market regime?",
            {"regime": {"state": "BULL_TRENDING"}, "macro_bias": "POSITIVE"})
    assert "BULL_TRENDING" in out
    # Verify the call was grounded with QuNtra context
    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-haiku-4-5"
    assert "PAPER-TRADING" in kwargs["system"]
    assert "BULL_TRENDING" in kwargs["system"]
    assert kwargs["messages"][0]["content"] == \
        "what is the current market regime?"


def test_chat_grounds_memory_into_prompt(db_url, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    KnowledgeManager(db_url).store(
        "TRADE_LESSON", "RELIANCE gap-ups after results tend to fade by noon")
    module, client = _fake_anthropic("Noted.")
    with patch.dict("sys.modules", {"anthropic": module}):
        ResearchWriter(db_url).answer_question(
            "what do we know about RELIANCE gap-ups?", {})
    _, kwargs = client.messages.create.call_args
    assert "RELIANCE gap-ups" in kwargs["system"]


def test_chat_falls_back_when_no_key(db_url, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    KnowledgeManager(db_url).store(
        "MARKET_OBSERVATION", "banks led the rally in the last session")
    out = ResearchWriter(db_url).answer_question("how did banks do?", {})
    # deterministic path — no 🤖 prefix, pulls from memory
    assert "🤖" not in out
    assert "banks led the rally" in out


def test_chat_falls_back_on_api_error(db_url, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    module = MagicMock()
    module.Anthropic.return_value.messages.create.side_effect = \
        RuntimeError("rate limited")
    with patch.dict("sys.modules", {"anthropic": module}):
        out = ResearchWriter(db_url).answer_question(
            "anything", {"regime": {"state": "SIDEWAYS"}})
    # degraded gracefully to the deterministic answer
    assert "🤖" not in out
    assert "SIDEWAYS" in out


def test_chat_never_raises(db_url, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    # anthropic import fails entirely
    with patch.dict("sys.modules", {"anthropic": None}):
        out = ResearchWriter(db_url).answer_question("hello", {})
    assert isinstance(out, str) and out
