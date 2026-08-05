"""Tests for HermesCoordinator (Task 1-5) — full dry-run with stubs."""
import pytest

import src.db.session as db_session
from src.db import init_db
from src.governor.brain import QuNtraBrain
from src.governor.hermes import HermesCoordinator
from src.risk.drawdown_circuit import DrawdownCircuitBreaker
from src.risk.consecutive_loss_guard import ConsecutiveLossGuard


class StubTrader:
    def __init__(self):
        self.orders = []

    def place_order(self, ticker, direction, qty, signal_hash=None, **meta):
        trade = {"ticker": ticker, "direction": direction, "qty": qty,
                 "signal_hash": signal_hash, "is_paper": True, **meta}
        self.orders.append(trade)
        return trade

    def get_positions(self):
        return self.orders

    def manage_positions(self):
        return []

    def reconcile(self):
        return {"reconciled": len(self.orders)}


class StubCouncil:
    def score_premarket(self, universe):
        return {t: (11 if i < 3 else 5) for i, t in enumerate(universe)}

    def live_signals(self, watchlist):
        return [{"ticker": t, "direction": "LONG", "score": 10,
                 "signal_hash": f"h-{t}"} for t in watchlist[:2]]

    def score_day(self):
        return [("technical", True), ("sentiment", False)]


class StubTelegram:
    def __init__(self):
        self.messages = []

    def send(self, msg):
        self.messages.append(msg)


class StubResearchAgent:
    """Offline stand-in for any research agent."""

    def __init__(self, name, payload=None):
        self.name = name
        self.payload = payload or {}
        self.calls = []

    def safe_run(self, context):
        from src.agents.research.base import ResearchOutput
        self.calls.append(context)
        return ResearchOutput(agent=self.name, summary=f"{self.name} ok",
                              confidence=0.7, payload=self.payload)

    def store(self, output):
        return None


def stub_research_team():
    return {
        "news_agent": StubResearchAgent("news_agent",
                                        {"avg_sentiment": 0.1}),
        "macro_agent": StubResearchAgent("macro_agent",
                                         {"macro_bias": "NEUTRAL",
                                          "moves": {}}),
        "geopolitical_agent": StubResearchAgent(
            "geopolitical_agent",
            {"geopolitical_risk_score": 2.0, "top_events": []}),
        "fundamental_agent": StubResearchAgent("fundamental_agent"),
        "sector_agent": StubResearchAgent("sector_agent",
                                          {"leaders": ["IT"],
                                           "laggards": ["BANKS"]}),
        "company_analysis_agent": StubResearchAgent(
            "company_analysis_agent", {"earnings_blackout": []}),
    }


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/hermes.db"
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)
    init_db(url)
    # No network in tests: kill RSS/yfinance helpers everywhere
    import src.agents.research.base as research_base
    monkeypatch.setattr(research_base, "fetch_rss", lambda *a, **k: [])
    monkeypatch.setattr(research_base, "yf_pct_change",
                        lambda *a, **k: None)
    import src.agents.research.macro_agent as macro_mod
    monkeypatch.setattr(macro_mod, "yf_pct_change", lambda *a, **k: None)
    import src.reporting.metrics as metrics_mod
    monkeypatch.setattr(metrics_mod, "nifty_move_today", lambda: None)
    h = HermesCoordinator(
        brain=QuNtraBrain(),
        trader=StubTrader(),
        telegram=StubTelegram(),
        circuit_breaker=DrawdownCircuitBreaker(),
        loss_guard=ConsecutiveLossGuard(),
        council=StubCouncil(),
        research_team=stub_research_team(),
    )
    yield h
    monkeypatch.setattr(db_session, "_engine", None)
    monkeypatch.setattr(db_session, "_SessionLocal", None)


def test_system_state_roundtrip(hermes):
    hermes.set_system_state("regime", {"state": "BULL"})
    assert hermes.get_system_state("regime") == {"state": "BULL"}
    hermes.set_system_state("regime", {"state": "BEAR"})
    assert hermes.get_system_state("regime") == {"state": "BEAR"}
    assert "regime" in hermes.get_system_state()


def test_pre_market_builds_watchlist(hermes):
    result = hermes.run_pre_market_sequence()
    assert len(result["watchlist"]) == 3  # stub scores 3 tickers >= 9
    stored = hermes.get_system_state("premarket")
    assert stored["watchlist"] == result["watchlist"]
    assert (hermes.get_system_state("oms") or {}).get("enabled")


def test_market_session_executes_signals(hermes, monkeypatch):
    # Signal-council trading is off by default since
    # scripts/backtest_signal_council.py showed it loses to buy-and-hold
    # NIFTY over 5 real years (docs/CEO_REVIEW.md) — this test explicitly
    # opts back in to exercise the execution path itself.
    monkeypatch.setenv("ENABLE_SIGNAL_COUNCIL_TRADING", "true")
    hermes.run_pre_market_sequence()
    actions = hermes.run_market_session()
    assert len(actions["executed"]) == 2
    assert hermes.trader.orders[0]["ticker"].endswith(".NS")


def test_market_session_signal_council_disabled_by_default(hermes, monkeypatch):
    monkeypatch.delenv("ENABLE_SIGNAL_COUNCIL_TRADING", raising=False)
    hermes.run_pre_market_sequence()
    actions = hermes.run_market_session()
    assert actions["executed"] == []
    assert any("signal-council disabled" in s
              for s in actions["skipped"])


def test_rejection_reason_never_exceeds_its_column(hermes, monkeypatch):
    """Signal.rejection_reason is String(100). An over-long value raises
    StringDataRightTruncation, which on 2026-08-05 killed every market tick
    for a whole session — the disable notice was 126 chars. Truncating is
    what keeps a wording change from stopping trading.
    """
    monkeypatch.delenv("ENABLE_SIGNAL_COUNCIL_TRADING", raising=False)
    hermes.run_pre_market_sequence()
    # Force a pathologically long skip reason through the same path.
    hermes.circuit = None
    actions = hermes.run_market_session()
    for reason in actions["skipped"]:
        pass  # reasons themselves may be long; what matters is the DB write
    from src.db import Signal, get_session
    with get_session(hermes.db_url) as s:
        for row in s.query(Signal).all():
            if row.rejection_reason is not None:
                assert len(row.rejection_reason) <= 100, (
                    f"rejection_reason {len(row.rejection_reason)} chars "
                    f"exceeds String(100): {row.rejection_reason!r}")


def test_market_session_respects_oms_disabled(hermes):
    hermes.run_pre_market_sequence()
    hermes.begin_close_management()  # disables OMS
    actions = hermes.run_market_session()
    assert actions["executed"] == []
    assert "oms disabled" in actions["skipped"]


def test_market_session_blocked_by_loss_guard(hermes):
    hermes.run_pre_market_sequence()
    for _ in range(3):
        hermes.loss_guard.record_trade_outcome(-100)
    actions = hermes.run_market_session()
    assert actions["executed"] == []
    assert any("loss guard" in s for s in actions["skipped"])


def test_post_market_updates_credibility(hermes):
    hermes.run_pre_market_sequence()
    hermes.run_post_market_sequence()
    assert hermes.brain.get_agent_credibility("technical") == pytest.approx(1.05)
    assert hermes.brain.get_agent_credibility("sentiment") == pytest.approx(0.95)
    # pre-market intelligence + market-close trade details + EOD daily report
    assert len(hermes.telegram.messages) == 3
    assert any("PRE-MARKET" in m for m in hermes.telegram.messages)
    assert any("TODAY'S TRADES" in m or "No trades executed" in m
              for m in hermes.telegram.messages)
    assert any("DAILY REPORT" in m for m in hermes.telegram.messages)


def test_post_market_pushes_real_trade_details_at_close(hermes):
    """The market-close push (docs: sent the moment the market closes, not
    the 17:00 compact summary) must carry actual per-trade content — entry,
    exit, P&L, exit reason — not just fire an empty notification."""
    hermes.brain.remember_trade({
        "signal_hash": "TCS-close-test", "ticker": "TCS.NS",
        "direction": "LONG", "entry_price": 4000.0, "exit_price": 4100.0,
        "quantity": 2, "pnl": 190.0, "exit_reason": "TAKE_PROFIT",
        "signal_score": 10, "is_paper": True,
    })
    hermes.run_post_market_sequence()
    trade_msg = next(m for m in hermes.telegram.messages
                     if "TODAY'S TRADES" in m)
    assert "TCS.NS" in trade_msg
    assert "4,000.00" in trade_msg and "4,100.00" in trade_msg
    assert "TAKE_PROFIT" in trade_msg
    assert "190.00" in trade_msg or "+190" in trade_msg


def test_pre_market_calls_all_research_agents(hermes):
    hermes.run_pre_market_sequence()
    for name, agent in hermes.research_team.items():
        assert agent.calls, f"{name} was not triggered"
    assert any("PRE-MARKET INTELLIGENCE" in m
               for m in hermes.telegram.messages)


def test_pre_market_respects_earnings_blackout(hermes):
    hermes.research_team["company_analysis_agent"].payload = {
        "earnings_blackout": [t for t in
                              __import__("src.utils.universe",
                                         fromlist=["UNIVERSE"]).UNIVERSE[:1]]
    }
    result = hermes.run_pre_market_sequence()
    blocked = hermes.research_team["company_analysis_agent"].payload[
        "earnings_blackout"]
    assert all(t not in result["watchlist"] for t in blocked)


def test_overnight_batch_skips_missing_handlers(hermes):
    result = hermes.run_overnight_batch()
    assert any("skipped" in s for s in result["steps"])


def test_full_daily_dry_run(hermes):
    """Hermes coordinates the complete day without raising."""
    hermes.run_pre_market_sequence()
    hermes.arm_system()
    hermes.observe_market_open()
    hermes.start_trading_session()
    hermes.run_market_session()
    hermes.begin_close_management()
    hermes.run_post_market_sequence()
    hermes.send_eod_report()
    hermes.run_overnight_batch()
    health = hermes.health_check()
    assert health["db"] is True
