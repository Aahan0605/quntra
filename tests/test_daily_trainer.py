"""Tests for src/learning/daily_trainer.py — offline, mock trades."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.learning import DailyTrainer


class FakeBrain:
    def __init__(self, trades):
        self.trades = trades
        self.credibility_updates = []

    def get_recent_trades(self, days=90):
        return self.trades

    def update_agent_credibility(self, name, correct):
        self.credibility_updates.append((name, correct))


class FakeTelegram:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return True


def _mock_trades(n=60, tickers=("RELIANCE.NS", "TCS.NS")):
    """n closed trades spread over the past ~n/2 business days."""
    rng = np.random.default_rng(7)
    now = datetime.now(timezone.utc)
    trades = []
    for i in range(n):
        entry = now - timedelta(days=(n - i) // 2 + 1)
        trades.append({
            "id": f"t{i}",
            "ticker": tickers[i % len(tickers)],
            "direction": "LONG",
            "pnl": float(rng.normal(50, 200)),
            "entry_time": entry,
            "exit_time": entry + timedelta(hours=6),
            "signal_score": 9,
            "regime": "BULL",
            "is_paper": True,
        })
    return trades


def _mock_features():
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=300)
    rng = np.random.default_rng(3)
    cols = ["ret_1d", "ret_5d", "vol_10d", "rsi_14", "mom_10d"]
    return pd.DataFrame(rng.normal(0, 1, (300, len(cols))),
                        index=idx, columns=cols)


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    import src.learning.daily_trainer as dt
    monkeypatch.setattr(dt, "RESEARCH_DIR", tmp_path / "research")
    monkeypatch.setattr(dt, "PRODUCTION_DIR", tmp_path / "prod")
    return DailyTrainer(FakeBrain(_mock_trades()), telegram=FakeTelegram())


def test_skips_with_insufficient_trades():
    t = DailyTrainer(FakeBrain(_mock_trades(5)), telegram=FakeTelegram())
    report = t.run()
    assert report.skipped
    assert "insufficient" in report.reason
    assert t.telegram.sent  # skip is still announced


def test_full_run_with_mock_trades(trainer):
    feats = _mock_features()
    with patch("src.ml.train_clean_models.build_features",
               return_value=feats), \
         patch("src.utils.cache_loader.load_ticker",
               return_value=pd.DataFrame({"close": [1]})), \
         patch("src.utils.cache_loader.load_benchmark", return_value=None), \
         patch.object(DailyTrainer, "log_to_mlflow") as mlflow_log:
        report = trainer.run()

    assert not report.skipped
    assert report.accuracy is not None
    assert report.baseline is not None
    mlflow_log.assert_called_once()
    assert trainer.telegram.sent
    assert "Daily training" in trainer.telegram.sent[0]


def test_gate_blocks_deployment(trainer, tmp_path):
    feats = _mock_features()
    with patch("src.ml.train_clean_models.build_features",
               return_value=feats), \
         patch("src.utils.cache_loader.load_ticker",
               return_value=pd.DataFrame({"close": [1]})), \
         patch("src.utils.cache_loader.load_benchmark", return_value=None), \
         patch.object(DailyTrainer, "log_to_mlflow"):
        report = trainer.run()

    # Random features vs random labels: deployment iff gate truly cleared
    research = tmp_path / "research"
    deployed_files = list(research.glob("*.pkl")) if research.exists() else []
    assert bool(deployed_files) == report.deployed
    if report.deployed:
        assert report.accuracy >= DailyTrainer.ACCURACY_GATE


def test_research_env_separate_from_production(trainer, tmp_path):
    """Deployment never writes into the production model directory."""
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=2)
    m.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    trainer.deploy_to_research_env(m, 0.60)
    assert list((tmp_path / "research").glob("trade_model_*.pkl"))
    prod = tmp_path / "prod"
    assert not prod.exists() or not list(prod.glob("trade_model_*.pkl"))


def test_credibility_updated_for_todays_exits(trainer):
    trades = _mock_trades(40)
    now = datetime.now(timezone.utc)
    trades[-1]["exit_time"] = now
    trades[-1]["pnl"] = 100.0
    trades[-2]["exit_time"] = now
    trades[-2]["pnl"] = -50.0
    trainer.brain = FakeBrain(trades)
    trainer.update_agent_credibility(trades)
    assert ("signal_council", True) in trainer.brain.credibility_updates
    assert ("signal_council", False) in trainer.brain.credibility_updates
