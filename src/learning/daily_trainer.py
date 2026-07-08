"""
DailyTrainer — the nightly self-learning loop (22:00 IST).

Learns from actual trade outcomes: features at each trade's entry date,
label = trade was profitable. Refits on a rolling 90-day window, validates
on the last 10 trading days, and deploys to the RESEARCH environment only
when holdout accuracy clears the gate. Production models are never touched
by this loop — promotion research -> production is a manual, reviewed step.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("quntra.trainer")

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "data" / "models" / "research"
PRODUCTION_DIR = ROOT / "data" / "models"


@dataclass
class TrainingReport:
    skipped: bool = False
    reason: str = ""
    accuracy: float | None = None
    baseline: float | None = None
    deployed: bool = False
    n_trades: int = 0
    drift_warning: str | None = None
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> str:
        if self.skipped:
            return f"Daily training skipped: {self.reason}"
        state = "DEPLOYED to research" if self.deployed else \
            "BELOW GATE, kept existing"
        msg = (f"Daily training: {self.n_trades} trades, holdout accuracy "
               f"{self.accuracy:.2%} (baseline {self.baseline:.2%}) → {state}")
        if self.drift_warning:
            msg += f" | DRIFT: {self.drift_warning}"
        return msg


class DailyTrainer:
    """Nightly trade-outcome learner. See module docstring."""

    ACCURACY_GATE = 0.54
    ROLLING_WINDOW = 90     # calendar days of trade history
    HOLDOUT_DAYS = 10       # most recent trading days held out
    MIN_TRADES = 30
    # EWC-style drift guard: flag when the new model disagrees with the
    # previous research model on more than this share of the holdout
    MAX_PREDICTION_DRIFT = 0.40

    def __init__(self, brain, db_url: str | None = None, telegram=None):
        self.brain = brain
        self.db_url = db_url
        self.telegram = telegram

    # ------------------------------------------------------------------ #

    def run(self) -> TrainingReport:
        trades = self.brain.get_recent_trades(days=self.ROLLING_WINDOW)
        trades = [t for t in trades if t.get("pnl") is not None
                  and t.get("entry_time") is not None]
        if len(trades) < self.MIN_TRADES:
            report = TrainingReport(
                skipped=True, n_trades=len(trades),
                reason=f"insufficient trade history "
                       f"({len(trades)}/{self.MIN_TRADES} closed trades)")
            self._notify(report)
            return report

        X, y, entry_dates = self.build_training_set(trades)
        if len(X) < self.MIN_TRADES:
            report = TrainingReport(
                skipped=True, n_trades=len(trades),
                reason=f"only {len(X)} trades had usable features")
            self._notify(report)
            return report

        X_tr, y_tr, X_ho, y_ho = self._split_by_date(X, y, entry_dates)
        if len(X_ho) < 5:
            report = TrainingReport(
                skipped=True, n_trades=len(trades),
                reason=f"holdout too small ({len(X_ho)} trades in last "
                       f"{self.HOLDOUT_DAYS} trading days)")
            self._notify(report)
            return report

        model = self._fit(X_tr, y_tr)
        preds = model.predict(X_ho)
        acc = float((preds == y_ho).mean())
        baseline = float(max(y_ho.mean(), 1 - y_ho.mean()))
        drift = self._drift_check(X_ho, preds)

        deployed = acc >= self.ACCURACY_GATE and drift is None
        if deployed:
            self.deploy_to_research_env(model, acc)
        self.log_to_mlflow(acc, baseline, len(X), deployed)
        self.update_agent_credibility(trades)

        report = TrainingReport(accuracy=acc, baseline=baseline,
                                deployed=deployed, n_trades=len(trades),
                                drift_warning=drift)
        self._notify(report)
        return report

    # ------------------------------------------------------------------ #
    # Dataset

    def build_training_set(self, trades: list[dict]
                           ) -> tuple[pd.DataFrame, pd.Series, list]:
        """Features at each trade's entry date; label = profitable."""
        from src.ml.train_clean_models import build_features
        from src.utils.cache_loader import load_benchmark, load_ticker

        try:
            bench = load_benchmark()
        except Exception:  # noqa: BLE001
            bench = None

        feat_cache: dict[str, pd.DataFrame] = {}
        rows, labels, dates = [], [], []
        for t in trades:
            ticker = t["ticker"]
            if ticker not in feat_cache:
                try:
                    feat_cache[ticker] = build_features(load_ticker(ticker),
                                                        bench)
                except FileNotFoundError:
                    feat_cache[ticker] = pd.DataFrame()
            feats = feat_cache[ticker]
            if feats.empty:
                continue
            entry = pd.Timestamp(t["entry_time"]).tz_localize(None).normalize()
            idx = feats.index[feats.index <= entry]
            if len(idx) == 0:
                continue
            row = feats.loc[idx[-1]]
            if row.isna().any():
                continue
            rows.append(row)
            labels.append(1 if t["pnl"] > 0 else 0)
            dates.append(entry)

        if not rows:
            return pd.DataFrame(), pd.Series(dtype=int), []
        return pd.DataFrame(rows).reset_index(drop=True), \
            pd.Series(labels), dates

    def _split_by_date(self, X, y, dates):
        """Last HOLDOUT_DAYS distinct trading days become the holdout."""
        unique_days = sorted(set(dates))
        holdout_days = set(unique_days[-self.HOLDOUT_DAYS:])
        mask = np.array([d in holdout_days for d in dates])
        return X[~mask], y[~mask], X[mask], y[mask]

    # ------------------------------------------------------------------ #
    # Model

    def load_production_model(self):
        """Latest research model if present, else any production pickle."""
        for d in (RESEARCH_DIR, PRODUCTION_DIR):
            if d.exists():
                pkls = sorted(d.glob("trade_model_*.pkl"), reverse=True) \
                    or sorted(d.glob("*.pkl"))
                for p in pkls:
                    try:
                        with open(p, "rb") as fp:
                            return pickle.load(fp)
                    except Exception:  # noqa: BLE001
                        continue
        return None

    def _fit(self, X, y):
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, min_child_weight=5,
            eval_metric="logloss", n_jobs=2, random_state=42,
        )
        model.fit(X, y)
        return model

    def _drift_check(self, X_ho, new_preds) -> str | None:
        """Flag catastrophic forgetting: new model wildly disagreeing
        with the previous research model on identical inputs."""
        prev = self.load_production_model()
        if prev is None:
            return None
        try:
            prev_preds = prev.predict(X_ho)
            disagreement = float((prev_preds != new_preds).mean())
            if disagreement > self.MAX_PREDICTION_DRIFT:
                return (f"{disagreement:.0%} prediction flip vs previous "
                        f"model (max {self.MAX_PREDICTION_DRIFT:.0%}) — "
                        f"not deployed")
        except Exception:  # noqa: BLE001 — feature-set changed, skip check
            return None
        return None

    # ------------------------------------------------------------------ #
    # Deployment / logging

    def deploy_to_research_env(self, model, acc: float) -> Path:
        RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = RESEARCH_DIR / f"trade_model_{stamp}.pkl"
        with open(path, "wb") as fp:
            pickle.dump(model, fp)
        (RESEARCH_DIR / f"trade_model_{stamp}.meta.json").write_text(
            json.dumps({"holdout_accuracy": acc,
                        "deployed_at": datetime.now(timezone.utc).isoformat(),
                        "environment": "research"}, indent=2))
        return path

    def log_to_mlflow(self, acc: float, baseline: float, n: int,
                      deployed: bool) -> None:
        try:
            import os

            import mlflow
            # Respect MLFLOW_TRACKING_URI when set; else a local SQLite DB.
            # (Recent MLflow deprecated the plain-file backend.)
            uri = os.getenv("MLFLOW_TRACKING_URI") \
                or f"sqlite:///{ROOT / 'data' / 'mlflow.db'}"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("quntra_daily_trainer")
            with mlflow.start_run():
                mlflow.log_metrics({"holdout_accuracy": acc,
                                    "holdout_baseline": baseline,
                                    "n_samples": n})
                mlflow.log_params({"gate": self.ACCURACY_GATE,
                                   "window_days": self.ROLLING_WINDOW,
                                   "deployed": deployed})
        except Exception as e:  # noqa: BLE001
            logger.warning("MLflow logging failed: %s", e)

    def update_agent_credibility(self, trades: list[dict]) -> None:
        """Today's closed trades feed the council's credibility scores."""
        today = datetime.now(timezone.utc).date()
        for t in trades:
            exit_time = t.get("exit_time")
            if exit_time is None or pd.Timestamp(exit_time).date() != today:
                continue
            correct = (t.get("pnl") or 0) > 0
            self.brain.update_agent_credibility("signal_council", correct)

    def _notify(self, report: TrainingReport) -> None:
        logger.info(report.summary())
        if self.telegram is not None:
            try:
                self.telegram.send(f"📊 {report.summary()}")
            except Exception:  # noqa: BLE001
                pass
