"""
SignalCouncil — the multi-agent vote that scores tickers 0-12.

Five voters, each with one responsibility:
  technical (0-3): trend vs SMA20/50 + RSI positioning, from the cache
  momentum  (0-3): 20d relative strength vs NIFTY
  ml        (0-2): deployed per-ticker model P(up over 5d); NEUTRAL (1)
                   when no model passed the honest OOS gate
  macro     (0-2): pre-market macro bias (POSITIVE 2 / NEUTRAL 1 / NEG 0)
  sector    (0-2): ticker's sector in today's leaders/laggards

Watchlist gate: score >= 9/12 (MIN_WATCHLIST_SCORE in hermes).
Long-only by design — shorting needs different risk plumbing (Phase 3+).
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("quntra.council")

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "models"


class SignalCouncil:
    """Deterministic, auditable scoring — every vote is logged."""

    MAX_TRADES_PER_DAY = 3

    def __init__(self, db_url: str | None = None, capital: float = 25_000.0):
        self.db_url = db_url
        self.capital = capital
        self._models: dict[str, object] = {}
        self._deployed_loaded = False
        self._trades_today: dict[str, int] = {}  # date-iso -> count

    # ------------------------------------------------------------------ #
    # Pre-market scoring

    def score_premarket(self, universe: list[str]) -> dict[str, int]:
        """Score every ticker 0-12; Hermes gates at >= 9 for the watchlist."""
        from src.utils.cache_loader import load_benchmark, load_close_panel

        try:
            panel = load_close_panel(universe)
        except FileNotFoundError:
            logger.error("No price cache — council cannot score")
            return {}
        try:
            bench = load_benchmark()
        except Exception:  # noqa: BLE001
            bench = None

        macro_vote = self._macro_vote()
        sector_votes = self._sector_votes(panel)
        scores: dict[str, int] = {}
        details: dict[str, dict] = {}
        for ticker in panel.columns:
            series = panel[ticker].dropna()
            if len(series) < 60:
                continue
            votes = {
                "technical": self._technical_vote(series),
                "momentum": self._momentum_vote(series, bench),
                "ml": self._ml_vote(ticker),
                "macro": macro_vote,
                "sector": sector_votes.get(ticker, 1),
            }
            scores[ticker] = int(sum(votes.values()))
            details[ticker] = votes

        self._log_scores(scores, details)
        return scores

    @staticmethod
    def _technical_vote(close: pd.Series) -> int:
        """0-3: above SMA20, above SMA50, RSI in the healthy 45-70 band."""
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 100.0
        px = close.iloc[-1]
        return int(px > sma20) + int(px > sma50) + int(45 <= rsi <= 70)

    @staticmethod
    def _momentum_vote(close: pd.Series, bench: pd.Series | None) -> int:
        """0-3: 20d return positive, beats NIFTY, 5d confirms."""
        r20 = close.iloc[-1] / close.iloc[-21] - 1
        r5 = close.iloc[-1] / close.iloc[-6] - 1
        vote = int(r20 > 0) + int(r5 > 0)
        if bench is not None and len(bench) > 21:
            b20 = bench.iloc[-1] / bench.iloc[-21] - 1
            vote += int(r20 > b20)
        return min(vote, 3)

    def _ml_vote(self, ticker: str) -> int:
        """0-2 from the deployed model's P(up over 5d); 1 (neutral) when
        no model passed the honest gate — coin-flip models don't vote."""
        model = self._deployed_model(ticker)
        if model is None:
            return 1
        try:
            from src.ml.train_clean_models import build_features
            from src.utils.cache_loader import load_benchmark, load_ticker
            feats = build_features(load_ticker(ticker), load_benchmark())
            row = feats.dropna().iloc[[-1]]
            proba = float(model.predict_proba(row)[0, 1])
            return 2 if proba >= 0.60 else (0 if proba <= 0.40 else 1)
        except Exception as e:  # noqa: BLE001
            logger.warning("ml vote failed for %s: %s", ticker, e)
            return 1

    def _macro_vote(self) -> int:
        from src.db import get_session, SystemState
        try:
            with get_session(self.db_url) as s:
                pre = s.get(SystemState, "premarket")
                bias = (pre.value or {}).get("macro_bias") if pre else None
                note = s.get(SystemState, "user_note_bias")
                caution = bool(note and (note.value or {}).get("bias")
                               == "CAUTION")
        except Exception:  # noqa: BLE001
            return 1
        base = {"POSITIVE": 2, "NEGATIVE": 0}.get(bias, 1)
        return max(0, base - (1 if caution else 0))

    @staticmethod
    def _sector_votes(panel: pd.DataFrame) -> dict[str, int]:
        from src.agents.research.sector_agent import SECTOR_MAP
        moves: dict[str, list[float]] = {}
        for t in panel.columns:
            sec = SECTOR_MAP.get(t)
            s = panel[t].dropna()
            if sec and len(s) > 6:
                moves.setdefault(sec, []).append(
                    float(s.iloc[-1] / s.iloc[-6] - 1))
        ranked = sorted(moves, key=lambda k: -np.mean(moves[k]))
        leaders, laggards = set(ranked[:2]), set(ranked[-2:])
        return {t: (2 if SECTOR_MAP.get(t) in leaders else
                    0 if SECTOR_MAP.get(t) in laggards else 1)
                for t in panel.columns}

    # ------------------------------------------------------------------ #
    # Session signals

    def live_signals(self, watchlist: list[str]) -> list[dict]:
        """Executable signals for watchlist tickers, respecting the
        daily trade cap. Long-only; qty sized as capital/3 per position."""
        if not watchlist:
            return []
        today = datetime.now(timezone.utc).date().isoformat()
        used = self._trades_today.get(today, 0)
        budget = self.MAX_TRADES_PER_DAY - used
        if budget <= 0:
            return []

        from src.utils.cache_loader import load_ticker
        signals = []
        for ticker in watchlist[:budget]:
            try:
                px = float(load_ticker(ticker)["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                continue
            per_trade = self.capital / self.MAX_TRADES_PER_DAY
            qty = max(1, int(per_trade / px)) if px < per_trade else 0
            if qty == 0:
                logger.info("%s too expensive for per-trade budget "
                            "₹%.0f (px %.2f)", ticker, per_trade, px)
                continue
            signals.append({
                "ticker": ticker,
                "direction": "LONG",
                "qty": qty,
                "score": 9,  # gate floor — council already filtered
                "signal_hash": f"{ticker}-{today}",  # one entry/ticker/day
            })
        self._trades_today[today] = used + len(signals)
        return signals

    def score_day(self) -> list[tuple[str, bool]]:
        """Score voters against today's realized market direction."""
        try:
            from src.utils.cache_loader import load_benchmark
            bench = load_benchmark()
            up_day = bool(bench.iloc[-1] > bench.iloc[-2])
        except Exception:  # noqa: BLE001
            return []
        macro = self._macro_vote()
        outcomes = []
        if macro != 1:  # only score conviction calls
            outcomes.append(("macro", (macro == 2) == up_day))
        return outcomes

    # ------------------------------------------------------------------ #

    def _deployed_model(self, ticker: str):
        if not self._deployed_loaded:
            summary = MODEL_DIR / "training_summary.json"
            if summary.exists():
                for r in json.loads(summary.read_text()):
                    if not r.get("passed_gate"):
                        continue
                    stem = r["ticker"].replace("&", "_")
                    try:
                        with open(MODEL_DIR / f"{stem}.pkl", "rb") as fp:
                            self._models[r["ticker"]] = pickle.load(fp)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("could not load model %s: %s",
                                       r["ticker"], e)
            self._deployed_loaded = True
        return self._models.get(ticker)

    def _log_scores(self, scores: dict, details: dict) -> None:
        """Every scoring run is auditable in the signals table."""
        try:
            from src.db import Signal, get_session
            now = datetime.now(timezone.utc)
            top = sorted(scores.items(), key=lambda x: -x[1])[:5]
            with get_session(self.db_url) as s:
                for ticker, score in top:
                    s.add(Signal(
                        ticker=ticker, score=score, direction="LONG",
                        agent_votes=details.get(ticker),
                        reasoning="council premarket scoring",
                        executed=False, signal_time=now,
                        signal_hash=f"score-{ticker}-{now.date()}",
                    ))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not log council scores: %s", e)
