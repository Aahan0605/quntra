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
    MAX_POSITION_PCT = 0.10    # hard cap on any single name, as % of capital

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
        news_sentiment = self._news_ticker_sentiment()   # research overlay
        fundamental_flagged = self._fundamental_flagged()  # veto set
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
                # per-stock research overlay: news tilts ±1, weak
                # fundamentals veto -1 (never a bonus)
                "news": self._news_vote(ticker, news_sentiment),
                "fundamental": self._fundamental_vote(ticker,
                                                      fundamental_flagged),
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

    def _latest_note_payload(self, source: str, hours: int = 48) -> dict:
        """Most recent research_note payload for a source, or {}."""
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import select
        from src.db import ResearchNote, get_session
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            with get_session(self.db_url) as s:
                row = s.execute(
                    select(ResearchNote)
                    .where(ResearchNote.source == source,
                           ResearchNote.created_at >= cutoff)
                    .order_by(ResearchNote.created_at.desc())
                    .limit(1)
                ).scalar_one_or_none()
                return (row.entities or {}) if row else {}
        except Exception:  # noqa: BLE001
            return {}

    def _news_ticker_sentiment(self) -> dict[str, float]:
        return self._latest_note_payload("news_agent").get(
            "ticker_sentiment", {}) or {}

    def _fundamental_flagged(self) -> set[str]:
        flagged = self._latest_note_payload(
            "fundamental_agent", hours=24 * 8).get("flagged", []) or []
        return {f.get("ticker") for f in flagged if isinstance(f, dict)}

    @staticmethod
    def _news_vote(ticker: str, sentiment_map: dict[str, float]) -> int:
        """+1 clearly positive news, -1 clearly negative, else 0. A research
        overlay so a stock's own headlines tilt its score."""
        s = sentiment_map.get(ticker)
        if s is None:
            return 0
        if s >= 0.3:
            return 1
        if s <= -0.3:
            return -1
        return 0

    @staticmethod
    def _fundamental_vote(ticker: str, flagged: set[str]) -> int:
        """-1 when the fundamental agent flagged weak valuation/balance sheet
        (high P/E, high debt, low ROE), else 0. Never a bonus — fundamentals
        can veto, not inflate."""
        return -1 if ticker in flagged else 0

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
        daily trade cap. Long-only; qty sized as capital/3 per position.

        Earnings-blacklisted tickers are dropped defensively here too —
        Hermes already filters them from the watchlist, but reading the
        persisted system_state['earnings_blacklist'] means a signal never
        fires into a company's results even if the watchlist path is
        bypassed."""
        if not watchlist:
            return []
        blacklist = set(self._earnings_blacklist())
        watchlist = [t for t in watchlist if t not in blacklist]
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
            # capital/MAX_TRADES_PER_DAY alone put 33% of the book in one
            # name — three positions were 88% of capital in three large caps
            # that correlate to ~1 in a crash. Cap each position separately.
            per_trade = min(self.capital / self.MAX_TRADES_PER_DAY,
                            self.capital * self.MAX_POSITION_PCT)
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

    def _earnings_blacklist(self) -> list[str]:
        """Tickers in an earnings blackout, from CompanyAnalysisAgent's
        persisted system_state entry. Empty on any read failure."""
        try:
            from src.db import SystemState, get_session
            with get_session(self.db_url) as s:
                row = s.get(SystemState, "earnings_blacklist")
                return (row.value or {}).get("tickers", []) if row else []
        except Exception:  # noqa: BLE001
            return []

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

    def _fdr_survivors(self) -> set[str] | None:
        """Tickers whose edge survived the multiple-testing correction.

        `passed_gate` is a per-model verdict; running it over 194 tickers
        makes ~10 passes inevitable from noise. None means the audit has
        not been run, in which case we fall back to the naive gate rather
        than silently trading nothing.
        """
        audit = MODEL_DIR / "multiple_testing.json"
        if not audit.exists():
            logger.warning("no multiple_testing.json in %s — ML votes rest on "
                           "the uncorrected gate; run src.ml.multiple_testing",
                           MODEL_DIR)
            return None
        res = json.loads(audit.read_text())
        return {s["ticker"] for s in res.get("survivors", [])}

    def _deployed_model(self, ticker: str):
        if not self._deployed_loaded:
            survivors = self._fdr_survivors()
            summary = MODEL_DIR / "training_summary.json"
            if summary.exists():
                for r in json.loads(summary.read_text()):
                    if not r.get("passed_gate"):
                        continue
                    if survivors is not None and r["ticker"] not in survivors:
                        logger.info("%s passed the naive gate but not the FDR "
                                    "correction — ML vote stays NEUTRAL",
                                    r["ticker"])
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
            from sqlalchemy import select

            from src.db import Signal, get_session
            now = datetime.now(timezone.utc)
            top = sorted(scores.items(), key=lambda x: -x[1])[:5]
            with get_session(self.db_url) as s:
                for ticker, score in top:
                    # signal_hash is unique and date-keyed, so a second
                    # pre-market run in the same day (cron at 06:00 plus a
                    # manual /start_trading, which happened 2026-08-06)
                    # raised UniqueViolation and lost the WHOLE batch's
                    # audit trail, not just the duplicate row. Today's
                    # scores are a snapshot, so re-running should refresh
                    # them rather than fail.
                    h = f"score-{ticker}-{now.date()}"
                    row = s.execute(
                        select(Signal).where(Signal.signal_hash == h)
                    ).scalar_one_or_none()
                    if row is not None:
                        row.score = score
                        row.agent_votes = details.get(ticker)
                        row.signal_time = now
                        continue
                    s.add(Signal(
                        ticker=ticker, score=score, direction="LONG",
                        agent_votes=details.get(ticker),
                        reasoning="council premarket scoring",
                        executed=False, signal_time=now,
                        signal_hash=h,
                    ))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not log council scores: %s", e)
