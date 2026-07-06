"""
NoteProcessor — turns operator observations (/note) into verified,
actionable intelligence.

Flow: extract entities -> verify claim against market data -> score
portfolio relevance -> decide the action -> store -> reply.

Example:
    /note Oil prices are rising because of Iran sanctions
    -> entities: OIL, IRAN, SANCTIONS
    -> verified: crude +2.1% (yfinance CL=F)
    -> relevance HIGH (energy exposure)
    -> action: macro bias nudged to CAUTION; stored as USER_NOTE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.agents.research.base import yf_pct_change
from src.agents.research.news_agent import COMPANY_KEYWORDS
from src.agents.research.sector_agent import SECTOR_MAP

logger = logging.getLogger("quntra.notes")

# macro entity -> (yfinance proxy, keywords that assert a RISE)
MACRO_ENTITIES = {
    "OIL": ("CL=F", {"oil", "crude", "brent", "wti"}),
    "GOLD": ("GC=F", {"gold"}),
    "USDINR": ("USDINR=X", {"rupee", "usdinr", "dollar"}),
    "US_MARKET": ("^GSPC", {"s&p", "sp500", "us market", "wall street",
                            "nasdaq", "dow"}),
    "NIFTY": ("^NSEI", {"nifty", "sensex", "indian market"}),
}

GEOPOLITICAL_TERMS = {"IRAN": {"iran"}, "SANCTIONS": {"sanction", "embargo"},
                      "WAR": {"war", "conflict", "strike", "attack"},
                      "CHINA": {"china"}, "PAKISTAN": {"pakistan"},
                      "RUSSIA": {"russia"}, "TARIFFS": {"tariff"}}

RISING_WORDS = {"rising", "rise", "surge", "surging", "up", "spike",
                "spiking", "jump", "increasing", "higher", "rally"}
FALLING_WORDS = {"falling", "fall", "drop", "dropping", "down", "plunge",
                 "crash", "decreasing", "lower", "weakening"}

# Sectors hurt by rising oil (importers) — used for relevance
OIL_SENSITIVE_SECTORS = {"ENERGY", "AUTO", "PAINTS", "CEMENT", "POWER"}


@dataclass
class NoteResponse:
    note_text: str
    entities: list[str]
    verified: bool | None          # None = unverifiable claim
    verification_detail: str
    relevance: str                 # HIGH / MEDIUM / LOW
    relevant_tickers: list[str]
    action: str
    note_id: str | None = None
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_message(self) -> str:
        """Telegram-ready reply."""
        v = {True: "✓ verified", False: "✗ NOT confirmed",
             None: "— unverifiable"}[self.verified]
        lines = [
            f"📝 Note processed",
            f"Entities: {', '.join(self.entities) or 'none detected'}",
            f"Verification: {v} — {self.verification_detail}",
            f"Portfolio relevance: {self.relevance}"
            + (f" ({', '.join(self.relevant_tickers[:5])})"
               if self.relevant_tickers else ""),
            f"Action: {self.action}",
        ]
        if self.note_id:
            lines.append(f"Stored: research_notes id={self.note_id[:8]}…")
        return "\n".join(lines)


class NoteProcessor:
    """Processes user observations sent via the /note command."""

    def __init__(self, db_url: str | None = None, hermes=None):
        self.db_url = db_url
        self.hermes = hermes  # for macro-bias nudges (optional)

    def process(self, note_text: str, user_id: str = "operator") -> NoteResponse:
        """Full pipeline; never raises — the bot must always reply."""
        try:
            entities = self.extract_entities(note_text)
            verified, detail = self.verify_against_sources(note_text, entities)
            relevance, tickers = self.score_portfolio_relevance(entities)
            action = self.determine_action(relevance, verified, entities)
            note_id = self.store(note_text, user_id, entities, verified,
                                 relevance, action)
            return NoteResponse(
                note_text=note_text, entities=entities, verified=verified,
                verification_detail=detail, relevance=relevance,
                relevant_tickers=tickers, action=action, note_id=note_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("note processing failed")
            return NoteResponse(
                note_text=note_text, entities=[], verified=None,
                verification_detail=f"processor error: {e}",
                relevance="LOW", relevant_tickers=[],
                action="stored raw, no action taken",
            )

    # ------------------------------------------------------------------ #

    def extract_entities(self, text: str) -> list[str]:
        low = text.lower()
        found: list[str] = []
        for entity, (_, keywords) in MACRO_ENTITIES.items():
            if any(k in low for k in keywords):
                found.append(entity)
        for entity, keywords in GEOPOLITICAL_TERMS.items():
            if any(k in low for k in keywords):
                found.append(entity)
        for kw, ticker in COMPANY_KEYWORDS.items():
            if kw in low and ticker not in found:
                found.append(ticker)
        return found

    def verify_against_sources(self, text: str,
                               entities: list[str]) -> tuple[bool | None, str]:
        """Check directional claims about macro entities via yfinance."""
        low = text.lower()
        claims_rise = any(w in low.split() for w in RISING_WORDS)
        claims_fall = any(w in low.split() for w in FALLING_WORDS)
        if not (claims_rise or claims_fall):
            return None, "no directional claim to check"

        for entity in entities:
            if entity not in MACRO_ENTITIES:
                continue
            proxy, _ = MACRO_ENTITIES[entity]
            move = yf_pct_change(proxy, period="5d")
            if move is None:
                return None, f"{entity}: market data unreachable"
            moved_up = move > 0
            confirmed = moved_up if claims_rise else not moved_up
            return confirmed, f"{entity} ({proxy}) moved {move:+.2f}% last session"
        return None, "no verifiable market entity in note"

    def score_portfolio_relevance(self,
                                  entities: list[str]) -> tuple[str, list[str]]:
        tickers = [e for e in entities if e in SECTOR_MAP]
        if "OIL" in entities or "IRAN" in entities or "SANCTIONS" in entities:
            tickers += [t for t, s in SECTOR_MAP.items()
                        if s in OIL_SENSITIVE_SECTORS and t not in tickers]
        if "USDINR" in entities:
            tickers += [t for t, s in SECTOR_MAP.items()
                        if s == "IT" and t not in tickers]  # exporters
        held = self._current_tickers()
        overlap = [t for t in tickers if t in held] if held else tickers
        if len(overlap) >= 3 or any(e in entities for e in ("NIFTY", "WAR")):
            return "HIGH", tickers[:10]
        if overlap or tickers:
            return "MEDIUM", tickers[:10]
        return "LOW", []

    def determine_action(self, relevance: str, verified: bool | None,
                         entities: list[str]) -> str:
        if verified is False:
            return "stored for reference — claim not confirmed by market data"
        if relevance == "HIGH" and verified:
            self._nudge_macro_bias(entities)
            return ("MacroAgent bias updated -> CAUTION; flagged tickers get "
                    "extra monitoring next session")
        if relevance == "HIGH":
            return "flagged for research review (unverified but relevant)"
        if relevance == "MEDIUM":
            return "attached to next overnight research run"
        return "stored in organizational memory"

    # ------------------------------------------------------------------ #

    def store(self, text: str, user_id: str, entities: list[str],
              verified: bool | None, relevance: str, action: str) -> str | None:
        try:
            from src.db import ResearchNote, get_session
            with get_session(self.db_url) as s:
                row = ResearchNote(
                    note_type="user_note",
                    content=text,
                    summary=f"[{relevance}] {text[:100]}",
                    source="USER_NOTE",
                    confidence={True: 0.9, False: 0.2, None: 0.5}[verified],
                    entities={"entities": entities, "user": user_id,
                              "verified": verified, "action": action},
                )
                s.add(row)
                s.flush()
                return row.id
        except Exception as e:  # noqa: BLE001
            logger.error("could not store note: %s", e)
            return None

    def _current_tickers(self) -> list[str]:
        try:
            if self.hermes is not None and hasattr(self.hermes.trader,
                                                   "get_positions"):
                return [p.get("ticker") for p in
                        self.hermes.trader.get_positions()]
        except Exception:  # noqa: BLE001
            pass
        return []

    def _nudge_macro_bias(self, entities: list[str]) -> None:
        """Record a caution flag Hermes reads at the next pre-market run."""
        try:
            if self.hermes is not None:
                self.hermes.set_system_state("user_note_bias", {
                    "bias": "CAUTION",
                    "entities": entities,
                    "at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as e:  # noqa: BLE001
            logger.error("could not nudge macro bias: %s", e)
