"""
SectorAgent — sector momentum and rotation over the 25-ticker universe.

Computes 5-day and 20-day momentum per sector from the local price cache
(no network needed), ranks sectors, and flags rotation.
"""

from __future__ import annotations

from src.agents.research.base import BaseResearchAgent, ResearchOutput
from src.utils import cache_loader
from src.utils.universe import UNIVERSE, UNIVERSE_SET

# The 25-name map (default). In nifty200 mode we merge in the full
# 200-name sector map so every traded ticker has a sector.
SECTOR_MAP = {
    "RELIANCE.NS": "ENERGY",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT",
    "HDFCBANK.NS": "BANKS", "ICICIBANK.NS": "BANKS", "SBIN.NS": "BANKS",
    "AXISBANK.NS": "BANKS", "KOTAKBANK.NS": "BANKS",
    "BAJFINANCE.NS": "FINANCE",
    "BHARTIARTL.NS": "TELECOM",
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG",
    "LT.NS": "INFRA",
    "MARUTI.NS": "AUTO", "M&M.NS": "AUTO", "TATAMOTORS.NS": "AUTO",
    "SUNPHARMA.NS": "PHARMA",
    "TITAN.NS": "CONSUMER",
    "ULTRACEMCO.NS": "CEMENT",
    "NTPC.NS": "POWER", "POWERGRID.NS": "POWER",
    "TATASTEEL.NS": "METALS",
    "ASIANPAINT.NS": "CONSUMER",
}

if UNIVERSE_SET == "nifty200":
    from src.utils.universe_nifty200 import NIFTY200_SECTOR_MAP
    # Nifty 200 map is authoritative for the wider set; keep the curated
    # 25-name labels where they differ (they're more granular).
    SECTOR_MAP = {**NIFTY200_SECTOR_MAP, **SECTOR_MAP}


class SectorAgent(BaseResearchAgent):
    name = "sector_agent"
    description = "sector momentum and rotation signals"
    note_type = "sector"

    def run(self, context: dict) -> ResearchOutput:
        closes = self._load_panel()
        if closes is None or closes.empty:
            return ResearchOutput(
                agent=self.name,
                summary="No price cache available — run fetch_data_cache.py",
                confidence=0.0,
                error="empty price panel",
            )

        sector_moves: dict[str, dict[str, list[float]]] = {}
        for ticker in closes.columns:
            sector = SECTOR_MAP.get(ticker)
            if sector is None or len(closes[ticker].dropna()) < 21:
                continue
            series = closes[ticker].dropna()
            m5 = float(series.iloc[-1] / series.iloc[-6] - 1) * 100
            m20 = float(series.iloc[-1] / series.iloc[-21] - 1) * 100
            bucket = sector_moves.setdefault(sector, {"m5": [], "m20": []})
            bucket["m5"].append(m5)
            bucket["m20"].append(m20)

        ranking = []
        for sector, m in sector_moves.items():
            avg5 = sum(m["m5"]) / len(m["m5"])
            avg20 = sum(m["m20"]) / len(m["m20"])
            ranking.append({
                "sector": sector,
                "momentum_5d": round(avg5, 2),
                "momentum_20d": round(avg20, 2),
                # rotation: short-term leadership diverging from trend
                "rotating_in": avg5 > 0 and avg5 > avg20 / 4,
            })
        ranking.sort(key=lambda x: -x["momentum_5d"])

        leaders = [r["sector"] for r in ranking[:2]]
        laggards = [r["sector"] for r in ranking[-2:]]
        summary = (f"Sector leaders: {', '.join(leaders)}; "
                   f"laggards: {', '.join(laggards)}")

        return ResearchOutput(
            agent=self.name,
            summary=summary,
            findings=ranking,
            confidence=0.8,
            sources=["local price cache"],
            reasoning="equal-weight 5d/20d momentum per sector",
            payload={"sector_ranking": ranking, "leaders": leaders,
                     "laggards": laggards},
        )

    @staticmethod
    def _load_panel():
        try:
            return cache_loader.load_close_panel(UNIVERSE)
        except Exception:  # noqa: BLE001
            return None
