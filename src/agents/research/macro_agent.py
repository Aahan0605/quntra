"""
MacroAgent — global macro snapshot before the NSE open.

Watches: US indices (prev close), crude oil, gold, USD/INR, and FII/DII
flows when NSE cooperates. Produces macro_bias POSITIVE/NEGATIVE/NEUTRAL
with explicit reasoning. All sources degrade independently.
"""

from __future__ import annotations

from src.agents.research.base import BaseResearchAgent, ResearchOutput, yf_pct_change

MACRO_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "crude_oil": "CL=F",
    "gold": "GC=F",
    "usdinr": "USDINR=X",
}


class MacroAgent(BaseResearchAgent):
    name = "macro_agent"
    description = "US/EU/Asia markets, commodities, FX, FII/DII flows"
    note_type = "macro"

    def __init__(self, db_url: str | None = None, fetcher=None):
        super().__init__(db_url)
        self.fetcher = fetcher  # UnifiedDataFetcher for FII/DII (optional)

    def run(self, context: dict) -> ResearchOutput:
        moves: dict[str, float | None] = {
            name: yf_pct_change(tick) for name, tick in MACRO_TICKERS.items()
        }
        fii_dii = self._fii_dii()

        bias, reasons = self._compute_bias(moves, fii_dii)
        available = {k: v for k, v in moves.items() if v is not None}
        summary = (f"Macro bias: {bias}. " + "; ".join(reasons)
                   if reasons else f"Macro bias: {bias} (no strong signals)")

        return ResearchOutput(
            agent=self.name,
            summary=summary,
            findings=[{"indicator": k, "pct_change": v}
                      for k, v in moves.items()],
            confidence=min(0.9, 0.15 * len(available)),
            sources=["yfinance"] + (["nse_fii_dii"] if fii_dii else []),
            reasoning="; ".join(reasons),
            payload={"macro_bias": bias, "moves": moves, "fii_dii": fii_dii},
        )

    def _fii_dii(self) -> dict | None:
        """FII/DII net flows via UnifiedDataFetcher, when available."""
        if self.fetcher is None:
            return None
        try:
            getter = getattr(self.fetcher, "get_fii_dii_data", None)
            if callable(getter):
                return getter()
        except Exception:  # noqa: BLE001 — NSE blocks often; not critical
            pass
        return None

    @staticmethod
    def _compute_bias(moves: dict, fii_dii: dict | None) -> tuple[str, list[str]]:
        """Score the tape: US equities lead, oil is a headwind for India."""
        score = 0.0
        reasons: list[str] = []

        us = [moves.get("sp500"), moves.get("nasdaq"), moves.get("dow")]
        us = [m for m in us if m is not None]
        if us:
            avg_us = sum(us) / len(us)
            if avg_us > 0.3:
                score += 1
                reasons.append(f"US equities up {avg_us:+.2f}%")
            elif avg_us < -0.3:
                score -= 1
                reasons.append(f"US equities down {avg_us:+.2f}%")

        oil = moves.get("crude_oil")
        if oil is not None and abs(oil) > 1.5:
            # India imports ~85% of its crude — spikes are negative
            score += -1 if oil > 0 else 0.5
            reasons.append(f"crude {'spike' if oil > 0 else 'relief'} {oil:+.2f}%")

        inr = moves.get("usdinr")
        if inr is not None and abs(inr) > 0.3:
            score += -0.5 if inr > 0 else 0.5
            reasons.append(f"rupee {'weakening' if inr > 0 else 'strengthening'} "
                           f"{inr:+.2f}%")

        if fii_dii:
            net_fii = fii_dii.get("fii_net")
            if isinstance(net_fii, (int, float)) and abs(net_fii) > 1000:
                score += 0.5 if net_fii > 0 else -0.5
                reasons.append(f"FII net {net_fii:+,.0f} Cr")

        if score >= 1:
            return "POSITIVE", reasons
        if score <= -1:
            return "NEGATIVE", reasons
        return "NEUTRAL", reasons
