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
    "us_10y": "^TNX",
    "vix": "^VIX",
    "nikkei": "^N225",
    "hang_seng": "^HSI",
}

# VIX is a level, not a %change — fetched separately
VIX_FEAR = 25.0
VIX_CALM = 15.0


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
        vix_level = self._vix_level()
        fii_dii = self._fii_dii()

        bias, reasons = self._compute_bias(moves, fii_dii, vix_level)
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
            payload={"macro_bias": bias, "moves": moves, "fii_dii": fii_dii,
                     "vix_level": vix_level,
                     "us_direction": self._direction(moves, ("sp500",
                                                             "nasdaq", "dow")),
                     "asia_direction": self._direction(moves, ("nikkei",
                                                               "hang_seng"))},
        )

    @staticmethod
    def _vix_level() -> float | None:
        """VIX absolute level (a %change of a fear index is meaningless)."""
        try:
            import yfinance as yf
            h = yf.Ticker("^VIX").history(period="2d")
            if len(h):
                return round(float(h["Close"].iloc[-1]), 2)
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _direction(moves: dict, keys: tuple) -> str:
        vals = [moves.get(k) for k in keys if moves.get(k) is not None]
        if not vals:
            return "UNKNOWN"
        avg = sum(vals) / len(vals)
        return "UP" if avg > 0.3 else ("DOWN" if avg < -0.3 else "FLAT")

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
    def _compute_bias(moves: dict, fii_dii: dict | None,
                      vix_level: float | None = None) -> tuple[str, list[str]]:
        """Score the tape: US equities lead, oil is a headwind for India,
        VIX extremes override mild equity signals."""
        score = 0.0
        reasons: list[str] = []

        if vix_level is not None:
            if vix_level > VIX_FEAR:
                score -= 1
                reasons.append(f"VIX elevated at {vix_level}")
            elif vix_level < VIX_CALM:
                score += 0.5
                reasons.append(f"VIX calm at {vix_level}")

        asia = [moves.get("nikkei"), moves.get("hang_seng")]
        asia = [m for m in asia if m is not None]
        if asia:
            avg_asia = sum(asia) / len(asia)
            if abs(avg_asia) > 0.5:
                score += 0.5 if avg_asia > 0 else -0.5
                reasons.append(f"Asia {'up' if avg_asia > 0 else 'down'} "
                               f"{avg_asia:+.2f}%")

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
