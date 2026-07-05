"""
Transaction cost model for QuNtra.

Loads broker/exchange/regulatory charges from config/costs.env so the
backtest engine, paper trader, and live OMS all price friction identically.
Never hardcode cost assumptions elsewhere — import CostModel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "costs.env"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    return values


@dataclass(frozen=True)
class CostModel:
    """All-in cost model for NSE equity trading via ICICI Direct."""

    equity_delivery_bps: float = 0.0
    equity_intraday_flat_inr: float = 20.0
    stt_bps: float = 10.0
    exchange_bps: float = 3.0
    gst_pct: float = 18.0
    sebi_turnover_bps: float = 0.0001
    slippage_pct_per_side: float = 0.05

    @classmethod
    def from_config(cls, path: Path | str | None = None) -> "CostModel":
        p = Path(path) if path else _CONFIG_PATH
        env = _load_env_file(p)

        def get(key: str, default: float) -> float:
            # Environment variables override the file, file overrides defaults.
            raw = os.environ.get(key, env.get(key))
            return float(raw) if raw is not None else default

        return cls(
            equity_delivery_bps=get("ICICI_EQUITY_DELIVERY_BPS", 0.0),
            equity_intraday_flat_inr=get("ICICI_EQUITY_INTRADAY_FLAT_INR", 20.0),
            stt_bps=get("ICICI_STT_BPS", 10.0),
            exchange_bps=get("ICICI_EXCHANGE_BPS", 3.0),
            gst_pct=get("ICICI_GST_PCT", 18.0),
            sebi_turnover_bps=get("SEBI_TURNOVER_BPS", 0.0001),
            slippage_pct_per_side=get("SLIPPAGE_PCT_PER_SIDE", 0.05),
        )

    # ------------------------------------------------------------------ #

    def one_way_cost_rate(self, delivery: bool = True) -> float:
        """
        Fractional cost of trading notional value one way (buy OR sell),
        excluding flat fees. STT on delivery applies both sides on NSE;
        we conservatively charge it per side here.
        """
        brokerage_bps = self.equity_delivery_bps if delivery else 0.0
        base_bps = brokerage_bps + self.stt_bps + self.exchange_bps + self.sebi_turnover_bps
        # GST applies to brokerage + exchange charges, not STT.
        gst_bps = (brokerage_bps + self.exchange_bps) * (self.gst_pct / 100.0)
        return (base_bps + gst_bps) / 10_000.0

    def round_trip_cost_rate(self, delivery: bool = True) -> float:
        return 2.0 * self.one_way_cost_rate(delivery=delivery)

    def cost_for_trade(self, notional_inr: float, delivery: bool = True) -> float:
        """Total INR friction for one side of a trade of given notional."""
        variable = notional_inr * self.one_way_cost_rate(delivery=delivery)
        flat = 0.0 if delivery else self.equity_intraday_flat_inr
        return variable + flat

    def slippage_rate(self) -> float:
        return self.slippage_pct_per_side / 100.0
