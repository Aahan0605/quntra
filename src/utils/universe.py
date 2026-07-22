"""
QuNtra trading universe — exactly 25 unique, liquid NSE large-cap tickers.

This is the single source of truth for the ticker list. All modules
(feature pipeline, rebalancer, validation, paper trader) must import
UNIVERSE from here rather than hardcoding tickers.

Note: M&M.NS appears exactly once (historical bug: duplicated metadata).
scripts/verify_universe.py guards against regressions.
"""

import os as _os

# The original validated 25-name large-cap set (the default).
_DEFAULT_UNIVERSE: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "M&M.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "TATASTEEL.NS",
    "TATAMOTORS.NS",
    "ASIANPAINT.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
]

# QUNTRA_UNIVERSE=nifty200 switches the ENTIRE system (fetch, train,
# validate, council scoring, sector map) to the Nifty 200 universe. The
# env var must be set before the process starts (put it in .env). Default
# stays the safe 25-name set.
UNIVERSE_SET = _os.getenv("QUNTRA_UNIVERSE", "default").lower()
if UNIVERSE_SET == "nifty200":
    from src.utils.universe_nifty200 import NIFTY200 as _N200
    UNIVERSE: list[str] = list(_N200)
elif UNIVERSE_SET == "nifty100":
    from src.utils.universe_nifty100 import NIFTY100 as _N100
    UNIVERSE = list(_N100)
else:
    UNIVERSE = _DEFAULT_UNIVERSE

EXPECTED_COUNT = len(UNIVERSE)

# Alias used by the completion-loop prompts
VALIDATED_TICKERS = UNIVERSE


def validate_universe(universe: list[str] | None = None) -> tuple[bool, str]:
    """Return (ok, message). Checks count, uniqueness, and .NS suffix."""
    u = universe if universe is not None else UNIVERSE
    if len(u) != EXPECTED_COUNT:
        return False, f"Expected {EXPECTED_COUNT} tickers, found {len(u)}"
    dupes = {t for t in u if u.count(t) > 1}
    if dupes:
        return False, f"Duplicate tickers: {sorted(dupes)}"
    bad = [t for t in u if not t.endswith(".NS")]
    if bad:
        return False, f"Non-NSE tickers: {bad}"
    return True, f"{EXPECTED_COUNT} unique tickers confirmed"


def nse_symbol(ticker: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE' (jugaad-data / NSE native symbol)."""
    return ticker.removesuffix(".NS")
