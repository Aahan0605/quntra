"""
QuNtra trading universe — exactly 25 unique, liquid NSE large-cap tickers.

This is the single source of truth for the ticker list. All modules
(feature pipeline, rebalancer, validation, paper trader) must import
UNIVERSE from here rather than hardcoding tickers.

Note: M&M.NS appears exactly once (historical bug: duplicated metadata).
scripts/verify_universe.py guards against regressions.
"""

UNIVERSE: list[str] = [
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

EXPECTED_COUNT = 25

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
