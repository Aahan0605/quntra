"""
Load the offline data cache (data/cache/*.csv) written by
scripts/fetch_data_cache.py. Lets backtests/training/validation run
without network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"


def cache_available(universe: list[str]) -> bool:
    return all((CACHE / f"{t.replace('&', '_')}.csv").exists() for t in universe)


def load_ticker(ticker: str) -> pd.DataFrame:
    """OHLCV frame for one ticker, indexed by date."""
    path = CACHE / f"{ticker.replace('&', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cache for {ticker}. Run scripts/fetch_data_cache.py "
            f"on a machine with internet access."
        )
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df


def load_close_panel(universe: list[str],
                     min_history: int = 1000) -> pd.DataFrame:
    """Wide DataFrame of close prices: index=date, columns=tickers.

    Tickers without a cache file are skipped (e.g. TATAMOTORS.NS after
    its 2025 demerger delisting). Tickers with fewer than `min_history`
    bars are ALSO dropped — otherwise a recent IPO (Swiggy, Paytm, Hyundai
    in the Nifty 200) would truncate the final dropna() window to a handful
    of days and make any backtest meaningless. ~1000 bars ≈ 4 years.
    """
    frames = {}
    for t in universe:
        try:
            s = load_ticker(t)["close"]
        except FileNotFoundError:
            continue
        if int(s.notna().sum()) >= min_history:
            frames[t] = s
    if not frames:
        raise FileNotFoundError(
            "No cached tickers with sufficient history. Run "
            "scripts/fetch_data_cache.py (or lower min_history).")
    panel = pd.DataFrame(frames).ffill().dropna()
    return panel


def load_benchmark() -> pd.Series | None:
    path = CACHE / "NIFTY50_BENCH.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df["close"]
