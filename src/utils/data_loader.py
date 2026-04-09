"""
Data loading and preprocessing utilities for Quantra.

Handles historical and live market data ingestion from yfinance,
data cleaning, caching, and return matrix computation for Nifty 50 assets.
"""

import warnings
import pandas as pd
import yfinance as yf
from typing import Optional


def fetch_nifty50_prices(
    tickers: list[str],
    start: str = "2021-01-01",
    end: str = "2024-01-01",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for the given Nifty 50 tickers.
    
    Parameters
    ----------
    tickers    : list of ticker symbols (e.g., ["RELIANCE.NS", "TCS.NS"])
    start, end : date range strings (YYYY-MM-DD)
    cache_path : optional path to save/load CSV cache
    
    Returns
    -------
    pd.DataFrame with datetime index and ticker columns
    """
    if cache_path:
        try:
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Clean dividend and split adjusted 'Close' price
    prices = raw["Close"].ffill().dropna()

    if cache_path:
        prices.to_csv(cache_path)

    return prices


def get_returns(prices: pd.DataFrame, method: str = "arithmetic") -> pd.DataFrame:
    """
    Compute assets' returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame of price series
    method : 'arithmetic' (percentage change) or 'log'
    
    Returns
    -------
    pd.DataFrame of returns
    """
    if method == "log":
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()
