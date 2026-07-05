"""
QuNtra UnifiedDataFetcher — single interface for all market data.

Routing table:
  NSE/BSE equity historical  -> jugaad-data
  NSE live quotes            -> jugaad-data live
  Options chain              -> Bharat-SM-Data (Derivatives.NSE)
  Fundamentals               -> Bharat-SM-Data (Fundamentals.MoneyControl)
  RBI series                 -> jugaad-data RBI
  Global indices ONLY        -> yfinance (never for Indian equity)
  Offline fallback           -> data/cache/*.csv (fetch_data_cache.py)

Every fetch can be validated with validate_data() -> DataQualityReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.universe import nse_symbol
from src.utils import cache_loader


@dataclass
class DataQualityReport:
    passed: bool
    n_rows: int
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"DataQuality[{status}] rows={self.n_rows} issues={self.issues}"


class UnifiedDataFetcher:
    """Routes each data need to the right source. See module docstring."""

    GLOBAL_TICKERS = {"^GSPC", "^IXIC", "^DJI", "^N225", "^HSI", "^FTSE", "^NSEI", "^NSEBANK"}

    def __init__(self, prefer_cache: bool = False):
        self.prefer_cache = prefer_cache

    # ------------------------------------------------------------------ #
    # Historical OHLC

    def get_historical_ohlc(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """Daily OHLCV, indexed by date, columns: open/high/low/close/volume."""
        start, end = _to_date(start), _to_date(end)

        if self.prefer_cache:
            try:
                df = cache_loader.load_ticker(ticker)
                return df.loc[str(start):str(end)]
            except FileNotFoundError:
                pass

        if exchange == "NSE":
            from jugaad_data.nse import stock_df
            raw = stock_df(symbol=nse_symbol(ticker), from_date=start,
                           to_date=end, series="EQ")
            df = raw.rename(columns={
                "DATE": "date", "OPEN": "open", "HIGH": "high",
                "LOW": "low", "CLOSE": "close", "VOLUME": "volume",
            })[["date", "open", "high", "low", "close", "volume"]]
            return df.set_index("date").sort_index()

        if exchange == "BSE":
            from jugaad_data.bse import bhavcopy_raw  # noqa: F401  (per-day bhavcopy)
            raise NotImplementedError(
                "BSE historical requires bhavcopy assembly — use NSE for the universe."
            )
        raise ValueError(f"Unknown exchange: {exchange}")

    # ------------------------------------------------------------------ #
    # Live quotes

    def get_live_quote(self, tickers: list[str]) -> pd.DataFrame:
        """Live quotes via jugaad-data NSELive. Columns: ticker, last_price, ..."""
        from jugaad_data.nse import NSELive
        live = NSELive()
        rows = []
        for t in tickers:
            q = live.stock_quote(nse_symbol(t))
            p = q.get("priceInfo", {})
            rows.append({
                "ticker": t,
                "last_price": p.get("lastPrice"),
                "close": p.get("lastPrice"),
                "open": p.get("open"),
                "day_high": p.get("intraDayHighLow", {}).get("max"),
                "day_low": p.get("intraDayHighLow", {}).get("min"),
                "prev_close": p.get("previousClose"),
                "change_pct": p.get("pChange"),
                "timestamp": q.get("metadata", {}).get("lastUpdateTime"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Options chain

    def get_options_chain(self, underlying: str = "NIFTY",
                          expiry: str | None = None) -> pd.DataFrame:
        """Options chain via Bharat-SM-Data Derivatives (NSE)."""
        from Derivatives import NSE
        nse = NSE()
        if expiry is None:
            expiry = nse.get_expiry_dates(underlying)[0]
        df = nse.get_option_chain(underlying, expiry)
        return df

    # ------------------------------------------------------------------ #
    # Fundamentals

    def get_fundamentals(self, ticker: str) -> dict:
        """Fundamental ratios via Bharat-SM-Data MoneyControl."""
        from Fundamentals import MoneyControl
        mc = MoneyControl()
        sym = nse_symbol(ticker)
        token, info = mc.get_ticker(sym)
        ratios = mc.get_complete_ratios_data(token, statement_type="standalone")
        return {"ticker": ticker, "info": info, "ratios": ratios}

    # ------------------------------------------------------------------ #
    # RBI data

    def get_rbi_data(self, series: str = "policy_repo_rate") -> pd.DataFrame:
        """Current RBI policy rates via jugaad-data."""
        from jugaad_data.rbi import RBI
        rbi = RBI()
        rates = rbi.current_rates()
        df = pd.DataFrame([rates])
        if series and series in df.columns:
            return df[[series]]
        return df

    # ------------------------------------------------------------------ #
    # Validation

    def validate_data(
        self,
        df: pd.DataFrame,
        max_gap_days: int = 5,
        max_stale_days: int = 7,
        price_col: str = "close",
    ) -> DataQualityReport:
        """Check for missing bars, NaNs, non-positive prices, stale data."""
        issues: list[str] = []
        if df is None or len(df) == 0:
            return DataQualityReport(False, 0, ["empty dataframe"])

        if price_col in df.columns:
            closes = df[price_col]
            n_nan = int(closes.isna().sum())
            if n_nan:
                issues.append(f"{n_nan} NaN {price_col} values")
            if (closes.dropna() <= 0).any():
                issues.append("non-positive prices found")
            # Bad ticks: >25% single-day move
            jumps = closes.pct_change().abs()
            n_jumps = int((jumps > 0.25).sum())
            if n_jumps:
                issues.append(f"{n_jumps} suspicious moves >25%/day")
        else:
            issues.append(f"missing column '{price_col}'")

        idx = df.index
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
            if not idx.is_monotonic_increasing:
                issues.append("index not sorted")
            gaps = idx.to_series().diff().dt.days.dropna()
            big = int((gaps > max_gap_days).sum())
            if big:
                issues.append(f"{big} gaps > {max_gap_days} days")
            age = (pd.Timestamp.now(tz=idx.tz) - idx[-1]).days
            if age > max_stale_days:
                issues.append(f"stale: last bar {age} days old")

        return DataQualityReport(len(issues) == 0, len(df), issues)


def _to_date(d: date | str) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(d, "%Y-%m-%d").date()
