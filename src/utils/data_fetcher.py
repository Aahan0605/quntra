"""
QuNtra UnifiedDataFetcher — single interface for all market data.

Routing table:
  NSE/BSE equity historical  -> jugaad-data, falling back to yfinance .NS
                                then local cache when NSE is unreachable
  NSE live quotes            -> jugaad-data live, per-ticker yfinance fallback
  Options chain              -> Bharat-SM-Data (Derivatives.NSE)
  Fundamentals               -> Bharat-SM-Data (Fundamentals.MoneyControl)
  RBI series                 -> jugaad-data RBI
  Global indices             -> yfinance (primary source for these only)
  Offline fallback           -> data/cache/*.csv (fetch_data_cache.py)

NSE's public API intermittently 503s non-browser clients (weekends
especially), so every NSE call must degrade rather than raise.

Every fetch can be validated with validate_data() -> DataQualityReport.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.utils.universe import nse_symbol
from src.utils import cache_loader

logger = logging.getLogger(__name__)

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
# Escape hatch: set QUNTRA_DISABLE_KITE=1 to force the free data sources
# (e.g. in tests or if a Kite plan lacks the market-data subscription).
DataFetcher_KITE_DISABLED = __import__("os").getenv(
    "QUNTRA_DISABLE_KITE", "") == "1"


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
            try:
                from jugaad_data.nse import stock_df
                raw = stock_df(symbol=nse_symbol(ticker), from_date=start,
                               to_date=end, series="EQ")
                df = raw.rename(columns={
                    "DATE": "date", "OPEN": "open", "HIGH": "high",
                    "LOW": "low", "CLOSE": "close", "VOLUME": "volume",
                })[["date", "open", "high", "low", "close", "volume"]]
                return df.set_index("date").sort_index()
            except Exception as exc:  # NSE 503 / cookie rejection / schema drift
                logger.warning("jugaad-data failed for %s (%s) — "
                               "falling back to yfinance", ticker, exc)
            df = self._yf_ohlc(ticker, start, end)
            if df is not None and len(df):
                return df
            try:  # last resort: whatever the local cache holds
                cached = cache_loader.load_ticker(ticker)
                return cached.loc[str(start):str(end)]
            except FileNotFoundError:
                raise RuntimeError(
                    f"All historical sources failed for {ticker} "
                    f"(jugaad-data, yfinance, cache)")

        if exchange == "BSE":
            from jugaad_data.bse import bhavcopy_raw  # noqa: F401  (per-day bhavcopy)
            raise NotImplementedError(
                "BSE historical requires bhavcopy assembly — use NSE for the universe."
            )
        raise ValueError(f"Unknown exchange: {exchange}")

    # ------------------------------------------------------------------ #
    # Live quotes

    def get_live_quote(self, tickers: list[str]) -> pd.DataFrame:
        """Live quotes, best source first: Kite (real-time) -> NSELive ->
        yfinance (~15 min delayed).

        Columns: ticker, last_price, close, open, day_high, day_low,
        prev_close, change_pct, timestamp, source. The 'source' column lets
        callers apply extra caution (wider slippage) on delayed data.

        Kite gives real-time NSE data but needs a valid daily access token
        (scripts/kite_login.py). This is DATA ONLY — ltp/quote, never an
        order call — so it's safe during paper trading. Without a token it
        silently falls through to the free sources.
        """
        # Preferred: Kite real-time (batch call for all tickers at once)
        kite_rows = self._kite_quotes(tickers)

        live = None
        try:
            from jugaad_data.nse import NSELive
            live = NSELive()
        except Exception as exc:
            logger.warning("NSELive unavailable (%s) — yfinance only", exc)

        rows = []
        for t in tickers:
            if t in kite_rows:
                rows.append(kite_rows[t])
                continue
            if live is not None:
                try:
                    q = live.stock_quote(nse_symbol(t))
                    p = q.get("priceInfo", {})
                    if p.get("lastPrice") is not None:
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
                            "source": "nse_live",
                        })
                        continue
                except Exception as exc:
                    logger.warning("NSELive quote failed for %s (%s) — "
                                   "trying yfinance", t, exc)
            row = self._yf_quote(t)
            if row is not None:
                rows.append(row)
            else:
                logger.error("No quote available for %s from any source", t)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Kite real-time quotes (data only — never places orders)

    _kite = None            # cached KiteConnect client
    _kite_tried = False     # don't retry connect() every call in one process

    def _get_kite(self):
        """Lazily build a KiteConnect client from secrets. Returns None if
        credentials are missing or the daily access token is invalid."""
        if self._kite is not None or DataFetcher_KITE_DISABLED:
            return self._kite
        if self.__class__._kite_tried:
            return self.__class__._kite
        self.__class__._kite_tried = True
        try:
            import os

            from dotenv import load_dotenv
            load_dotenv(_ROOT / "config" / "secrets.env")
            api_key = os.getenv("KITE_API_KEY")
            token = os.getenv("KITE_ACCESS_TOKEN")
            if not (api_key and token):
                return None
            from kiteconnect import KiteConnect
            k = KiteConnect(api_key=api_key)
            k.set_access_token(token)
            k.ltp(["NSE:INFY"])  # cheap auth probe; raises on a bad token
            self.__class__._kite = k
            logger.info("Kite real-time quotes enabled")
            return k
        except Exception as exc:  # noqa: BLE001 — degrade to free sources
            logger.info("Kite real-time unavailable (%s) — using NSELive/"
                        "yfinance. Run scripts/kite_login.py for a fresh "
                        "token.", str(exc)[:80])
            return None

    @staticmethod
    def _kite_symbol(ticker: str) -> str:
        """RELIANCE.NS -> NSE:RELIANCE (Kite's exchange:tradingsymbol)."""
        return f"NSE:{ticker[:-3] if ticker.endswith('.NS') else ticker}"

    def _kite_quotes(self, tickers: list[str]) -> dict[str, dict]:
        """Real-time quotes for as many tickers as Kite can serve. Empty
        dict when Kite is unavailable — caller falls through to free sources."""
        k = self._get_kite()
        if k is None:
            return {}
        symmap = {self._kite_symbol(t): t for t in tickers}
        try:
            data = k.quote(list(symmap.keys()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kite quote batch failed (%s) — falling back", exc)
            return {}
        out: dict[str, dict] = {}
        for sym, q in data.items():
            t = symmap.get(sym)
            if t is None or q.get("last_price") is None:
                continue
            ohlc = q.get("ohlc") or {}
            prev = ohlc.get("close")
            last = q.get("last_price")
            out[t] = {
                "ticker": t, "last_price": last, "close": last,
                "open": ohlc.get("open"), "day_high": ohlc.get("high"),
                "day_low": ohlc.get("low"), "prev_close": prev,
                "change_pct": ((last / prev - 1) * 100 if prev else None),
                "timestamp": q.get("timestamp"),
                "source": "kite_realtime",
            }
        return out

    # ------------------------------------------------------------------ #
    # yfinance fallbacks (delayed data — acceptable when NSE is down)

    @staticmethod
    def _yf_symbol(ticker: str) -> str:
        """Map to Yahoo notation: bare NSE symbols get the .NS suffix."""
        if ticker.startswith("^") or "." in ticker or "=" in ticker:
            return ticker
        return f"{ticker}.NS"

    def _yf_ohlc(self, ticker: str, start: date, end: date) -> pd.DataFrame | None:
        """Daily OHLCV via yfinance, normalized to the jugaad schema."""
        try:
            import yfinance as yf
            raw = yf.download(self._yf_symbol(ticker), start=str(start),
                              end=str(end + timedelta(days=1)),
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            return df.sort_index()
        except Exception as exc:
            logger.warning("yfinance OHLC failed for %s: %s", ticker, exc)
            return None

    def _yf_quote(self, ticker: str) -> dict | None:
        """Delayed quote via yfinance fast_info."""
        try:
            import yfinance as yf
            info = yf.Ticker(self._yf_symbol(ticker)).fast_info
            last = info.get("lastPrice") if hasattr(info, "get") else info.last_price
            prev = (info.get("previousClose") if hasattr(info, "get")
                    else info.previous_close)
            if last is None:
                return None
            return {
                "ticker": ticker,
                "last_price": float(last),
                "close": float(last),
                "open": info.get("open") if hasattr(info, "get") else info.open,
                "day_high": info.get("dayHigh") if hasattr(info, "get") else info.day_high,
                "day_low": info.get("dayLow") if hasattr(info, "get") else info.day_low,
                "prev_close": prev,
                "change_pct": (float(last) / float(prev) - 1) * 100 if prev else None,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source": "yfinance_delayed",
            }
        except Exception as exc:
            logger.warning("yfinance quote failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------ #
    # Options chain

    def get_options_chain(self, underlying: str = "NIFTY",
                          expiry: datetime | None = None) -> pd.DataFrame:
        """Options chain via Bharat-SM-Data Derivatives (NSE).

        Returns an empty DataFrame when NSE blocks the client (frequent
        403s outside market hours) — callers must handle len(df) == 0.
        """
        try:
            from Derivatives import NSE
            nse = NSE()
            is_index = underlying.upper() in {"NIFTY", "BANKNIFTY", "FINNIFTY",
                                              "MIDCPNIFTY", "NIFTYNXT50"}
            if expiry is None:
                expiry = nse.get_options_expiry(underlying, is_index=is_index)
            return nse.get_option_chain(underlying, is_index=is_index,
                                        expiry=expiry)
        except Exception as exc:
            logger.warning("Options chain unavailable for %s: %s",
                           underlying, exc)
            return pd.DataFrame()

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
