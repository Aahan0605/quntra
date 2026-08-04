"""
QuNtra UnifiedDataFetcher — single interface for all market data.

Routing table:
  NSE/BSE equity historical  -> jugaad-data, falling back to yfinance .NS
                                then local cache when NSE is unreachable
  NSE live quotes            -> ICICI Breeze (real-time) -> yfinance
                                (~15 min delayed). Kite and NSELive were
                                dropped — neither ever served a real quote
                                on this account/machine (see get_live_quote).
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
# Escape hatch: set QUNTRA_DISABLE_BREEZE=1 to force yfinance-only
# (e.g. in tests, or if the Breeze subscription lapses).
DataFetcher_BREEZE_DISABLED = __import__("os").getenv(
    "QUNTRA_DISABLE_BREEZE", "") == "1"


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
        """Live quotes, best source first: ICICI Breeze (real-time) ->
        yfinance (~15 min delayed). Kite and NSELive were dropped: Kite's
        market-data subscription isn't on this account (PermissionException
        on every quote() call) and NSELive has never once worked on this
        machine (NSE tarpits it) — two sources that never actually served
        a quote, just extra failure paths to read through.

        Columns: ticker, last_price, close, open, day_high, day_low,
        prev_close, change_pct, timestamp, source. The 'source' column lets
        callers apply extra caution (wider slippage) on delayed data.
        """
        breeze_rows = self._breeze_quotes(tickers)

        rows = []
        for t in tickers:
            if t in breeze_rows:
                rows.append(breeze_rows[t])
                continue
            row = self._yf_quote(t)
            if row is not None:
                rows.append(row)
            else:
                logger.error("No quote available for %s from any source", t)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # ICICI Breeze real-time quotes (data only — never places orders)

    _breeze = None
    _breeze_tried = False
    _breeze_symbol_map: dict | None = None   # NSE symbol -> Breeze stock_code

    def _get_breeze(self):
        """Lazily authenticate a BreezeConnect client from secrets. None if
        credentials are missing or the daily session_token is invalid."""
        if self._breeze is not None or DataFetcher_BREEZE_DISABLED:
            return self._breeze
        if self.__class__._breeze_tried:
            return self.__class__._breeze
        self.__class__._breeze_tried = True
        try:
            import os

            from dotenv import load_dotenv
            load_dotenv(_ROOT / "config" / "secrets.env")
            key = os.getenv("ICICI_BREEZE_API_KEY")
            secret = os.getenv("ICICI_BREEZE_API_SECRET")
            # DB first, env second — a token refreshed via /breeze_token
            # lands in system_state, while the env var only ever holds the
            # deploy-time value, which is stale by the next morning.
            from src.integrations.breeze_session import stored_token
            token = stored_token()
            if not (key and secret and token):
                return None
            # breeze_connect downloads its security-master zip at import
            # time via bare urllib — the Python.org macOS build's bundled
            # cert.pem lacks the issuing CA, so first import 500s on
            # CERTIFICATE_VERIFY_FAILED without this. certifi (already a
            # transitive dep) is what `requests` uses successfully.
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            from breeze_connect import BreezeConnect
            b = BreezeConnect(api_key=key)
            b.generate_session(api_secret=secret, session_token=token)
            b.get_quotes(stock_code="RELIND", exchange_code="NSE",
                        product_type="cash")  # cheap auth probe
            self.__class__._breeze = b
            logger.info("ICICI Breeze real-time quotes enabled")
            return b
        except Exception as exc:  # noqa: BLE001 — degrade to Kite/free sources
            logger.info("Breeze real-time unavailable (%s) — trying Kite/"
                        "NSELive/yfinance. Session tokens expire daily.",
                        str(exc)[:100])
            return None

    def _load_breeze_symbol_map(self) -> dict:
        """NSE symbol -> Breeze stock_code, from ICICI's own security
        master (cached locally; the file changes rarely). Matching by
        plain string transforms (RELIANCE -> RELIND) isn't reliable —
        confirmed by inspection that the master file's last column is
        the actual NSE trading symbol, so join on that instead of guessing.
        """
        if self.__class__._breeze_symbol_map is not None:
            return self.__class__._breeze_symbol_map
        cache_path = _ROOT / "data" / "cache" / "icici_nse_scripmaster.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path, dtype=str)
            mapping = dict(zip(df["symbol"], df["stock_code"]))
            self.__class__._breeze_symbol_map = mapping
            return mapping
        import csv
        import io
        from urllib.request import urlopen
        from zipfile import ZipFile

        mapping: dict[str, str] = {}
        try:
            resp = urlopen(
                "https://directlink.icicidirect.com/MotherAppMaster/"
                "SecurityMaster.zip", timeout=30)
            with ZipFile(io.BytesIO(resp.read())) as z, \
                 z.open("NSEScripMaster.txt") as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                next(reader)  # header
                for row in reader:
                    if len(row) < 61:
                        continue
                    symbol, stock_code = row[-1].strip(), row[1].strip()
                    if symbol:
                        mapping[symbol] = stock_code
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"symbol": list(mapping), "stock_code": list(mapping.values())}
                        ).to_csv(cache_path, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not load Breeze symbol map (%s)", exc)
        self.__class__._breeze_symbol_map = mapping
        return mapping

    def _breeze_quotes(self, tickers: list[str]) -> dict[str, dict]:
        """Real-time quotes via ICICI Breeze. Empty dict when Breeze is
        unavailable or a ticker has no Breeze stock_code mapping."""
        b = self._get_breeze()
        if b is None:
            return {}
        symmap = self._load_breeze_symbol_map()
        out: dict[str, dict] = {}
        for t in tickers:
            bare = t[:-3] if t.endswith(".NS") else t
            code = symmap.get(bare)
            if not code:
                continue
            try:
                resp = b.get_quotes(stock_code=code, exchange_code="NSE",
                                    product_type="cash")
                rows = resp.get("Success") or []
                q = next((r for r in rows if r.get("exchange_code") == "NSE"),
                        rows[0] if rows else None)
                if not q or q.get("ltp") is None:
                    continue
                last = float(q["ltp"])
                prev = q.get("previous_close")
                out[t] = {
                    "ticker": t, "last_price": last, "close": last,
                    "open": q.get("open"), "day_high": q.get("high"),
                    "day_low": q.get("low"), "prev_close": prev,
                    "change_pct": (q.get("ltp_percent_change")
                                  if q.get("ltp_percent_change") is not None
                                  else ((last / prev - 1) * 100 if prev else None)),
                    "timestamp": q.get("ltt"),
                    "source": "breeze_realtime",
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Breeze quote failed for %s (%s)", t, exc)
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
