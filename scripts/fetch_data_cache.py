#!/usr/bin/env python3
"""
Populate data/cache/ with historical OHLCV for the 25-ticker universe.

RUN THIS ON A MACHINE WITH OPEN INTERNET (your Mac):

    cd ~/Claude/Projects/quntra
    pip install yfinance jugaad-data pandas
    python3 scripts/fetch_data_cache.py --years 4

Primary source: jugaad-data (NSE official). Fallback: yfinance.
Output: data/cache/<TICKER>.csv with columns date,open,high,low,close,volume
plus data/cache/NIFTY50_BENCH.csv (^NSEI) for benchmark comparison.
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.universe import UNIVERSE, nse_symbol  # noqa: E402

CACHE = ROOT / "data" / "cache"
COLS = ["date", "open", "high", "low", "close", "volume"]


def fetch_jugaad(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        from jugaad_data.nse import stock_df
        df = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
        df = df.rename(columns={
            "DATE": "date", "OPEN": "open", "HIGH": "high",
            "LOW": "low", "CLOSE": "close", "VOLUME": "volume",
        })[COLS]
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        print(f"  jugaad-data failed for {symbol}: {e}")
        return None


def fetch_yf(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=str(start), end=str(end),
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={"index": "date"})
        return df[COLS]
    except Exception as e:  # noqa: BLE001
        print(f"  yfinance failed for {ticker}: {e}")
        return None


def upsert_to_db(ticker: str, df: pd.DataFrame) -> bool:
    """Best-effort mirror of the cache into the price_data table."""
    try:
        from src.db import PriceData, get_session, init_db
        init_db()
        rows = df.to_dict("records")
        with get_session() as s:
            for r in rows:
                s.merge(PriceData(
                    ticker=ticker,
                    date=pd.Timestamp(r["date"]).date(),
                    open=r["open"], high=r["high"], low=r["low"],
                    close=r["close"], volume=int(r["volume"]),
                    adjusted_close=r["close"],
                ))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  (DB mirror skipped: {e})")
        return False


def main() -> int:
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--start", type=str, default=None,
                    help="YYYY-MM-DD (overrides --years)")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between tickers (NSE rate-limit courtesy)")
    ap.add_argument("--no-db", action="store_true",
                    help="skip mirroring into the price_data table")
    args = ap.parse_args()

    end = date.today()
    start = (date.fromisoformat(args.start) if args.start
             else end - timedelta(days=365 * args.years))
    CACHE.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    first_date, last_date = None, None
    for ticker in UNIVERSE:
        out = CACHE / f"{ticker.replace('&', '_')}.csv"
        print(f"{ticker} ...")
        df = fetch_jugaad(nse_symbol(ticker), start, end)
        if df is None or len(df) < 200:
            df = fetch_yf(ticker, start, end)
        if df is None or len(df) < 200:
            failed.append(ticker)
            print(f"  FAILED: {ticker}")
            continue
        df.to_csv(out, index=False)
        if not args.no_db:
            upsert_to_db(ticker, df)
        ok += 1
        d0, d1 = str(df["date"].iloc[0])[:10], str(df["date"].iloc[-1])[:10]
        first_date = min(first_date or d0, d0)
        last_date = max(last_date or d1, d1)
        print(f"  saved {len(df)} rows -> {out.name}")
        time.sleep(args.sleep)

    bench = fetch_yf("^NSEI", start, end)
    if bench is not None:
        bench.to_csv(CACHE / "NIFTY50_BENCH.csv", index=False)
        print(f"benchmark ^NSEI: {len(bench)} rows")

    print(f"\nDATA FETCH COMPLETE: {ok}/{len(UNIVERSE)} tickers, "
          f"date range {first_date} to {last_date}")
    if failed:
        print("Failed:", failed)
    if ok < 23:
        print(f"GATE FAIL: need >= 23/25 tickers, got {ok}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
