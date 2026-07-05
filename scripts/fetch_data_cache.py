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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=4)
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=365 * args.years)
    CACHE.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
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
        ok += 1
        print(f"  saved {len(df)} rows -> {out.name}")

    bench = fetch_yf("^NSEI", start, end)
    if bench is not None:
        bench.to_csv(CACHE / "NIFTY50_BENCH.csv", index=False)
        print(f"benchmark ^NSEI: {len(bench)} rows")

    print(f"\n{ok}/{len(UNIVERSE)} tickers cached. Failed: {failed or 'none'}")
    return 0 if ok == len(UNIVERSE) else 1


if __name__ == "__main__":
    sys.exit(main())
