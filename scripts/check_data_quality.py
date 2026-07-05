#!/usr/bin/env python3
"""
Data quality audit over data/cache/ — referenced by the completion loop.

For each cached ticker: row count, date range, NaN closes, gaps > 5
trading days, suspicious >25%/day moves, staleness.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.universe import UNIVERSE  # noqa: E402
from src.utils.cache_loader import load_ticker  # noqa: E402
from src.utils.data_fetcher import UnifiedDataFetcher  # noqa: E402


def main() -> int:
    fetcher = UnifiedDataFetcher()
    n_pass, n_fail, missing = 0, 0, []
    for t in UNIVERSE:
        try:
            df = load_ticker(t)
        except FileNotFoundError:
            missing.append(t)
            continue
        report = fetcher.validate_data(df, max_stale_days=7)
        tag = "PASS" if report.passed else "WARN"
        rng = f"{df.index[0].date()} -> {df.index[-1].date()}"
        print(f"{tag}  {t:16s} {report.n_rows:5d} rows  {rng}"
              + (f"  issues: {report.issues}" if report.issues else ""))
        n_pass += report.passed
        n_fail += (not report.passed)

    for t in missing:
        print(f"MISS  {t:16s} no cache file")
    print(f"\n{n_pass} clean / {n_fail} with issues / {len(missing)} missing "
          f"(of {len(UNIVERSE)})")
    return 0 if (n_pass >= 23 and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
