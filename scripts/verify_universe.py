#!/usr/bin/env python3
"""Verify the QuNtra universe has exactly 25 unique NSE tickers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.universe import UNIVERSE, validate_universe  # noqa: E402


def main() -> int:
    for i, t in enumerate(UNIVERSE, 1):
        print(f"{i:2d}. {t}")
    ok, msg = validate_universe()
    print(msg if ok else f"FAIL: {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
