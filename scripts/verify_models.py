#!/usr/bin/env python3
"""
Verify all 25 per-ticker model pickles load in the pinned runtime.

Looks in data/models/ (canonical) and models/xgboost/ (legacy) for
<TICKER>.pkl files. Passes only when every ticker in the universe has
a loadable model: prints "25/25 models loaded successfully".
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.universe import UNIVERSE, EXPECTED_COUNT  # noqa: E402

MODEL_DIRS = [ROOT / "data" / "models", ROOT / "models" / "xgboost"]


def find_model_file(ticker: str) -> Path | None:
    stem = ticker.replace(".NS", "").replace("&", "_")
    for d in MODEL_DIRS:
        for cand in (d / f"{ticker}.pkl", d / f"{stem}.pkl", d / f"{stem}_NS.pkl"):
            if cand.exists():
                return cand
    return None


def main() -> int:
    loaded, failed, missing = 0, [], []
    for ticker in UNIVERSE:
        path = find_model_file(ticker)
        if path is None:
            missing.append(ticker)
            continue
        try:
            with open(path, "rb") as fp:
                pickle.load(fp)
            loaded += 1
        except Exception as e:  # noqa: BLE001
            failed.append((ticker, path.name, str(e)))

    print(f"{loaded}/{EXPECTED_COUNT} models loaded successfully")
    for t in missing:
        print(f"MISSING: {t} — no pickle found in {[str(d) for d in MODEL_DIRS]}")
    for t, f, e in failed:
        print(f"FAILED: {t} ({f}) — {e}")
    return 0 if loaded == EXPECTED_COUNT else 1


if __name__ == "__main__":
    sys.exit(main())
