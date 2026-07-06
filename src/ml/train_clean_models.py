"""
Train the 25 per-ticker XGBoost direction models — leakage-safe.

* Features computed only from information available at time t
  (returns, vol, RSI, MACD, momentum, volume z-score, benchmark-relative
  strength — all lagged).
* Label: 5-trading-day forward close-to-close direction (1 = up).
  Matches the weekly rebalancing horizon; next-day direction on large-cap
  NSE names proved unlearnable (~50% OOS across 24 tickers, 2026-07-06).
* Chronological split: last 20% of rows are out-of-sample (OOS), with a
  HORIZON-day purge gap at the boundary so overlapping labels can't leak.
* Deployment gate: OOS accuracy >= max(0.54, OOS base rate + 0.01).
  The base-rate floor stops upward drift from passing as skill — a model
  that can't beat always-up has no edge worth deploying.

Run inside the pinned environment (requirements-pinned.txt):

    python3 -m src.ml.train_clean_models
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.universe import UNIVERSE  # noqa: E402
from src.utils.cache_loader import load_ticker  # noqa: E402

MODEL_DIR = ROOT / "data" / "models"
REJECT_DIR = MODEL_DIR / "rejected"
OOS_GATE = 0.54
OOS_FRACTION = 0.20
HORIZON = 5                  # forward label horizon, trading days
BASE_RATE_MARGIN = 0.01      # must beat always-up by at least this


# --------------------------------------------------------------------- #
# Features (leakage-safe: everything is a function of data up to t)

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame,
                   bench: pd.Series | None = None) -> pd.DataFrame:
    close, vol = df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)
    f["ret_1d"] = close.pct_change()
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)
    f["vol_10d"] = f["ret_1d"].rolling(10).std()
    f["vol_20d"] = f["ret_1d"].rolling(20).std()
    f["rsi_14"] = _rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    f["macd"] = (ema12 - ema26) / close
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
    f["mom_10d"] = close / close.shift(10) - 1
    f["px_vs_sma20"] = close / close.rolling(20).mean() - 1
    f["px_vs_sma50"] = close / close.rolling(50).mean() - 1
    f["volume_z"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    f["hl_range"] = (df["high"] - df["low"]) / close
    f["dist_52w_high"] = close / close.rolling(252, min_periods=60).max() - 1
    if bench is not None:
        b = bench.reindex(f.index).ffill()
        f["rel_nifty_5d"] = close.pct_change(5) - b.pct_change(5)
        f["rel_nifty_20d"] = close.pct_change(20) - b.pct_change(20)
    return f


def build_dataset(df: pd.DataFrame,
                  bench: pd.Series | None = None
                  ) -> tuple[pd.DataFrame, pd.Series]:
    X = build_features(df, bench)
    # HORIZON-day forward direction — matches the weekly holding period
    y = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
    data = X.join(y.rename("label")).iloc[:-HORIZON].dropna()
    return data.drop(columns="label"), data["label"]


# --------------------------------------------------------------------- #

def _load_benchmark() -> pd.Series | None:
    try:
        from src.utils.cache_loader import load_benchmark
        return load_benchmark()
    except Exception:  # noqa: BLE001
        return None


def train_one(ticker: str, bench: pd.Series | None = None) -> dict:
    from xgboost import XGBClassifier

    df = load_ticker(ticker)
    X, y = build_dataset(df, bench)
    split = int(len(X) * (1 - OOS_FRACTION))
    # Purge HORIZON rows at the boundary: a t in train with t+5 label
    # overlapping the OOS window would leak future information.
    X_tr, X_te = X.iloc[:split - HORIZON], X.iloc[split:]
    y_tr, y_te = y.iloc[:split - HORIZON], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5,
        eval_metric="logloss", n_jobs=2, random_state=42,
    )
    model.fit(X_tr, y_tr)
    oos_acc = float((model.predict(X_te) == y_te).mean())
    base_rate = float(max(y_te.mean(), 1 - y_te.mean()))
    effective_gate = max(OOS_GATE, base_rate + BASE_RATE_MARGIN)
    passed = oos_acc >= effective_gate

    out_dir = MODEL_DIR if passed else REJECT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ticker.replace("&", "_")
    with open(out_dir / f"{stem}.pkl", "wb") as fp:
        pickle.dump(model, fp)

    meta = {
        "ticker": ticker,
        "oos_accuracy": round(oos_acc, 4),
        "oos_base_rate": round(base_rate, 4),
        "gate": round(effective_gate, 4),
        "label_horizon_days": HORIZON,
        "passed_gate": passed,
        "n_train": len(X_tr),
        "n_oos": len(X_te),
        "features": list(X.columns),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_versions(),
    }
    with open(out_dir / f"{stem}.meta.json", "w") as fp:
        json.dump(meta, fp, indent=2)
    return meta


def _runtime_versions() -> dict:
    import sklearn
    import xgboost
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def main() -> int:
    import argparse
    global MODEL_DIR, REJECT_DIR, OOS_GATE

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="cache dir (default data/cache)")
    ap.add_argument("--output-dir", default=None,
                    help="model output dir (default data/models)")
    ap.add_argument("--oos-threshold", type=float, default=OOS_GATE)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.output_dir:
        MODEL_DIR = Path(args.output_dir)
        REJECT_DIR = MODEL_DIR / "rejected"
    OOS_GATE = args.oos_threshold
    if args.data_dir:
        from src.utils import cache_loader
        cache_loader.CACHE = Path(args.data_dir)

    bench = _load_benchmark()
    results = []
    for t in UNIVERSE:
        try:
            meta = train_one(t, bench)
            tag = "PASS" if meta["passed_gate"] else "REJECT"
            print(f"{tag}  {t:16s} OOS acc {meta['oos_accuracy']:.4f} "
                  f"(base {meta['oos_base_rate']:.4f}, "
                  f"gate {meta['gate']:.4f})")
            results.append(meta)
        except FileNotFoundError as e:
            print(f"SKIP  {t:16s} {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t:16s} {type(e).__name__}: {e}")

    n_pass = sum(r["passed_gate"] for r in results)
    print(f"\n{len(results)}/{len(UNIVERSE)} trained, {n_pass} passed the "
          f"{OOS_GATE:.0%} OOS gate")
    summary = MODEL_DIR / "training_summary.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2))
    return 0 if len(results) == len(UNIVERSE) else 1


if __name__ == "__main__":
    sys.exit(main())
