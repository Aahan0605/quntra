#!/usr/bin/env python3
"""
Research sweep: improved features × 5 model candidates × 24 tickers.

Everything lands in data/models_research/ — production models in
data/models/ are FROZEN for the 40-day paper gate and never touched.

    python3 scripts/research_model_sweep.py

Evaluation matches production conventions: chronological 80/20 split
with a 5-day purge at the boundary, and BOTH gates reported —
flat 0.54 and the honest gate max(0.54, OOS base rate + 1%).

Multiple-comparison caveat: 5 models × 24 tickers = 120 trials; at a 5%
false-positive rate ~6 spurious "passes" are EXPECTED. Nothing here is
promoted automatically — human review after the paper gate, always.
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score  # noqa: E402

from src.ml.research.improved_features import ImprovedFeaturePipeline  # noqa: E402
from src.ml.research.model_candidates import get_model_candidates  # noqa: E402
from src.utils.cache_loader import load_benchmark, load_ticker  # noqa: E402
from src.utils.universe import UNIVERSE  # noqa: E402

RESEARCH_DIR = ROOT / "data" / "models_research"
PROD_DIR = ROOT / "data" / "models"
GATE = 0.54
PURGE = 5
OOS_FRACTION = 0.20


def main() -> int:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    prod_before = sorted(p.name for p in PROD_DIR.glob("*"))

    try:
        bench = load_benchmark()
    except Exception:  # noqa: BLE001
        bench = None
    pipeline = ImprovedFeaturePipeline()

    all_results: dict = {}
    flat_pass, honest_pass = 0, 0

    for ticker in UNIVERSE:
        try:
            prices = load_ticker(ticker)
        except FileNotFoundError:
            print(f"SKIP  {ticker:16s} no cache")
            continue
        try:
            X, y = pipeline.build(prices, bench)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {ticker:16s} features: {e}")
            continue

        split = int(len(X) * (1 - OOS_FRACTION))
        X_tr, y_tr = X.iloc[:split - PURGE], y.iloc[:split - PURGE]
        X_te, y_te = X.iloc[split:], y.iloc[split:]
        if len(X_tr) < 200 or len(X_te) < 50:
            print(f"SKIP  {ticker:16s} insufficient rows")
            continue
        base_rate = float(max(y_te.mean(), 1 - y_te.mean()))
        honest_gate = max(GATE, base_rate + 0.01)

        ticker_results, best_acc, best_name, best_model = {}, 0.0, None, None
        for name, model in get_model_candidates().items():
            try:
                model.fit(X_tr, y_tr)
                acc = float(accuracy_score(y_te, model.predict(X_te)))
                ticker_results[name] = round(acc, 4)
                if acc > best_acc:
                    best_acc, best_name, best_model = acc, name, model
            except Exception as e:  # noqa: BLE001
                ticker_results[name] = f"error: {e}"

        flat = best_acc >= GATE
        honest = best_acc >= honest_gate
        flat_pass += flat
        honest_pass += honest
        tag = "HONEST-PASS" if honest else ("flat-pass" if flat else "fail")
        print(f"{ticker:16s} best {best_acc:.4f} ({best_name}) "
              f"base {base_rate:.4f} -> {tag}")

        all_results[ticker] = {
            "models": ticker_results, "best": best_name,
            "best_acc": round(best_acc, 4),
            "base_rate": round(base_rate, 4),
            "honest_gate": round(honest_gate, 4),
            "flat_pass": flat, "honest_pass": honest,
        }
        if flat and best_model is not None:
            stem = ticker.replace("&", "_")
            with open(RESEARCH_DIR / f"{stem}.pkl", "wb") as fp:
                pickle.dump(best_model, fp)
            (RESEARCH_DIR / f"{stem}.meta.json").write_text(
                json.dumps({**all_results[ticker], "ticker": ticker,
                            "environment": "RESEARCH",
                            "features": list(X.columns)}, indent=2))

    (RESEARCH_DIR / "sweep_results.json").write_text(
        json.dumps(all_results, indent=2))

    prod_after = sorted(p.name for p in PROD_DIR.glob("*"))
    frozen_ok = prod_before == prod_after
    print("\n" + "=" * 60)
    print(f"Research sweep complete over {len(all_results)} tickers")
    print(f"Flat 0.54 gate:   {flat_pass}/{len(all_results)} "
          f"(production baseline: 3 on honest gate)")
    print(f"Honest gate:      {honest_pass}/{len(all_results)}")
    print(f"Expected false positives across 120 trials: ~6 — "
          f"treat marginal passes as noise")
    print(f"Production data/models/ unchanged: "
          f"{'YES' if frozen_ok else 'NO — INVESTIGATE'}")
    print("Promotion research -> production: manual review, "
          "only after the 40-day gate")
    return 0 if frozen_ok else 1


if __name__ == "__main__":
    sys.exit(main())
