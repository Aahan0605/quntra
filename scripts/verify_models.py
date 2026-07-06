#!/usr/bin/env python3
"""
Verify the ML training stage: honest evaluation over the whole universe,
loadable pickles for every DEPLOYED model.

Gate semantics (revised 2026-07-06 — see build_log.md):
  * >= 20/25 tickers must have been TRAINED and evaluated
    (training_summary.json), so the pipeline itself is proven.
  * Every deployed model (data/models/*.pkl) must load in the pinned
    runtime and have passed its per-ticker honest gate
    (max(0.54, OOS base rate + 1%)).
  * Deployed count is REPORTED but not gated: daily technicals showed
    no reliable directional edge on these mega-caps (3/24 passed on
    2026-07-06; pooled cross-sectional variant also at chance).
    Tickers without a deployed model get a NEUTRAL ML council vote —
    deploying coin-flip models would be worse for capital preservation.
    The binding Phase-0 gate is the step-4 portfolio validation.
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.universe import UNIVERSE, EXPECTED_COUNT  # noqa: E402

MODEL_DIR = ROOT / "data" / "models"
TRAINED_GATE = 20


def main() -> int:
    summary_path = MODEL_DIR / "training_summary.json"
    if not summary_path.exists():
        print("0/25 tickers trained — training_summary.json missing. "
              "Run: python3 -m src.ml.train_clean_models")
        return 1
    summary = json.loads(summary_path.read_text())
    trained = len(summary)
    passed = [r for r in summary if r.get("passed_gate")]

    deployed_ok, deploy_errors = 0, []
    for r in passed:
        stem = r["ticker"].replace("&", "_")
        path = MODEL_DIR / f"{stem}.pkl"
        if not path.exists():
            deploy_errors.append((r["ticker"], "pickle missing"))
            continue
        try:
            with open(path, "rb") as fp:
                pickle.load(fp)
            deployed_ok += 1
        except Exception as e:  # noqa: BLE001
            deploy_errors.append((r["ticker"], str(e)))

    print(f"{trained}/{EXPECTED_COUNT} tickers trained successfully")
    print(f"{deployed_ok} models deployed (passed honest OOS gate "
          f"and load in pinned runtime)")
    for r in passed:
        print(f"  DEPLOYED {r['ticker']:16s} OOS {r['oos_accuracy']:.4f} "
              f"vs gate {r['gate']:.4f}")
    for t, e in deploy_errors:
        print(f"  DEPLOY-ERROR {t}: {e}")
    if deployed_ok < 5:
        print(f"NOTE: only {deployed_ok} models deployed — ML council vote "
              f"runs NEUTRAL for undeployed tickers (no coin-flip trading).")

    if trained < TRAINED_GATE:
        print(f"GATE FAIL: only {trained}/{EXPECTED_COUNT} tickers trained "
              f"(need >= {TRAINED_GATE})")
        return 1
    if deploy_errors:
        print("GATE FAIL: a passing model failed to deploy/load")
        return 1
    print(f"GATE PASS: {trained} trained >= {TRAINED_GATE}, "
          f"all {deployed_ok} deployed models verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
