"""Was that model skillful, or did we just run 194 lotteries?

The per-ticker gate (OOS accuracy >= max(0.54, base rate + 1%)) asks a
single-model question. Training 194 tickers asks it 194 times, and roughly
10 tickers clear a 5%-level bar on noise alone. The gate cannot tell those
apart from real edge — the survivors were never corrected for the number
of attempts that produced them.

This applies Benjamini-Hochberg to the whole family of trials:

    p_i  = P(at least this many correct | model is a coin weighted to the
           base rate)                        [one-sided exact binomial]
    BH   = reject H0_i where p_(i) <= (i/m) * q

BH controls the *false discovery rate* — the expected share of surviving
models that are noise — rather than the probability of any single false
positive. That is the right target here: some false discoveries are
tolerable, a survivor list that is mostly noise is not.

Reads the meta.json files the trainer already writes, so it needs no
retraining and no change to the training loop. Rejected models must be
included: the correction is only valid over every trial that was run.

    ./venv/bin/python -m src.ml.multiple_testing data/models_nifty200
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import binomtest

DEFAULT_FDR = 0.10


REQUIRED = ("n_oos", "oos_accuracy", "oos_base_rate")


def load_trials(models_dir: Path) -> tuple[list[dict], list[str]]:
    """Every trial, passed and rejected — a partial family invalidates BH.

    Returns (usable, skipped). Older meta.json files predate `oos_base_rate`
    and cannot be scored; they are reported rather than dropped quietly,
    because a silently shrunken family understates the correction.
    """
    usable, skipped = [], []
    for p in sorted(models_dir.rglob("*.meta.json")):
        t = json.loads(p.read_text())
        (usable if all(k in t for k in REQUIRED) else skipped).append(
            t if all(k in t for k in REQUIRED) else p.name)
    return usable, skipped


def p_value(trial: dict) -> float:
    """One-sided exact binomial test against that ticker's own base rate."""
    n = int(trial["n_oos"])
    correct = round(float(trial["oos_accuracy"]) * n)
    base = min(max(float(trial["oos_base_rate"]), 1e-9), 1 - 1e-9)
    return binomtest(correct, n, base, alternative="greater").pvalue


def benjamini_hochberg(pvals: list[float], q: float = DEFAULT_FDR) -> list[bool]:
    """True where H0 is rejected, controlling FDR at q. Step-up procedure."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    survives = [False] * m
    cutoff_rank = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank / m * q:
            cutoff_rank = rank
    # Everything up to the largest passing rank survives, including any
    # larger p-values below it — that step-up is what makes BH valid.
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff_rank:
            survives[idx] = True
    return survives


def evaluate(models_dir: Path, q: float = DEFAULT_FDR) -> dict:
    trials, skipped = load_trials(models_dir)
    if not trials:
        raise SystemExit(f"no scoreable *.meta.json under {models_dir}")
    pvals = [p_value(t) for t in trials]
    survives = benjamini_hochberg(pvals, q)
    for t, p, s in zip(trials, pvals, survives):
        t["p_value"] = p
        t["survives_fdr"] = s
    naive = [t for t in trials if t.get("passed_gate")]
    return {
        "models_dir": str(models_dir),
        "n_trials": len(trials),
        "n_skipped_old_schema": len(skipped),
        "skipped": skipped,
        "fdr_q": q,
        "n_passed_naive_gate": len(naive),
        "expected_false_positives_at_naive_gate": round(0.05 * len(trials), 1),
        "n_survives_fdr": sum(survives),
        "survivors": sorted(
            ({"ticker": t["ticker"], "oos_accuracy": t["oos_accuracy"],
              "base_rate": t["oos_base_rate"], "n_oos": t["n_oos"],
              "p_value": round(t["p_value"], 5)}
             for t in trials if t["survives_fdr"]),
            key=lambda d: d["p_value"]),
    }


def main(argv: list[str]) -> int:
    models_dir = Path(argv[1]) if len(argv) > 1 else Path("data/models_nifty200")
    q = float(argv[2]) if len(argv) > 2 else DEFAULT_FDR
    res = evaluate(models_dir, q)

    print(f"\nMultiple-testing audit — {res['models_dir']}")
    print("=" * 62)
    print(f"Trials run                        {res['n_trials']}")
    if res["n_skipped_old_schema"]:
        print(f"  (skipped, pre-schema meta)      {res['n_skipped_old_schema']}")
    print(f"Passed the naive per-model gate   {res['n_passed_naive_gate']}")
    print(f"  ...expected from noise alone    "
          f"~{res['expected_false_positives_at_naive_gate']}")
    print(f"Survive BH at FDR q={res['fdr_q']}          "
          f"{res['n_survives_fdr']}")
    print("-" * 62)
    for s in res["survivors"]:
        print(f"  {s['ticker']:<18} acc {s['oos_accuracy']:.4f} "
              f"(base {s['base_rate']:.4f}, n={s['n_oos']}, p={s['p_value']:.5f})")
    if not res["survivors"]:
        print("  none — no ticker's edge is distinguishable from luck.")
    print()
    out = models_dir / "multiple_testing.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"written -> {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
