"""Benjamini-Hochberg must be right — it decides which models may trade."""

import json

import pytest

from src.ml.multiple_testing import (
    benjamini_hochberg,
    evaluate,
    load_trials,
    p_value,
)


def test_bh_matches_worked_example():
    """Textbook step-up case: m=5, q=0.05, ranks 1-2 pass."""
    pvals = [0.009, 0.021, 0.049, 0.30, 0.70]
    # thresholds i/m*q = 0.01, 0.02, 0.03, 0.04, 0.05
    # only rank 1 (0.009 <= 0.01) passes; largest passing rank = 1
    assert benjamini_hochberg(pvals, q=0.05) == [True, False, False, False, False]


def test_bh_is_step_up_not_step_down():
    """A later rank passing must rescue earlier, larger p-values below it."""
    # m=4, q=0.05 -> thresholds .0125 .025 .0375 .05
    # rank1 .02 fails, rank2 .024 passes -> ranks 1..2 both survive.
    pvals = [0.02, 0.024, 0.9, 0.9]
    assert benjamini_hochberg(pvals, q=0.05) == [True, True, False, False]


def test_bh_rejects_nothing_when_all_null():
    assert benjamini_hochberg([0.4, 0.6, 0.8, 0.99], q=0.10) == [False] * 4


def test_bh_rejects_everything_when_all_tiny():
    assert benjamini_hochberg([1e-9] * 6, q=0.10) == [True] * 6


def test_bh_is_stricter_than_uncorrected():
    """The whole point: surviving BH implies passing a naive 0.05 bar."""
    pvals = [0.001, 0.03, 0.04, 0.045, 0.049] + [0.5] * 95
    survives = benjamini_hochberg(pvals, q=0.05)
    naive = [p < 0.05 for p in pvals]
    assert sum(survives) < sum(naive)
    assert all(naive[i] for i, s in enumerate(survives) if s)


def test_p_value_coin_flip_model_is_not_significant():
    """Accuracy equal to the base rate is no evidence at all."""
    trial = {"n_oos": 250, "oos_accuracy": 0.52, "oos_base_rate": 0.52}
    assert p_value(trial) > 0.4


def test_p_value_strong_edge_is_significant():
    trial = {"n_oos": 250, "oos_accuracy": 0.70, "oos_base_rate": 0.52}
    assert p_value(trial) < 0.001


def test_p_value_shrinks_with_more_evidence():
    """Same edge, more out-of-sample days -> more convincing."""
    small = p_value({"n_oos": 60, "oos_accuracy": 0.60, "oos_base_rate": 0.52})
    large = p_value({"n_oos": 600, "oos_accuracy": 0.60, "oos_base_rate": 0.52})
    assert large < small


def _write(dirpath, name, **kw):
    meta = {"ticker": name, "n_oos": 250, "oos_base_rate": 0.52,
            "passed_gate": False}
    meta.update(kw)
    (dirpath / f"{name}.meta.json").write_text(json.dumps(meta))


def test_load_trials_reports_old_schema_instead_of_dropping_it(tmp_path):
    _write(tmp_path, "GOOD", oos_accuracy=0.60)
    (tmp_path / "OLD.meta.json").write_text(json.dumps({"ticker": "OLD"}))
    usable, skipped = load_trials(tmp_path)
    assert [t["ticker"] for t in usable] == ["GOOD"]
    assert skipped == ["OLD.meta.json"]


def test_load_trials_includes_rejected_subdir(tmp_path):
    """A family missing its losers would understate the correction."""
    rej = tmp_path / "rejected"
    rej.mkdir()
    _write(tmp_path, "WINNER", oos_accuracy=0.60, passed_gate=True)
    _write(rej, "LOSER", oos_accuracy=0.48)
    usable, _ = load_trials(tmp_path)
    assert {t["ticker"] for t in usable} == {"WINNER", "LOSER"}


def test_evaluate_kills_a_lone_winner_among_many_lotteries(tmp_path):
    """One 'winner' out of 200 coin flips is not an edge."""
    rej = tmp_path / "rejected"
    rej.mkdir()
    _write(tmp_path, "LUCKY", oos_accuracy=0.58, passed_gate=True)
    for i in range(199):
        _write(rej, f"T{i:03d}", oos_accuracy=0.52)
    res = evaluate(tmp_path, q=0.10)
    assert res["n_trials"] == 200
    assert res["n_passed_naive_gate"] == 1
    assert res["n_survives_fdr"] == 0, "a single lottery winner must not survive"


def test_evaluate_keeps_a_genuinely_strong_model(tmp_path):
    """The test must be capable of passing something, or it proves nothing."""
    rej = tmp_path / "rejected"
    rej.mkdir()
    _write(tmp_path, "REAL", oos_accuracy=0.72, passed_gate=True)
    for i in range(50):
        _write(rej, f"T{i:03d}", oos_accuracy=0.52)
    res = evaluate(tmp_path, q=0.10)
    assert [s["ticker"] for s in res["survivors"]] == ["REAL"]


def test_evaluate_refuses_an_empty_family(tmp_path):
    with pytest.raises(SystemExit):
        evaluate(tmp_path)
