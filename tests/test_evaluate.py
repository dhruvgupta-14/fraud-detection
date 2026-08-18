"""Evaluation guards (architecture.md 9).

Two classes of failure are covered, both of which produce a *plausible number* rather than
an error:

* a metric computed at the wrong operating point (the threshold leak, 8.4);
* a mechanism check that manufactures a verdict out of small-count noise (9.3).

The second is the reason ``asymmetry_check`` and ``summarise`` are tested at all: an
evaluation helper that always returns a confident answer is worse than none, because it
launders noise into a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.evaluate.metrics import (
    alerts_per_day,
    auprc,
    bootstrap_auprc_ci,
    evaluate_split,
    threshold_at_recall,
)
from aml.evaluate.typology import (
    asymmetry_check,
    compare_arms_at_budget,
    threshold_for_budget,
    typology_recall,
    wilson_interval,
)
from aml.evaluate.walkforward import make_blocks, summarise


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_auprc_of_a_perfect_ranker_is_one():
    y = np.array([0, 0, 1, 1])
    assert auprc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auprc_of_a_random_ranker_approaches_prevalence():
    """The floor every reported AUPRC must be read against."""
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.01).astype(int)
    assert auprc(y, rng.random(20000)) == pytest.approx(y.mean(), abs=0.01)


def test_threshold_at_recall_reaches_the_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    threshold = threshold_at_recall(y, scores, 0.75)
    assert ((scores >= threshold) & (y == 1)).sum() / y.sum() >= 0.75


def test_threshold_at_recall_picks_the_strictest_qualifying_point():
    """Among thresholds meeting the recall target, the fewest-alerts one is chosen -- that
    is the operating point a compliance team would actually take."""
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    threshold = threshold_at_recall(y, scores, 1.0)
    assert (scores >= threshold).sum() == 2  # not 3, 4, 5 or 6


def test_alerts_per_day_divides_by_the_window():
    scores = np.array([0.9, 0.8, 0.1])
    assert alerts_per_day(scores, 0.5, n_days=2) == pytest.approx(1.0)


def test_alerts_per_day_rejects_a_zero_window():
    with pytest.raises(ValueError, match="n_days"):
        alerts_per_day(np.array([1.0]), 0.5, n_days=0)


def test_evaluate_split_reports_lift_against_prevalence():
    """AUPRC alone is unreadable at 0.1 % prevalence; the multiple over random is the
    honest framing and must be present."""
    y = np.array([0] * 99 + [1])
    scores = np.linspace(0, 1, 100)
    out = evaluate_split(y, scores, threshold=0.5, n_days=1)
    assert out["prevalence"] == pytest.approx(0.01)
    assert out["auprc_lift_over_random"] == pytest.approx(out["auprc"] / 0.01)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    scores = rng.random(2000) + y * 0.5
    lo, hi = bootstrap_auprc_ci(y, scores, iterations=80, seed=7)
    assert lo <= auprc(y, scores) <= hi


# --------------------------------------------------------------------------------------
# Typology
# --------------------------------------------------------------------------------------


def test_wilson_interval_is_wide_at_small_n():
    """The whole point: 9/10 must not read as confidently as 900/1000."""
    small = wilson_interval(9, 10)
    large = wilson_interval(900, 1000)
    assert (small[1] - small[0]) > (large[1] - large[0]) * 3


def test_wilson_interval_does_not_collapse_at_the_boundary():
    """A family where every positive was caught still carries uncertainty."""
    lo, hi = wilson_interval(10, 10)
    assert lo < 1.0 and hi <= 1.0


def test_threshold_for_budget_flags_exactly_that_many():
    scores = np.arange(100, dtype=float)
    threshold = threshold_for_budget(scores, 10)
    assert (scores >= threshold).sum() == 10


def test_threshold_for_budget_is_capped_by_available_rows():
    scores = np.arange(5, dtype=float)
    assert (scores >= threshold_for_budget(scores, 999)).sum() == 5


def _toy_arms():
    """Two arms over 6 positives in two typologies, with known detections."""
    predictions = {}
    tx = np.arange(12)
    labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # arm A ranks CYCLE positives low; arm B ranks them high.
    predictions["A"] = pd.DataFrame(
        {"tx_id": tx, "split": "test", "label": labels,
         "score": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]}
    )
    predictions["B"] = pd.DataFrame(
        {"tx_id": tx, "split": "test", "label": labels,
         "score": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0]}
    )
    typology = pd.DataFrame(
        {"tx_id": [0, 1, 2, 3, 4, 5],
         "attempt_id": [0, 0, 0, 1, 1, 1],
         "typology": ["RANDOM"] * 3 + ["CYCLE"] * 3}
    )
    return predictions, typology


def test_typology_recall_counts_detections_per_family():
    predictions, typology = _toy_arms()
    table = typology_recall(predictions["A"], typology, threshold=0.5)
    by_name = table.set_index("typology")
    assert by_name.loc["RANDOM", "n_detected"] == 3
    assert by_name.loc["CYCLE", "n_detected"] == 0


def test_unannotated_positives_get_their_own_bucket():
    """Only 61 % of test positives carry an annotation; the rest must be visible as a
    bucket rather than silently dropped from the denominator."""
    predictions, typology = _toy_arms()
    trimmed = typology[typology["tx_id"] < 3]  # leave 3 positives unannotated
    table = typology_recall(predictions["A"], trimmed, threshold=0.5)
    assert "UNANNOTATED" in set(table["typology"])
    assert int(table["n_positives"].sum()) == 6


def test_comparison_at_a_shared_budget_gives_every_arm_the_same_alert_count():
    predictions, typology = _toy_arms()
    comparison = compare_arms_at_budget(predictions, typology, budget=6)
    for arm, group in comparison.groupby("arm"):
        rows = predictions[arm]
        threshold = group["threshold"].iloc[0]
        assert (rows["score"] >= threshold).sum() == 6


def test_asymmetry_check_refuses_a_verdict_when_intervals_overlap():
    """The regression this exists for.

    An earlier version averaged per-family recall rates and declared a leak alarm off a
    two-positive change on the smallest family. Pooling counts and requiring disjoint
    intervals is what stops small-N noise becoming a finding.
    """
    predictions, typology = _toy_arms()
    comparison = compare_arms_at_budget(predictions, typology, budget=6)
    result = asymmetry_check(comparison, "A", "B")
    assert result["intervals_disjoint"] is False
    assert "inconclusive" in result["verdict"]


def test_asymmetry_check_pools_counts_rather_than_averaging_rates():
    predictions, typology = _toy_arms()
    comparison = compare_arms_at_budget(predictions, typology, budget=6)
    result = asymmetry_check(comparison, "A", "B")
    assert "/3" in result["structured_pooled"]  # CYCLE has 3 positives
    assert "/3" in result["random_pooled"]


# --------------------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------------------


def test_blocks_tile_the_modelling_window():
    from aml.config import load_config

    cfg = load_config()
    blocks = make_blocks(cfg)
    assert len(blocks) == cfg.evaluate.walkforward_blocks
    assert blocks[0][0] == 0
    assert blocks[-1][1] <= cfg.time.max_day


def test_blocks_overrunning_the_window_are_rejected():
    from aml.config import load_config

    cfg = load_config(overrides={"evaluate": {"walkforward_blocks": 9}})
    with pytest.raises(ValueError, match="overrun"):
        make_blocks(cfg)


def _walkforward_frame(frozen_values, retrained_values):
    rows = []
    for block, (frozen, retrained) in enumerate(zip(frozen_values, retrained_values), start=1):
        for arm, value in (("frozen", frozen), ("retrained", retrained)):
            rows.append(
                {"arm": arm, "block": block, "auprc": value,
                 "auprc_lo": value - 0.01, "auprc_hi": value + 0.01}
            )
    return pd.DataFrame(rows)


def test_a_flat_frozen_arm_is_not_reported_as_decay():
    """The correction this function exists for.

    A gap between the arms is not evidence of decay. If the frozen arm holds steady while
    the retrained one improves, that is a training-data-volume effect, and calling it decay
    would be the easiest wrong claim to draw from F6.
    """
    results = _walkforward_frame([0.07, 0.066, 0.066, 0.067], [0.07, 0.42, 0.49, 0.49])
    verdict = summarise(results)["verdict"]
    assert "NOT decay" in verdict
    assert "volume" in verdict


def test_a_genuinely_declining_frozen_arm_is_reported_as_decay():
    results = _walkforward_frame([0.50, 0.40, 0.30, 0.20], [0.50, 0.50, 0.51, 0.50])
    summary = summarise(results)
    assert summary["frozen_declined"] is True
    assert "decay" in summary["verdict"]


def test_no_separation_at_any_block_yields_no_claim():
    results = _walkforward_frame([0.30, 0.30, 0.30, 0.30], [0.30, 0.30, 0.30, 0.30])
    assert "cannot support any claim" in summarise(results)["verdict"]
