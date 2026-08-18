"""Evaluation metrics (architecture.md 9.1, 9.2, 8.4).

**Accuracy is never computed here, and that is deliberate.** At 0.102 % prevalence,
predicting all-negative scores 99.9 % and catches nothing. Any number that can be gamed by
predicting nothing is not a measurement.

Primary is **AUPRC**, which unlike ROC-AUC responds to the negatives that dominate this
problem. ROC-AUC is computed and reported once, as a footnote, explicitly labelled
misleading -- omitting it invites the question, so it is better answered than dodged.

The **threshold policy** (8.4) is the other thing this module exists to enforce: the
operating point is chosen on validation at the recall target and applied *unchanged* to
test. Picking a threshold on test is the most common leak in hackathon submissions and it
inflates every downstream business number.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

LOGGER = logging.getLogger("aml.evaluate.metrics")


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve. The headline number."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Reported as a footnote only. See the module docstring."""
    return float(roc_auc_score(y_true, scores))


def threshold_at_recall(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Lowest-alert threshold on **validation** achieving at least ``target`` recall.

    Returned to be applied unchanged to test. Among thresholds that meet the recall target
    we take the strictest (highest) one, because that is the operating point a compliance
    team would actually choose: hit the mandated recall while sending the fewest alerts.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns len(thresholds) == len(recall) - 1; the final point is
    # recall=0, precision=1 with no corresponding threshold.
    usable = recall[:-1] >= target
    if not usable.any():
        LOGGER.warning(
            "recall target %.2f unreachable (max %.3f); falling back to the lowest threshold",
            target,
            float(recall[:-1].max()) if len(recall) > 1 else 0.0,
        )
        return float(thresholds.min())
    return float(thresholds[usable].max())


def alerts_per_day(scores: np.ndarray, threshold: float, n_days: int) -> float:
    """The business-framed metric (9.2): how many alerts analysts receive each day.

    This is what converts an abstract AUPRC gap into a staffing statement -- "same recall,
    N fewer alerts per day" -- and is far more persuasive to a compliance audience than an
    F1 score.
    """
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")
    return float((scores >= threshold).sum()) / n_days


def bootstrap_auprc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI over test-set resamples (R8).

    The test window holds ~1,495 positives, so differences between models are often inside
    the noise. Reporting an interval is what stops us ranking two arms on a gap that is not
    there -- the architecture pre-commits to never ranking on a difference inside the CI.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        resampled = y_true[idx]
        # A resample with no positives has undefined AUPRC; skip rather than score it 0,
        # which would drag the interval down for a purely arithmetic reason.
        if resampled.sum() == 0:
            continue
        values.append(average_precision_score(resampled, scores[idx]))
    if not values:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def evaluate_split(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_days: int,
    prevalence_baseline: float | None = None,
) -> dict:
    """Every reported number for one split at one operating point."""
    flagged = scores >= threshold
    n_flagged = int(flagged.sum())
    true_positives = int((flagged & (y_true == 1)).sum())
    n_positives = int(y_true.sum())

    prevalence = n_positives / len(y_true)
    score = auprc(y_true, scores)

    return {
        "n_rows": int(len(y_true)),
        "n_positives": n_positives,
        "prevalence": prevalence,
        "auprc": score,
        # A random ranker scores AUPRC == prevalence, so the ratio is the honest statement
        # of how much the model actually adds. An AUPRC of 0.05 sounds poor until you note
        # the floor is 0.001.
        "auprc_lift_over_random": score / prevalence if prevalence else float("nan"),
        "roc_auc_footnote": roc_auc(y_true, scores),
        "threshold": threshold,
        "n_flagged": n_flagged,
        "alert_rate": n_flagged / len(y_true),
        "alerts_per_day": alerts_per_day(scores, threshold, n_days),
        "recall_at_threshold": true_positives / n_positives if n_positives else float("nan"),
        "precision_at_threshold": true_positives / n_flagged if n_flagged else float("nan"),
    }


def pr_curve(y_true: np.ndarray, scores: np.ndarray, max_points: int = 2000) -> dict:
    """Precision-recall curve, thinned for plotting and JSON storage.

    At 1.35M test rows the raw curve has as many points; storing all of them would bloat
    every metrics file for a figure that cannot render them anyway.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    if len(precision) > max_points:
        keep = np.linspace(0, len(precision) - 1, max_points).astype(int)
        precision, recall = precision[keep], recall[keep]
    return {"precision": precision.tolist(), "recall": recall.tolist()}
