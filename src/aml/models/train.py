"""Fit, persist, predict (architecture.md 8.1-8.4).

One function does the whole job for one model, because the order of operations *is* the
methodology and splitting it across helpers would let a future edit reorder it:

    1. split temporally, purging straddling attempts out of train
    2. sub-sample negatives -- train only
    3. fit on the sampled train rows
    4. score the FULL val and test sets, never sub-sampled
    5. choose the threshold on val at the recall target
    6. apply that same threshold to test, unchanged

Steps 4 and 6 are where the common leaks live, so they are in one readable sequence.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from aml.config import Config
from aml.evaluate.metrics import bootstrap_auprc_ci, evaluate_split, pr_curve
from aml.models.registry import get_model
from aml.models.sampling import positive_weight, subsample_negatives
from aml.models.splits import Split

LOGGER = logging.getLogger("aml.models.train")

MODEL_STAGE = "models"
# Everything that changes what a model learns. Feature sections are included because a
# model trained on a different feature matrix is a different model.
MODEL_SECTIONS = ("dataset", "time", "graph", "features", "sampling", "models")


def train_and_score(
    name: str,
    features: pd.DataFrame,
    feature_columns: list[str],
    split: Split,
    cfg: Config,
) -> tuple[object, pd.DataFrame, dict]:
    """Train one model and score val and test. Returns ``(model, predictions, metrics)``."""
    labels = features["label"].to_numpy()

    # --- 2. sample (train only; the function refuses any other split) ---
    train_idx = subsample_negatives(
        labels, split.train, cfg.sampling.negative_ratio, cfg.seed, split="train"
    )
    pos_weight = positive_weight(labels, train_idx)

    X = features[feature_columns]
    model = get_model(name, cfg, pos_weight)

    # --- 3. fit ---
    started = time.perf_counter()
    model.fit(X.iloc[train_idx].to_numpy(), labels[train_idx])
    fit_secs = time.perf_counter() - started
    LOGGER.info("%s: fitted on %d rows in %.1fs", name, len(train_idx), fit_secs)

    # --- 4. score the FULL val and test sets ---
    val_scores = _positive_scores(model, X.iloc[split.val].to_numpy())
    test_scores = _positive_scores(model, X.iloc[split.test].to_numpy())

    # --- 5. threshold chosen on validation ---
    from aml.evaluate.metrics import threshold_at_recall

    threshold = threshold_at_recall(
        labels[split.val], val_scores, cfg.models.recall_target
    )

    # --- 6. same threshold applied to test ---
    val_days = cfg.time.val_days[1] - cfg.time.val_days[0] + 1
    test_days = cfg.time.test_days[1] - cfg.time.test_days[0] + 1

    metrics = {
        "model": name,
        "fit_seconds": round(fit_secs, 2),
        "n_train_rows": int(len(train_idx)),
        "n_train_positives": int(labels[train_idx].sum()),
        "scale_pos_weight": round(pos_weight, 2),
        "threshold_source": "validation",
        "recall_target": cfg.models.recall_target,
        "val": evaluate_split(labels[split.val], val_scores, threshold, val_days),
        "test": evaluate_split(labels[split.test], test_scores, threshold, test_days),
    }
    metrics["test"]["auprc_ci95"] = bootstrap_auprc_ci(
        labels[split.test], test_scores, cfg.evaluate.bootstrap_iterations, cfg.seed
    )
    metrics["test"]["pr_curve"] = pr_curve(labels[split.test], test_scores)

    predictions = pd.DataFrame(
        {
            "tx_id": np.concatenate(
                [
                    features["tx_id"].to_numpy()[split.val],
                    features["tx_id"].to_numpy()[split.test],
                ]
            ),
            "split": ["val"] * len(split.val) + ["test"] * len(split.test),
            "score": np.concatenate([val_scores, test_scores]),
            "label": np.concatenate([labels[split.val], labels[split.test]]),
        }
    )
    return model, predictions, metrics


def _positive_scores(model, X: np.ndarray) -> np.ndarray:
    """Probability of the positive class.

    ``predict_proba`` column order follows ``model.classes_``, so the positive column is
    located rather than assumed to be index 1 -- it is index 1 for every estimator here,
    but a silently wrong column would invert every score and still produce a valid-looking
    AUPRC near 1 - x.
    """
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    return proba[:, classes.index(1)]
