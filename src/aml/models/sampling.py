"""Negative sub-sampling for training (architecture.md 8.3).

At 0.102 % prevalence, most rows carry no information a tree can use. Keeping every positive
and a fixed multiple of negatives makes training fast without changing what the model can
learn, provided the resulting probability shift is corrected -- which is what
``scale_pos_weight`` / ``class_weight`` do.

**The one rule that matters: this is for TRAIN only.** Sub-sampling negatives from a test set
inflates AUPRC by construction, because AUPRC depends directly on prevalence. It would raise
the headline number while measuring nothing, and it is the kind of error that survives review
because the code looks identical either way. So the function *requires* the caller to name
the split and refuses anything but ``"train"`` -- a guard rail rather than a comment.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger("aml.models.sampling")


def subsample_negatives(
    labels: np.ndarray,
    indices: np.ndarray,
    ratio: int,
    seed: int,
    split: str,
) -> np.ndarray:
    """Keep every positive and ``ratio`` negatives per positive, from ``indices``.

    ``split`` must be ``"train"``. Returns positions into the same frame ``indices`` refers
    to, sorted, so downstream row order stays time-ordered.
    """
    if split != "train":
        raise ValueError(
            f"negative sub-sampling is train-only; got split={split!r}. Sub-sampling "
            f"val or test inflates AUPRC by construction (architecture.md 8.3)."
        )
    if ratio < 1:
        raise ValueError(f"negative_ratio must be >= 1, got {ratio}")

    y = labels[indices]
    positives = indices[y == 1]
    negatives = indices[y == 0]

    if len(positives) == 0:
        raise ValueError("cannot sub-sample: the training split contains no positives")

    target = min(len(negatives), ratio * len(positives))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(negatives, size=target, replace=False)

    kept = np.sort(np.concatenate([positives, chosen]))
    LOGGER.info(
        "sampled train: %d positives + %d of %d negatives (1:%d) = %d rows, %.1f%% of original",
        len(positives),
        target,
        len(negatives),
        ratio,
        len(kept),
        100.0 * len(kept) / len(indices),
    )
    return kept


def positive_weight(labels: np.ndarray, indices: np.ndarray) -> float:
    """``scale_pos_weight`` for the sampled set: negatives per positive.

    Applied so the model's scores stay comparable across arms even though the training
    prevalence was changed. Ranking metrics like AUPRC are invariant to a monotone rescale,
    but the operating threshold (8.4) is not, and that is chosen on un-sampled validation.
    """
    y = labels[indices]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (n_neg / n_pos) if n_pos else 1.0
