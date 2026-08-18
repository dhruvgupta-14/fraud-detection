"""SHAP attribution (architecture.md 9.4) -- figure F5.

Answers one question directly: **do graph-derived or raw transaction features dominate the
model's decisions?** Aggregated by feature group, that is the cleanest evidence for the
thesis that does not depend on a metric delta.

It matters more here than originally planned. The per-typology exhibit (F3) turned out to be
near-saturated -- every arm catches 97-100 % of annotated positives -- so it cannot
demonstrate the structured-vs-RANDOM asymmetry. SHAP by group is the remaining direct
mechanism evidence, so it is treated as a primary exhibit rather than a supporting one.

**Sampled, not exhaustive.** TreeSHAP on 1.35M test rows is minutes of work for a figure
that plots eight bars. A stratified sample -- all positives plus a random draw of negatives
-- gives the same group ranking at a fraction of the cost, and the sample size is reported
alongside the figure so the reader knows what it rests on.

**A leak alarm, not a win.** Per 11.3, a graph feature dominating SHAP is something to
investigate before it is something to celebrate. That rule was written for
``community_illicit_prior``, which was dropped in Phase 5; it still applies to whatever
ranks first here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("aml.evaluate.explain")

DEFAULT_SAMPLE = 20_000


def stratified_sample(
    labels: np.ndarray, indices: np.ndarray, size: int, seed: int
) -> np.ndarray:
    """All positives plus a random draw of negatives, capped at ``size``.

    Positives are 0.1 % of rows; a uniform sample of 20,000 would contain roughly 22 of
    them and the attribution would be dominated by negatives. Keeping every positive is
    what makes the explanation about the thing being detected.
    """
    y = labels[indices]
    positives = indices[y == 1]
    negatives = indices[y == 0]
    n_neg = max(size - len(positives), 0)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(negatives, size=min(n_neg, len(negatives)), replace=False)
    return np.sort(np.concatenate([positives, chosen]))


def shap_by_group(
    model,
    features: pd.DataFrame,
    feature_columns: list[str],
    manifest: dict,
    indices: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sample_size: int = DEFAULT_SAMPLE,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Mean |SHAP| per column and per feature group.

    Returns ``(per_column, per_group, n_sampled)``. Group membership comes from the feature
    manifest, so the figure cannot drift out of sync with what was actually built.
    """
    import shap

    sample = stratified_sample(labels, indices, sample_size, seed)
    X = features[feature_columns].iloc[sample]
    LOGGER.info(
        "TreeSHAP on %d sampled rows (%d positives)", len(sample), int(labels[sample].sum())
    )

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X.to_numpy())
    # LightGBM binary returns either one array or a list of two; the positive class is what
    # we explain either way.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    group_of = {c["column"]: c["group"] for c in manifest["columns"]}
    causality_of = {c["column"]: c["causality"] for c in manifest["columns"]}

    per_column = pd.DataFrame(
        {
            "column": feature_columns,
            "group": [group_of.get(c, "unknown") for c in feature_columns],
            "causality": [causality_of.get(c, "unknown") for c in feature_columns],
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    total = per_column["mean_abs_shap"].sum()
    per_group = (
        per_column.groupby("group", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), n_columns=("column", "size"))
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    per_group["share"] = per_group["mean_abs_shap"] / total if total else np.nan
    # Share per column matters as much as share per group: a group with 32 columns will
    # out-total a group with 17 simply by being larger, which says nothing about strength.
    per_group["share_per_column"] = per_group["share"] / per_group["n_columns"]

    return per_column, per_group, len(sample)
