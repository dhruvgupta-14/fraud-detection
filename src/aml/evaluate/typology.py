"""Per-typology recall (architecture.md 9.3) -- figure F3.

**This is the mechanism check, and it is the exhibit that can falsify Phase 5's result.**

The thesis predicts an *asymmetry*: graph features should help disproportionately on
structurally-patterned typologies (CYCLE, FAN-OUT, FAN-IN, SCATTER-GATHER) and much less on
RANDOM, which has no structure to detect. If the graph lift were uniform across every family
*including* RANDOM, that would be evidence something is leaking rather than evidence the
features work -- a suspiciously even improvement is the signature of a feature that knows
something it should not.

So this table is not decoration on top of the headline AUPRC. It is the test of whether the
headline came from the mechanism we claim.

Two honesty constraints, both enforced here rather than left to the writing:

1. **Coverage is partial.** Only 913 of 1,495 test positives carry a typology annotation
   (61 %); the rest form an explicit ``UNANNOTATED`` bucket. Reporting per-typology recall as
   if it covered all positives would be a quiet error.
2. **N is small per family** -- RANDOM has 56 positives, BIPARTITE 68. A three-positive
   difference is noise. Every rate is therefore reported with its count and a Wilson
   confidence interval, and the report does not rank families whose intervals overlap.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("aml.evaluate.typology")

UNANNOTATED = "UNANNOTATED"

# Families the thesis predicts graph structure should help, versus the control family.
STRUCTURED = ("CYCLE", "FAN-OUT", "FAN-IN", "SCATTER-GATHER", "GATHER-SCATTER", "BIPARTITE")
CONTROL = "RANDOM"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of a bootstrap because these are small counts of a binary outcome, where
    Wilson is both exact enough and well-behaved at the boundaries -- a family where every
    positive was detected gets an honest interval rather than [1.0, 1.0].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def attach_typology(
    predictions: pd.DataFrame, typology_map: pd.DataFrame, split: str = "test"
) -> pd.DataFrame:
    """Join scored rows to their typology, bucketing unannotated positives explicitly."""
    rows = predictions.loc[predictions["split"] == split].copy()
    merged = rows.merge(
        typology_map[["tx_id", "attempt_id", "typology"]], on="tx_id", how="left"
    )
    positives = merged.loc[merged["label"] == 1].copy()
    positives["typology"] = positives["typology"].fillna(UNANNOTATED)
    return positives


def typology_recall(
    predictions: pd.DataFrame,
    typology_map: pd.DataFrame,
    threshold: float,
    split: str = "test",
) -> pd.DataFrame:
    """Recall per typology at the validation-chosen threshold.

    One row per family plus ``UNANNOTATED``, with counts and a Wilson interval so the
    reader can see which differences are real.
    """
    positives = attach_typology(predictions, typology_map, split)
    positives["detected"] = positives["score"] >= threshold

    records = []
    for typology, group in positives.groupby("typology", observed=True):
        n = len(group)
        detected = int(group["detected"].sum())
        lo, hi = wilson_interval(detected, n)
        records.append(
            {
                "typology": typology,
                "n_positives": n,
                "n_detected": detected,
                "recall": detected / n if n else float("nan"),
                "recall_lo": lo,
                "recall_hi": hi,
                "n_attempts": int(group["attempt_id"].nunique()),
            }
        )

    table = pd.DataFrame(records)
    # Annotated families first by size, with the UNANNOTATED bucket pinned last so it
    # reads as the caveat it is rather than as another typology.
    table["_order"] = np.where(table["typology"] == UNANNOTATED, 1, 0)
    table = table.sort_values(["_order", "n_positives"], ascending=[True, False])
    return table.drop(columns="_order").reset_index(drop=True)


def compare_arms(
    arms: dict[str, tuple[pd.DataFrame, float]],
    typology_map: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Per-typology recall for several arms side by side.

    ``arms`` maps an arm label to ``(predictions, threshold)``. Each arm keeps its own
    threshold because each chose one on its own validation split (8.4) -- forcing a shared
    threshold would compare the arms at different recall targets.
    """
    frames = []
    for label, (predictions, threshold) in arms.items():
        table = typology_recall(predictions, typology_map, threshold, split)
        table.insert(0, "arm", label)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def threshold_for_budget(scores: np.ndarray, n_alerts: int) -> float:
    """Threshold that flags exactly ``n_alerts`` rows -- the highest-scoring ones."""
    if n_alerts <= 0:
        raise ValueError(f"alert budget must be positive, got {n_alerts}")
    n_alerts = min(n_alerts, len(scores))
    return float(np.partition(scores, -n_alerts)[-n_alerts])


def compare_arms_at_budget(
    arms: dict[str, pd.DataFrame],
    typology_map: pd.DataFrame,
    budget: int,
    split: str = "test",
) -> pd.DataFrame:
    """Per-typology recall with every arm held to the **same alert budget**.

    **This replaces the fixed-recall comparison specified in 9.3, which is degenerate.**
    Each arm's operating threshold is chosen on validation to hit 90 % recall (8.4), so
    comparing per-typology recall across arms at those thresholds compares 90 % against
    90 % -- the answer is flat before any data is involved. Measured: all three arms score
    0.97-1.00 on every annotated family, including the tabular-only baseline.

    The AUPRC difference between arms lives in **precision**, not in recall on annotated
    rows. So the question that actually distinguishes them is the one a compliance team
    would ask: *given the same number of alerts an analyst can work, which typologies does
    each arm catch?* Fixing the budget and letting recall vary answers that; fixing recall
    and letting the budget vary does not.
    """
    frames = []
    for label, predictions in arms.items():
        rows = predictions.loc[predictions["split"] == split]
        threshold = threshold_for_budget(rows["score"].to_numpy(), budget)
        table = typology_recall(predictions, typology_map, threshold, split)
        table.insert(0, "arm", label)
        table["threshold"] = threshold
        table["budget"] = budget
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def asymmetry_check(comparison: pd.DataFrame, baseline_arm: str, graph_arm: str) -> dict:
    """Does the graph lift concentrate on structured typologies, as the thesis predicts?

    **Counts are pooled, not rates averaged, and no verdict is issued unless the intervals
    separate.** An earlier version averaged the per-family recall deltas and compared that
    to RANDOM's delta. On this data it declared a leak alarm off the back of RANDOM moving
    from 53/56 to 55/56 detected -- a two-positive change on the smallest family, which is
    exactly the noise 9.3 warns against ranking on. Averaging rates gives a 56-positive
    family the same weight as a 181-positive one.

    Pooling successes and trials across the structured families weights by N naturally, and
    requiring the Wilson intervals to be disjoint before naming a verdict stops the check
    from manufacturing findings out of single-digit count changes.
    """
    needed = {baseline_arm, graph_arm}
    if not needed <= set(comparison["arm"].unique()):
        raise KeyError(f"need both {baseline_arm!r} and {graph_arm!r} in the comparison")

    def pooled(arm: str, families: list[str]) -> tuple[int, int]:
        rows = comparison[(comparison["arm"] == arm) & (comparison["typology"].isin(families))]
        return int(rows["n_detected"].sum()), int(rows["n_positives"].sum())

    present = set(comparison["typology"].unique())
    structured = [t for t in STRUCTURED if t in present]

    struct_base, struct_n = pooled(baseline_arm, structured)
    struct_graph, _ = pooled(graph_arm, structured)
    rand_base, rand_n = pooled(baseline_arm, [CONTROL])
    rand_graph, _ = pooled(graph_arm, [CONTROL])

    struct_delta = (struct_graph - struct_base) / struct_n if struct_n else float("nan")
    rand_delta = (rand_graph - rand_base) / rand_n if rand_n else float("nan")

    # Intervals on the graph-arm rates; if these overlap, the two families are not
    # distinguishable and the honest answer is "inconclusive at this sample size".
    struct_ci = wilson_interval(struct_graph, struct_n)
    rand_ci = wilson_interval(rand_graph, rand_n)
    disjoint = struct_ci[0] > rand_ci[1] or rand_ci[0] > struct_ci[1]

    if not disjoint:
        verdict = (
            f"inconclusive -- structured {struct_graph}/{struct_n} and RANDOM "
            f"{rand_graph}/{rand_n} have overlapping intervals at this sample size"
        )
    elif struct_delta > rand_delta:
        verdict = "supports mechanism -- lift concentrates on structured typologies"
    else:
        verdict = "LEAK ALARM -- graph features help RANDOM more than structured families"

    pivot = comparison.pivot_table(
        index="typology", columns="arm", values="recall", observed=True
    )
    delta = pivot[graph_arm] - pivot[baseline_arm]

    return {
        "structured_pooled": f"{struct_base}/{struct_n} -> {struct_graph}/{struct_n}",
        "structured_delta": float(struct_delta),
        "structured_ci95": struct_ci,
        "random_pooled": f"{rand_base}/{rand_n} -> {rand_graph}/{rand_n}",
        "random_delta": float(rand_delta),
        "random_ci95": rand_ci,
        "intervals_disjoint": bool(disjoint),
        "verdict": verdict,
        "per_typology_delta": {k: float(v) for k, v in delta.items() if not np.isnan(v)},
    }
