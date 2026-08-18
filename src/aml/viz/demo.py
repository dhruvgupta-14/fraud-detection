"""Demo bundle and ego-network extraction for the Streamlit viewer (architecture.md 10.2).

The viewer is a **results viewer over batch predictions**, not a monitoring system. It reads
what the pipeline already produced; it never scores a transaction live, and nothing here
implements a new algorithm.

**Why a bundle rather than reading the artifacts directly.** The E2 feature matrix is 559 MB
and the prediction table is 1.8M rows. Loading those to display a few hundred alerts would
make the app slow to start and awkward to demo. So one small parquet is precomputed --
top-scoring alerts, their feature values, and their per-row SHAP contributions -- and the app
reads that plus the day-9 snapshot, which is only a few MB.

Ego extraction reuses ``GraphBackend.neighbors`` from Phase 2. No new traversal is written.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("aml.viz.demo")

BUNDLE_STAGE = "demo"
ALERTS_FILE = "alerts.parquet"
SHAP_FILE = "alert_shap.parquet"
SUMMARY_FILE = "demo_summary.json"

# Enough rows to browse convincingly without making the bundle large.
DEFAULT_TOP_ALERTS = 300
DEFAULT_SAMPLE_OTHERS = 200
# Display cap for the ego network. The hub has 14,230 counterparties (architecture.md 2.1);
# drawing them is neither readable nor useful. This is a rendering limit, not an algorithm.
MAX_NEIGHBOURS = 12


def build_alert_table(
    predictions: pd.DataFrame,
    transactions: pd.DataFrame,
    typology_map: pd.DataFrame,
    node_index: pd.DataFrame,
    threshold: float,
    top_n: int = DEFAULT_TOP_ALERTS,
    n_sample: int = DEFAULT_SAMPLE_OTHERS,
    seed: int = 42,
) -> pd.DataFrame:
    """Top-scoring test transactions plus a random sample, joined to readable columns.

    The sample matters for the demo: a table containing only the model's top alerts makes
    every row look like a hit. Including lower-scored rows lets the viewer show what a
    confident negative looks like too.
    """
    test = predictions.loc[predictions["split"] == "test"].copy()
    ranked = test.sort_values("score", ascending=False)

    top = ranked.head(top_n)
    rest = ranked.iloc[top_n:]
    rng = np.random.default_rng(seed)
    take = min(n_sample, len(rest))
    sample = rest.iloc[rng.choice(len(rest), size=take, replace=False)] if take else rest
    selected = pd.concat([top, sample], ignore_index=True)

    columns = [
        "tx_id", "timestamp", "day_idx", "src_node", "dst_node",
        "src_bank", "dst_bank", "amount_paid", "amount_received",
        "currency_paid", "payment_format", "is_self_loop", "is_cross_bank",
    ]
    merged = selected.merge(transactions[columns], on="tx_id", how="left")
    merged = merged.merge(
        typology_map[["tx_id", "attempt_id", "typology"]], on="tx_id", how="left"
    )

    names = node_index.set_index("node_id")["acct"]
    merged["src_acct"] = merged["src_node"].map(names)
    merged["dst_acct"] = merged["dst_node"].map(names)

    merged["alert"] = merged["score"] >= threshold
    merged["rank"] = merged["score"].rank(ascending=False, method="first").astype(int)
    return merged.sort_values("score", ascending=False, ignore_index=True)


def alert_shap(
    model,
    features: pd.DataFrame,
    feature_columns: list[str],
    tx_ids: np.ndarray,
    manifest: dict,
) -> pd.DataFrame:
    """Per-row SHAP contributions for the bundled alerts.

    Long format (one row per transaction per feature) so the app can pull the top drivers
    for a selected row without loading a wide matrix.
    """
    import shap

    rows = features["tx_id"].isin(tx_ids)
    subset = features.loc[rows]
    X = subset[feature_columns]

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X.to_numpy())
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    group_of = {c["column"]: c["group"] for c in manifest["columns"]}
    long = pd.DataFrame(values, columns=feature_columns)
    long["tx_id"] = subset["tx_id"].to_numpy()
    melted = long.melt(id_vars="tx_id", var_name="feature", value_name="shap")
    melted["group"] = melted["feature"].map(group_of)

    # Carry the raw feature value so the app can say "turnover_latency = 42s" alongside the
    # contribution, which is what makes an explanation legible to an analyst.
    raw = subset[["tx_id"] + feature_columns].melt(
        id_vars="tx_id", var_name="feature", value_name="value"
    )
    return melted.merge(raw, on=["tx_id", "feature"], how="left")


def ego_edges(
    snapshot,
    node: int,
    max_neighbours: int = MAX_NEIGHBOURS,
) -> pd.DataFrame:
    """1-hop neighbourhood of ``node`` as an edge list.

    Reuses ``GraphBackend.neighbors`` (Phase 2, component 9) in both directions. The cap is
    applied by descending edge weight so the strongest relationships survive truncation, and
    the caller is told how many were hidden rather than being shown a silently short list.
    """
    backend = snapshot.backend("igraph", weight="amount_norm")
    records = []

    for direction, label in (("out", "sends to"), ("in", "receives from")):
        neighbours = backend.neighbors(node, direction)
        if len(neighbours) == 0:
            continue
        weights = _edge_weights(snapshot, node, neighbours, direction)
        order = np.argsort(-weights)[:max_neighbours]
        for i in order:
            other = int(neighbours[i])
            records.append(
                {
                    "source": node if direction == "out" else other,
                    "target": other if direction == "out" else node,
                    "direction": label,
                    "weight": float(weights[i]),
                    "hidden": max(len(neighbours) - max_neighbours, 0),
                }
            )

    return pd.DataFrame(records, columns=["source", "target", "direction", "weight", "hidden"])


def _edge_weights(snapshot, node: int, neighbours: np.ndarray, direction: str) -> np.ndarray:
    """Transaction counts on the edges between ``node`` and each neighbour."""
    if direction == "out":
        start, end = snapshot.indptr[node], snapshot.indptr[node + 1]
        targets = snapshot.indices[start:end]
        counts = snapshot.tx_count[start:end]
        lookup = dict(zip(targets.tolist(), counts.tolist()))
        return np.array([lookup.get(int(n), 0) for n in neighbours], dtype=np.float64)

    # In-direction: scan the neighbours' out-edges for this node.
    weights = []
    for other in neighbours:
        start, end = snapshot.indptr[other], snapshot.indptr[other + 1]
        targets = snapshot.indices[start:end]
        match = np.flatnonzero(targets == node)
        weights.append(float(snapshot.tx_count[start:end][match[0]]) if len(match) else 0.0)
    return np.array(weights, dtype=np.float64)
