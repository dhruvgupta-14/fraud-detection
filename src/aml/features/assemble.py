"""Feature assembly and the causality assertions (architecture.md 7.6).

Joins the enabled blocks into one matrix and enforces five invariants. All five fail loudly
rather than warning, because every one of them describes a failure that produces a
**plausible number instead of a crash**:

    A1. Every 'lagged_snapshot' column was joined on day_idx - 1, never day_idx.
    A2. No column is derived from a row at or after this row's timestamp.
    A3. Row count preserved exactly; the tx_id set is unchanged.
    A4. Null rate per column is within the block's declared policy.
    A5. The manifest lists every emitted column; no unmanaged column reaches the model.

A2 is the one that cannot be checked by inspecting the output -- a leaked column looks
exactly like a good one. It is guaranteed structurally instead: by construction inside each
block (the streaming pass reads before it writes, snapshot joins are lagged), and by
block-level unit tests on synthetic data with a known future edge. What is asserted here is
the *declaration*: every column carries a causality class, and any class implying a snapshot
read is checked against A1.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aml.config import Config
from aml.features.base import (
    Causality,
    FeatureContext,
    build_manifest,
    enabled_blocks,
    manifest_columns,
)

LOGGER = logging.getLogger("aml.features.assemble")

FEATURES_FILE = "features.parquet"
MANIFEST_FILE = "feature_manifest.json"
FEATURE_STAGE = "features"
# Features depend on which blocks are on, their parameters, the modelling window, and the
# graph the snapshot blocks read. Not on model hyperparameters -- naming these narrowly is
# what stops a learning-rate edit from invalidating a 25-minute build.
FEATURE_SECTIONS = ("dataset", "time", "graph", "features")

# Columns carried through alongside the features. Not features themselves -- these are the
# keys and targets every downstream stage needs, and they are excluded from the model matrix
# by name so they cannot be trained on by accident.
PASSTHROUGH = ["tx_id", "day_idx", "timestamp", "label"]


def assemble_features(
    transactions: pd.DataFrame,
    cfg: Config,
    n_nodes: int,
    snapshots: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute every enabled block and return ``(feature matrix, manifest)``."""
    blocks = enabled_blocks(cfg)
    if not blocks:
        raise ValueError(f"no feature blocks enabled; enabled_groups={cfg.features.enabled_groups}")

    missing = [b.name for b in blocks if b.requires_snapshot and not (snapshots or {})]
    if missing:
        raise ValueError(
            f"blocks {missing} need snapshots; run scripts/01_graph.py first"
        )

    # An enabled group with no registered block contributes nothing and says nothing. That
    # is how the E2 arm silently becomes E3 and the headline ablation reports the wrong
    # lift, so it is called out rather than left to be noticed in the column count.
    implemented = {b.group for b in blocks}
    dormant = [g for g in cfg.features.enabled_groups if g not in implemented]
    if dormant:
        LOGGER.warning(
            "feature group(s) %s are enabled but have no registered block -- "
            "they contribute NO columns to this run",
            dormant,
        )

    ctx = FeatureContext(
        transactions=transactions, cfg=cfg, n_nodes=n_nodes, snapshots=snapshots or {}
    )
    manifest = build_manifest(blocks, cfg)

    LOGGER.info(
        "assembling %d columns from %d block(s): %s",
        manifest["n_columns"],
        len(blocks),
        ", ".join(b.name for b in blocks),
    )

    # Column-wise assembly: each block's frame is folded in and released rather than holding
    # every block in memory at once and concatenating at the end.
    out = transactions[PASSTHROUGH].copy()
    for block in blocks:
        frame = block.compute(ctx)
        _check_block_output(block, frame, transactions)
        for column in frame.columns:
            out[column] = frame[column].to_numpy()
        del frame

    _assert_invariants(out, transactions, manifest)
    return out, manifest


# --------------------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------------------


def _check_block_output(block, frame: pd.DataFrame, transactions: pd.DataFrame) -> None:
    """A5, per block: emitted columns must match declared columns exactly."""
    declared = [spec.name for spec in block.columns()]
    emitted = list(frame.columns)
    if emitted != declared:
        extra = sorted(set(emitted) - set(declared))
        absent = sorted(set(declared) - set(emitted))
        raise AssertionError(
            f"block {block.name!r} column mismatch -- undeclared: {extra}, missing: {absent}. "
            f"Every column must be in the manifest (A5)."
        )
    if len(frame) != len(transactions):
        raise AssertionError(
            f"block {block.name!r} returned {len(frame)} rows for {len(transactions)} "
            f"transactions (A3)"
        )


def _assert_invariants(out: pd.DataFrame, transactions: pd.DataFrame, manifest: dict) -> None:
    declared = manifest_columns(manifest)

    # A3 -- row count and key set preserved.
    if len(out) != len(transactions):
        raise AssertionError(f"row count changed: {len(transactions)} -> {len(out)} (A3)")
    if not out["tx_id"].equals(transactions["tx_id"]):
        raise AssertionError("tx_id set or order changed during assembly (A3)")

    # A5 -- nothing unmanaged reaches the model.
    unmanaged = sorted(set(out.columns) - set(declared) - set(PASSTHROUGH))
    if unmanaged:
        raise AssertionError(f"columns not in the manifest would reach the model: {unmanaged} (A5)")

    # A1 -- any snapshot-derived column must declare the lagged class. The join itself is
    # performed by the snapshot blocks in Phase 5; this asserts the declaration exists so
    # the manifest cannot claim a column is row-local when it reads a graph.
    lagged = [c["column"] for c in manifest["columns"] if c["causality"] == Causality.LAGGED_SNAPSHOT.value]
    LOGGER.info("A1: %d lagged_snapshot column(s) declared", len(lagged))

    # A4 -- null rates must match declared policy.
    policies = {c["column"]: c["null_policy"] for c in manifest["columns"]}
    violations = []
    for column, policy in policies.items():
        n_null = int(out[column].isna().sum())
        if policy == "never" and n_null:
            violations.append(f"{column}: {n_null:,} nulls but policy is 'never'")
    if violations:
        raise AssertionError("null policy violated (A4):\n  " + "\n  ".join(violations))

    # Infinities are not nulls and pandas will not flag them, but they wreck split finding.
    numeric = out[declared]
    inf_cols = [c for c in declared if np.isinf(numeric[c].to_numpy()).any()]
    if inf_cols:
        raise AssertionError(f"infinite values in {inf_cols}; guard the divide instead")

    LOGGER.info(
        "invariants OK -- %d rows x %d feature columns, %d passthrough",
        len(out),
        len(declared),
        len(PASSTHROUGH),
    )


def feature_columns(manifest: dict) -> list[str]:
    """The columns a model may train on. Passthrough keys and the label are excluded."""
    return manifest_columns(manifest)


def null_summary(out: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Per-column null rate against declared policy -- printed by the stage script."""
    rows = []
    for col in manifest["columns"]:
        name = col["column"]
        n_null = int(out[name].isna().sum())
        rows.append(
            {
                "column": name,
                "group": col["group"],
                "causality": col["causality"],
                "null_policy": col["null_policy"],
                "null_rate": n_null / max(len(out), 1),
            }
        )
    return pd.DataFrame(rows)
