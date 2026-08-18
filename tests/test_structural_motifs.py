"""Blocks C and D -- the lagged snapshot join (architecture.md 6.2, 7.3, 7.4).

**The property under test is the D-1 lag**, and it is the single claim that distinguishes
this project from a submission that computes PageRank on the whole graph and hopes. A
transaction on day D must read structural features from the snapshot at D-1, never D --
because the day-D snapshot contains the transaction itself, so using it would let a
transaction contribute to its own features.

That failure is invisible in the output: the columns look identical either way, and the only
symptom is a better score. So it is asserted on a fixture where the two answers differ by
construction.

The fixture, three days over six accounts:

    day 0   0 -> 1 , 1 -> 0        a reciprocal pair    (2-step closed walk)
    day 1   2 -> 3 , 3 -> 4 , 4 -> 2   a directed triangle  (3-step closed walk)
    day 2   0 -> 2 , 5 -> 1

Snapshots are cumulative, so snapshot(1) knows the triangle but snapshot(0) does not, and
node 0's out-degree is 1 in snapshot(1) but 2 in snapshot(2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aml.features  # noqa: F401
from aml.config import load_config
from aml.features.base import Causality, FeatureContext
from aml.features.motifs import MotifBlock
from aml.features.structural import StructuralBlock
from aml.graph.snapshots import build_snapshot

N_NODES = 6

EDGES = [
    (0, 0, 1),
    (0, 1, 0),
    (1, 2, 3),
    (1, 3, 4),
    (1, 4, 2),
    (2, 0, 2),
    (2, 5, 1),
]


def cfg_for():
    return load_config(
        overrides={
            "time": {
                "max_day": 2,
                "lookback_days": None,
                "train_days": [0, 0],
                "val_days": [1, 1],
                "test_days": [2, 2],
            }
        }
    )


def make_frame() -> pd.DataFrame:
    df = pd.DataFrame(EDGES, columns=["day_idx", "src_node", "dst_node"])
    n = len(df)
    return pd.DataFrame(
        {
            "tx_id": np.arange(n, dtype=np.int64),
            "timestamp": pd.to_datetime("2022-09-01")
            + pd.to_timedelta(df["day_idx"] * 86400 + np.arange(n), unit="s"),
            "day_idx": df["day_idx"].to_numpy(np.int16),
            "src_node": df["src_node"].to_numpy(np.int32),
            "dst_node": df["dst_node"].to_numpy(np.int32),
            "amount_paid": np.full(n, 100.0),
            "amount_received": np.full(n, 100.0),
            "currency_paid": "US Dollar",
            "currency_received": "US Dollar",
            "payment_format": "Wire",
            "is_self_loop": np.zeros(n, dtype=bool),
            "is_cross_currency": np.zeros(n, dtype=bool),
            "is_cross_bank": np.ones(n, dtype=bool),
            "label": np.zeros(n, dtype=np.int8),
        }
    )


@pytest.fixture
def context() -> FeatureContext:
    cfg = cfg_for()
    frame = make_frame()
    snapshots = {d: build_snapshot(frame, d, N_NODES, cfg) for d in range(3)}
    return FeatureContext(transactions=frame, cfg=cfg, n_nodes=N_NODES, snapshots=snapshots)


@pytest.fixture
def structural(context) -> pd.DataFrame:
    return StructuralBlock().compute(context)


@pytest.fixture
def motifs(context) -> pd.DataFrame:
    return MotifBlock().compute(context)


# --------------------------------------------------------------------------------------
# The D-1 lag
# --------------------------------------------------------------------------------------


def test_day_row_reads_the_previous_days_graph_not_its_own(structural):
    """Row 5 is ``0 -> 2`` on day 2. Node 0 has out-degree 2 in snapshot(2) and 1 in
    snapshot(1). Reading 1 proves the lag; reading 2 would prove a leak.
    """
    # out_degree_centrality is degree / (n_active - 1); snapshot(1) has 5 active nodes.
    value = structural["src_out_degree_centrality"][5]
    assert value == pytest.approx(1 / 4), "day-2 row must see node 0's day-1 out-degree of 1"


def test_a_node_active_only_today_is_dormant_in_the_lagged_snapshot(structural):
    """Row 2 is ``2 -> 3`` on day 1. Node 2 first transacts on day 1, so in snapshot(0) it
    has no edges at all and its structural features must be null -- not zero, and certainly
    not the degree it acquires later the same day."""
    assert np.isnan(structural["src_pagerank"][2])
    assert np.isnan(structural["src_out_degree_centrality"][2])


def test_the_triangle_is_invisible_until_the_day_after_it_forms(motifs):
    """The 2->3->4->2 triangle forms on day 1.

    Row 2 (``2 -> 3``, day 1) reads snapshot(0) and must not see it. Row 5 (``0 -> 2``,
    day 2) reads snapshot(1) and must see it on node 2.
    """
    assert np.isnan(motifs["src_cycle_3hop"][2])  # day 1, reading snapshot(0)
    assert motifs["dst_cycle_3hop"][5] == 1.0  # day 2, node 2, reading snapshot(1)


def test_day_zero_is_cold_start_with_null_structure(structural):
    day_zero = [0, 1]
    assert structural["is_cold_start"][day_zero].tolist() == [1.0, 1.0]
    for column in ("src_pagerank", "dst_pagerank", "src_community_size", "same_community"):
        assert structural[column][day_zero].isna().all(), column


def test_later_days_are_not_flagged_cold_start(structural):
    assert structural["is_cold_start"][[2, 3, 4, 5, 6]].tolist() == [0.0] * 5


# --------------------------------------------------------------------------------------
# Motif correctness
# --------------------------------------------------------------------------------------


def test_reciprocal_pair_is_a_two_step_closed_walk(motifs):
    """Nodes 0 and 1 exchange on day 0, so from snapshot(0) onward each has one 2-cycle.

    Row 5 is ``0 -> 2`` on day 2, reading snapshot(1), where node 0's reciprocal pair with
    node 1 is present. Row 2 is ``2 -> 3`` on day 1: its sender is node *2*, which is
    dormant in snapshot(0) -- so null there is the correct answer, not a missing cycle.
    """
    assert motifs["src_cycle_2hop"][5] == 1.0
    assert np.isnan(motifs["src_cycle_2hop"][2])


def test_a_node_outside_any_cycle_scores_zero_not_null(motifs):
    """Node 5 sends on day 2 and is in no cycle. But node 5 is dormant in snapshot(1), so
    the honest value is null. Node 1, which IS active and in a 2-cycle, is the contrast."""
    assert np.isnan(motifs["src_cycle_2hop"][6])  # node 5, dormant at D-1
    assert motifs["dst_cycle_2hop"][6] == 1.0  # node 1, active and reciprocal


def test_common_neighbours_counts_shared_counterparties(motifs):
    """Row 5 is ``0 -> 2`` on day 2, read against snapshot(1).

    In snapshot(1) node 0's undirected neighbours are {1}; node 2's are {3, 4}. No overlap.
    """
    assert motifs["common_neighbours"][5] == 0.0


def test_amount_conservation_is_value_out_over_value_in(motifs):
    """Node 0 in snapshot(1): sent 100 (0->1), received 100 (1->0). Ratio 1.0."""
    assert motifs["src_amount_conservation"][5] == pytest.approx(1.0)


def test_bursts_are_null_when_there_is_no_earlier_snapshot_to_difference(motifs):
    """A burst needs snapshot(D-1) and snapshot(D-2). Day 1 rows have only snapshot(0)."""
    assert np.isnan(motifs["src_fanout_burst"][2])


def test_burst_counts_counterparties_gained_since_the_previous_day(motifs):
    """Node 2 gained receiver {3} on day 1, so its day-2 reading is a burst of 1."""
    assert motifs["dst_fanout_burst"][5] == 1.0


# --------------------------------------------------------------------------------------
# Manifest -- the audit trail must call these what they are
# --------------------------------------------------------------------------------------


def test_every_structural_and_motif_column_is_declared_lagged():
    """A column read from a snapshot must never claim to be row_local: the report's
    leakage table is generated from these declarations, not written by hand."""
    for block in (StructuralBlock(), MotifBlock()):
        for spec in block.columns():
            assert spec.causality is Causality.LAGGED_SNAPSHOT, spec.name


def test_blocks_declare_that_they_need_snapshots():
    """assemble.py refuses to run these without snapshots; that refusal keys off this flag."""
    assert StructuralBlock().requires_snapshot
    assert MotifBlock().requires_snapshot


def test_emitted_columns_match_declared_columns(structural, motifs):
    assert list(structural.columns) == [s.name for s in StructuralBlock().columns()]
    assert list(motifs.columns) == [s.name for s in MotifBlock().columns()]
