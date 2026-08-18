"""Guards on the snapshot builder (architecture.md section 6.2, component 10).

The snapshot is where the temporal contract is physically implemented, so the tests that
matter here are the ones that would otherwise fail *silently*: a window that includes one
day too many still produces a graph, still produces features, and still produces an AUPRC.
It just produces a leaked one.

Synthetic fixtures throughout, with every expected number derived in the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.config import load_config
from aml.graph.snapshots import (
    Snapshot,
    build_snapshot,
    collapse_edges,
    currency_medians,
    window_bounds,
)

N_NODES = 6


def cfg_for(lookback: int | None = None, max_day: int = 3):
    """A real Config with the time window overridden to something test-sized."""
    return load_config(
        overrides={
            "time": {
                "lookback_days": lookback,
                "max_day": max_day,
                "train_days": [0, 1],
                "val_days": [2, 2],
                "test_days": [3, max_day],
            }
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    """Four days of traffic, including a repeated pair and a self-loop.

    day 0:  0->1 (100 USD), 0->1 (300 USD)   <- parallel pair, must collapse to one edge
    day 1:  1->2 (200 USD)
    day 2:  2->2 (500 USD)                   <- self-loop, must never become an edge
    day 3:  2->3 (7 BTC), 3->0 (900 USD)
    """
    rows = [
        (0, 0, 1, 100.0, "US Dollar", "2022-09-01 01:00", False),
        (0, 0, 1, 300.0, "US Dollar", "2022-09-01 05:00", False),
        (1, 1, 2, 200.0, "US Dollar", "2022-09-02 01:00", False),
        (2, 2, 2, 500.0, "US Dollar", "2022-09-03 01:00", True),
        (3, 2, 3, 7.0, "Bitcoin", "2022-09-04 01:00", False),
        (3, 3, 0, 900.0, "US Dollar", "2022-09-04 02:00", False),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "day_idx",
            "src_node",
            "dst_node",
            "amount_paid",
            "currency_paid",
            "timestamp",
            "is_self_loop",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_idx"] = df["day_idx"].astype(np.int16)
    df["src_node"] = df["src_node"].astype(np.int32)
    df["dst_node"] = df["dst_node"].astype(np.int32)
    return df


def graph_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df["is_self_loop"]]


# --------------------------------------------------------------------------------------
# Window semantics -- the off-by-one that would mislabel every E7 result
# --------------------------------------------------------------------------------------


def test_cumulative_window_starts_at_day_zero():
    assert window_bounds(0, None) == (0, 0)
    assert window_bounds(5, None) == (0, 5)


def test_lookback_window_spans_exactly_that_many_days():
    """A 3-day lookback must cover 3 days, not 4.

    Section 6.2 writes the bound as ``day_idx - lookback_days <= d``, which spans
    lookback_days + 1 days. Implementing that literally would mean the E7 sweep's "3d"
    point was really 4d and its "7d" point really 8d -- a published number that is simply
    mislabelled. This asserts the corrected semantics.
    """
    assert window_bounds(5, 3) == (3, 5)  # days 3, 4, 5 -- three days
    assert window_bounds(5, 1) == (5, 5)  # one day: today only
    assert window_bounds(9, 7) == (3, 9)  # seven days


def test_lookback_window_clamps_at_the_start_of_history():
    assert window_bounds(1, 7) == (0, 1)


def test_zero_lookback_is_rejected():
    with pytest.raises(ValueError, match="lookback_days"):
        window_bounds(3, 0)


# --------------------------------------------------------------------------------------
# Causality -- the whole point of the module
# --------------------------------------------------------------------------------------


def test_snapshot_never_contains_a_future_row(frame):
    """The single most important property in the file.

    A snapshot for day D that contained a day D+1 row would let a transaction contribute to
    its own features once joined at the D-1 lag, which is precisely the leak this project
    claims to have solved. Nothing would crash; the AUPRC would just be too good.
    """
    cfg = cfg_for(lookback=None)
    rows = graph_rows(frame)
    for day in range(4):
        snap = build_snapshot(rows, day, N_NODES, cfg)
        in_window = rows.loc[rows["day_idx"] <= day]
        assert snap.n_rows == len(in_window)
        assert snap.tx_count.sum() == len(in_window)


def test_cumulative_snapshots_are_monotone_in_edges(frame):
    cfg = cfg_for(lookback=None)
    rows = graph_rows(frame)
    counts = [build_snapshot(rows, d, N_NODES, cfg).n_edges for d in range(4)]
    assert counts == sorted(counts)


def test_lookback_snapshot_drops_rows_that_fell_out_of_the_window(frame):
    cfg = cfg_for(lookback=1)
    rows = graph_rows(frame)
    snap = build_snapshot(rows, 1, N_NODES, cfg)
    # Day 1 only: the single 1->2 transaction. Day 0's two rows are outside the window.
    assert snap.n_rows == 1
    assert snap.n_edges == 1


# --------------------------------------------------------------------------------------
# Self-loops
# --------------------------------------------------------------------------------------


def test_self_loops_never_become_edges(frame):
    """Self-loops are 11.6% of rows and carry ~zero signal (architecture.md 2.1).

    They must not inflate degree -- but they remain scoreable transactions, so the caller
    filters them for the *graph* view only. This asserts the graph view is clean.
    """
    cfg = cfg_for(lookback=None)
    snap = build_snapshot(graph_rows(frame), 3, N_NODES, cfg)
    sources = np.repeat(np.arange(N_NODES, dtype=np.int32), np.diff(snap.indptr))
    assert not np.any(sources == snap.indices)
    # And the backend, which rejects self-loops outright, accepts this snapshot.
    assert snap.backend("igraph").n_edges == snap.n_edges


# --------------------------------------------------------------------------------------
# Collapsing
# --------------------------------------------------------------------------------------


def test_parallel_edges_collapse_but_keep_their_totals(frame):
    cfg = cfg_for(lookback=None)
    snap = build_snapshot(graph_rows(frame), 0, N_NODES, cfg)
    # Two 0->1 transactions become one edge carrying both.
    assert snap.n_edges == 1
    assert snap.tx_count.tolist() == [2]
    assert snap.weight_amount.tolist() == [400.0]  # 100 + 300
    assert snap.n_rows == 2


def test_collapse_records_the_most_recent_timestamp():
    """last_ts must be the max within the pair, not the first or an arbitrary one."""
    indptr, indices, w, wn, counts, last_ts = collapse_edges(
        src=np.array([0, 0, 0], dtype=np.int32),
        dst=np.array([1, 1, 1], dtype=np.int32),
        amount=np.array([1.0, 2.0, 3.0]),
        amount_norm=np.array([1.0, 2.0, 3.0]),
        ts=np.array([500, 100, 300], dtype=np.int64),  # deliberately unsorted
        n_nodes=N_NODES,
    )
    assert counts.tolist() == [3]
    assert last_ts.tolist() == [500]


def test_collapsed_edges_come_out_in_csr_order():
    """The packed key sorts by (src, dst), which is CSR order -- no second sort needed."""
    indptr, indices, *_ = collapse_edges(
        src=np.array([2, 0, 1, 0], dtype=np.int32),
        dst=np.array([3, 4, 2, 1], dtype=np.int32),
        amount=np.ones(4),
        amount_norm=np.ones(4),
        ts=np.arange(4, dtype=np.int64),
        n_nodes=N_NODES,
    )
    sources = np.repeat(np.arange(N_NODES, dtype=np.int32), np.diff(indptr))
    assert sources.tolist() == [0, 0, 1, 2]
    assert indices.tolist() == [1, 4, 2, 3]


def test_collapse_of_an_empty_window_is_a_valid_empty_graph():
    indptr, indices, w, wn, counts, last_ts = collapse_edges(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([]),
        np.array([]),
        np.array([], dtype=np.int64),
        N_NODES,
    )
    assert len(indptr) == N_NODES + 1
    assert indptr.tolist() == [0] * (N_NODES + 1)
    assert len(indices) == 0


# --------------------------------------------------------------------------------------
# Currency normalisation -- the unit bug this exists to avoid
# --------------------------------------------------------------------------------------


def test_currency_medians_are_computed_per_currency(frame):
    medians = currency_medians(graph_rows(frame))
    assert medians["Bitcoin"] == 7.0
    assert medians["US Dollar"] == 250.0  # median of 100, 200, 300, 900


def test_normalised_weight_puts_currencies_on_a_comparable_scale(frame):
    """Raw amounts mix 15 currencies spanning six orders of magnitude.

    A 7 BTC transaction and a 900 USD one are both roughly one typical transaction in their
    own currency; under the raw sum the USD edge outweighs the BTC edge 129:1 purely
    because of the unit it is denominated in.
    """
    cfg = cfg_for(lookback=None)
    snap = build_snapshot(graph_rows(frame), 3, N_NODES, cfg)

    raw = dict(zip(snap.indices.tolist(), snap.weight_amount.tolist()))
    norm = dict(zip(snap.indices.tolist(), snap.weight_amount_norm.tolist()))
    btc_edge, usd_edge = 3, 0  # 2->3 is the Bitcoin edge; 3->0 is the 900 USD edge

    assert raw[usd_edge] / raw[btc_edge] > 100  # the distortion, in the raw weight
    assert 0.1 < norm[usd_edge] / norm[btc_edge] < 10  # removed by normalising


def test_weight_selection_covers_every_kind(frame):
    cfg = cfg_for(lookback=None)
    snap = build_snapshot(graph_rows(frame), 0, N_NODES, cfg)
    assert snap.weights("amount").tolist() == [400.0]
    assert snap.weights("tx_count").tolist() == [2.0]
    assert snap.weights("none").tolist() == [1.0]
    with pytest.raises(ValueError, match="weight kind"):
        snap.weights("vibes")


# --------------------------------------------------------------------------------------
# Structure and persistence
# --------------------------------------------------------------------------------------


def test_snapshot_spans_the_full_node_space(frame):
    """Arrays must be node_id-indexable, so the graph is never compacted to active nodes."""
    cfg = cfg_for(lookback=None)
    snap = build_snapshot(graph_rows(frame), 0, N_NODES, cfg)
    assert snap.n_nodes == N_NODES
    assert len(snap.indptr) == N_NODES + 1
    assert len(snap.active_mask()) == N_NODES
    # Day 0 touches only nodes 0 and 1.
    assert snap.active_mask().tolist() == [True, True, False, False, False, False]


def test_snapshot_roundtrips_through_disk(tmp_path, frame):
    from aml.io import StageStore

    cfg = cfg_for(lookback=None)
    snap = build_snapshot(graph_rows(frame), 3, N_NODES, cfg)
    store = StageStore(tmp_path, "snapshots", "testhash")
    store.dir.mkdir(parents=True, exist_ok=True)
    snap.save(store)

    loaded = Snapshot.load(store, 3)
    assert loaded.day_idx == snap.day_idx
    assert loaded.lookback_days == snap.lookback_days
    assert loaded.n_nodes == snap.n_nodes
    assert loaded.n_rows == snap.n_rows
    assert np.array_equal(loaded.indptr, snap.indptr)
    assert np.array_equal(loaded.indices, snap.indices)
    assert np.array_equal(loaded.weight_amount, snap.weight_amount)
    assert np.array_equal(loaded.tx_count, snap.tx_count)
    assert np.array_equal(loaded.last_ts, snap.last_ts)


def test_cumulative_lookback_roundtrips_as_none(tmp_path, frame):
    """None is stored as a sentinel int; it must come back as None, not as -1."""
    from aml.io import StageStore

    cfg = cfg_for(lookback=None)
    snap = build_snapshot(graph_rows(frame), 0, N_NODES, cfg)
    store = StageStore(tmp_path, "snapshots", "h")
    store.dir.mkdir(parents=True, exist_ok=True)
    snap.save(store)
    assert Snapshot.load(store, 0).lookback_days is None
