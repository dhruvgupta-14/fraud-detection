"""Block D -- motif features (architecture.md 7.4).

This is where the thesis becomes literal: these columns are *shaped like the patterns in
Patterns.txt*. Cycles for CYCLE, degree bursts for FAN-OUT and FAN-IN, their product for
GATHER-SCATTER, value conservation for layering, neighbourhood overlap for BIPARTITE.

Like Block C, every column is read from the ``D - 1`` snapshot.

Bounded search, implemented as sparse linear algebra
----------------------------------------------------
7.4 specifies a bounded-depth directed path search with a per-node branch cap and a
``motif_censored`` flag for nodes that exceed it. **We do not need either**, because the
quantity that search was approximating has a closed form:

    diag(A^2) = row-sum of  A o A^T          2-step closed walks through each node
    diag(A^3) = row-sum of  A^2 o A^T        3-step
    diag(A^4) = row-sum of  A^2 o (A^2)^T    4-step

Measured on the day-9 snapshot: ``A @ A`` is 12.3M non-zeros in 0.24 s, and all three
diagonals take 1.05 s for all 515,088 nodes at once. A depth-4 DFS from every node would
have been far slower *and* would have needed the branch cap that makes the hub's answer
wrong. Exact beats approximate-and-capped when exact is also faster.

**This retires R6** (motif search degenerates on hub nodes): there is no branching to
degenerate. The hub gets an exact answer like every other node.

**What these counts are, precisely.** Closed *walks*, not simple cycles. For length 2 and 3
the two coincide here, because self-loops are excluded from the graph. For length 4 a walk
may retrace a 2-cycle (A→B→A→B→A), so ``cycle_4hop`` is an over-count of genuine 4-cycles.
Stated rather than glossed: it remains a monotone signal of cyclic entanglement, which is
what the feature is for.

Two omissions from the 7.4 list
-------------------------------
* **``chain_depth_est``** (targets STACK) is dropped. Longest-bounded-chain is the one
  quantity here that does not reduce to sparse algebra -- it needs a real traversal, which
  is exactly the cost this module was designed to avoid. STACK therefore has no dedicated
  feature, and the per-typology breakdown (9.3) must say so rather than let a reader assume
  even coverage.
* **``motif_censored``** is unnecessary, per the note above.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from aml.features.base import Causality, FeatureBlock, FeatureContext, FeatureSpec, register

LOGGER = logging.getLogger("aml.features.motifs")

LAGGED = Causality.LAGGED_SNAPSHOT

_PER_NODE: list[tuple[str, str, str]] = [
    ("cycle_2hop", "dormant", "2-step closed walks through this account -- reciprocal pairs"),
    ("cycle_3hop", "dormant", "3-step closed walks -- directed triangles"),
    ("cycle_4hop", "dormant", "4-step closed walks (over-counts retraced 2-cycles)"),
    ("fanout_burst", "cold_start", "new distinct receivers acquired in the last day"),
    ("fanin_burst", "cold_start", "new distinct senders acquired in the last day"),
    ("gather_scatter_score", "cold_start", "fan-in burst x fan-out burst on the same account"),
    ("amount_conservation", "dormant", "value out / value in -- laundering preserves value minus a fee"),
]

_PER_ROW: list[tuple[str, str, str]] = [
    ("common_neighbours", "dormant", "shared neighbours of sender and receiver -- BIPARTITE"),
]


@register
class MotifBlock:
    name = "motifs"
    group = "motif"
    requires_snapshot = True

    def columns(self) -> list[FeatureSpec]:
        cols = [
            FeatureSpec(f"{side}_{suffix}", LAGGED, f"{label} ({side})", null_policy=policy)
            for suffix, policy, label in _PER_NODE
            for side in ("src", "dst")
        ]
        cols += [FeatureSpec(n, LAGGED, label, null_policy=p) for n, p, label in _PER_ROW]
        return cols

    def compute(self, ctx: FeatureContext) -> pd.DataFrame:
        df = ctx.transactions
        names = [spec.name for spec in self.columns()]
        out = pd.DataFrame(np.nan, index=df.index, columns=names, dtype=np.float32)

        days = df["day_idx"].to_numpy()
        src = df["src_node"].to_numpy()
        dst = df["dst_node"].to_numpy()

        for day in np.unique(days):
            if day == 0:
                continue  # cold start; Block C carries the is_cold_start flag
            snapshot = ctx.snapshots.get(day - 1)
            if snapshot is None:
                raise KeyError(f"missing snapshot for day {day - 1}; run scripts/01_graph.py")

            # The burst features need the day before the lagged snapshot to difference
            # against. At day 1 there is no snapshot 0-minus-1, so bursts are null there.
            previous = ctx.snapshots.get(day - 2)
            stats = _snapshot_motifs(snapshot, previous)
            rows = np.flatnonzero(days == day)
            _assign(out, rows, stats, snapshot, src[rows], dst[rows])

        return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Per-snapshot computation
# --------------------------------------------------------------------------------------


def _adjacency(snapshot) -> sp.csr_matrix:
    return sp.csr_matrix(
        (np.ones(snapshot.n_edges, dtype=np.float32), snapshot.indices, snapshot.indptr),
        shape=(snapshot.n_nodes, snapshot.n_nodes),
    )


def _snapshot_motifs(snapshot, previous) -> dict:
    """Every per-node motif quantity for one snapshot."""
    backend = snapshot.backend("igraph", weight="amount_norm")
    active = backend.active_mask()
    in_deg, out_deg = backend.degrees()
    in_str, out_str = backend.strengths()

    A = _adjacency(snapshot)
    At = A.T.tocsr()
    A2 = (A @ A).tocsr()

    cycle_2 = np.asarray(A.multiply(At).sum(axis=1)).ravel()
    cycle_3 = np.asarray(A2.multiply(At).sum(axis=1)).ravel()
    cycle_4 = np.asarray(A2.multiply(A2.T.tocsr()).sum(axis=1)).ravel()
    del A2

    # Burst = distinct counterparties gained since the previous day's snapshot. 7.4 asks
    # for a trailing-Delta-t window; snapshots are daily (2.1), so the natural window is
    # one day and the difference of two cumulative degrees gives it for free.
    if previous is None:
        fanout_burst = np.full(snapshot.n_nodes, np.nan)
        fanin_burst = np.full(snapshot.n_nodes, np.nan)
    else:
        prev_backend = previous.backend("igraph", weight="none")
        prev_in, prev_out = prev_backend.degrees()
        fanout_burst = (out_deg - prev_out).astype(np.float64)
        fanin_burst = (in_deg - prev_in).astype(np.float64)

    # Value conservation: a laundering hop passes on what it received, minus a fee, so a
    # ratio near 1 on a high-throughput account is the layering signature.
    conservation = np.divide(
        out_str, in_str, out=np.full(snapshot.n_nodes, np.nan), where=in_str > 0
    )

    return {
        "cycle_2hop": cycle_2,
        "cycle_3hop": cycle_3,
        "cycle_4hop": cycle_4,
        "fanout_burst": fanout_burst,
        "fanin_burst": fanin_burst,
        "gather_scatter_score": fanin_burst * fanout_burst,
        "amount_conservation": conservation,
        "active": active,
    }


def _common_neighbours(snapshot, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Shared undirected neighbours of each (src, dst) pair -- the BIPARTITE signal.

    Computed by gathering only the rows we need rather than forming the full undirected
    square: A_u @ A_u is 330M non-zeros and 1.3 GB on the day-9 snapshot, against 1.9 s and
    no large intermediate for the row-gather. Same answer, measured.
    """
    directed = _adjacency(snapshot)
    undirected = ((directed + directed.T) > 0).astype(np.float32).tocsr()
    return np.asarray(undirected[src].multiply(undirected[dst]).sum(axis=1)).ravel()


def _assign(
    out: pd.DataFrame, rows: np.ndarray, stats: dict, snapshot, src: np.ndarray, dst: np.ndarray
) -> None:
    active = stats["active"]

    for suffix, _, _ in _PER_NODE:
        values = np.asarray(stats[suffix], dtype=np.float64)
        for side, nodes in (("src", src), ("dst", dst)):
            column = np.where(active[nodes], values[nodes], np.nan)
            # Cast before assigning: pandas deprecates writing float64 into a float32
            # column and will raise on it in a future version.
            out.iloc[rows, out.columns.get_loc(f"{side}_{suffix}")] = column.astype(np.float32)

    shared = _common_neighbours(snapshot, src, dst)
    both_active = active[src] & active[dst]
    out.iloc[rows, out.columns.get_loc("common_neighbours")] = np.where(
        both_active, shared, np.nan
    ).astype(np.float32)


assert isinstance(MotifBlock(), FeatureBlock)
