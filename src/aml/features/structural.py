"""Block C -- structural features from the lagged snapshot (architecture.md 7.3).

**This is where the temporal contract is actually cashed in.** A transaction on day ``D``
reads its structural features from the snapshot built at ``D - 1``. Not ``D``: the day-``D``
snapshot contains the transaction itself, so using it would let a transaction help compute
its own PageRank -- the exact leak this project claims to have solved. The lag is applied in
one place, in ``_per_day_lookup``, so there is a single line to audit.

Day 0 has no prior snapshot. Its structural columns are null and carry ``is_cold_start``.
Dormant accounts -- present in the node space but with no edge in the window -- are also
null rather than being given a teleport-only PageRank that means nothing.

Feature list follows 7.3 with **one deliberate omission**, described below.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

from aml.features.base import Causality, FeatureBlock, FeatureContext, FeatureSpec, register

LOGGER = logging.getLogger("aml.features.structural")

LAGGED = Causality.LAGGED_SNAPSHOT

# Emitted for both endpoints. (suffix, null policy, description)
_PER_NODE: list[tuple[str, str, str]] = [
    ("pagerank", "dormant", "PageRank on the D-1 graph"),
    ("pagerank_rank_pct", "dormant", "percentile rank of PageRank among active nodes"),
    ("in_degree_centrality", "dormant", "distinct senders / (n_active - 1)"),
    ("out_degree_centrality", "dormant", "distinct receivers / (n_active - 1)"),
    ("community_size", "dormant", "size of the Louvain community"),
    ("community_density", "dormant", "internal edges / possible internal edges"),
    ("ego_size", "dormant", "1-hop neighbourhood size, undirected"),
    ("ego_mean_degree", "dormant", "mean degree of immediate neighbours"),
    ("ego_max_degree", "dormant", "max degree of immediate neighbours"),
    ("neighbour_pagerank_mean", "dormant", "mean PageRank of immediate neighbours"),
]

# Emitted once per row, not per endpoint.
_PER_ROW: list[tuple[str, str, str]] = [
    ("same_community", "dormant", "sender and receiver share a Louvain community"),
    ("is_cold_start", "never", "day 0 -- no prior snapshot exists"),
]


@register
class StructuralBlock:
    name = "structural"
    group = "structural"
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

        weighted = ctx.cfg.graph.pagerank_weighted
        # Phase 2 measured 100% top-100 overlap between raw and currency-normalised
        # weighting, so this choice is low-stakes -- but amount_norm is the dimensionally
        # coherent one and defending it costs nothing (architecture.md 16, Phase 2).
        weight_kind = "amount_norm" if weighted else "none"

        days = df["day_idx"].to_numpy()
        src = df["src_node"].to_numpy()
        dst = df["dst_node"].to_numpy()

        out["is_cold_start"] = (days == 0).astype(np.float32)

        for day in np.unique(days):
            if day == 0:
                continue  # no snapshot at D-1; columns stay null, is_cold_start marks it
            snapshot = ctx.snapshots.get(day - 1)
            if snapshot is None:
                raise KeyError(f"missing snapshot for day {day - 1}; run scripts/01_graph.py")

            stats = _snapshot_stats(snapshot, weight_kind, ctx.cfg.seed, ctx.cfg.graph.pagerank_damping)
            rows = np.flatnonzero(days == day)
            _assign(out, rows, stats, src[rows], dst[rows])

        return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Per-snapshot computation
# --------------------------------------------------------------------------------------


def _snapshot_stats(snapshot, weight_kind: str, seed: int, damping: float) -> dict:
    """Every per-node quantity Block C needs, computed once for one snapshot."""
    backend = snapshot.backend("igraph", weight=weight_kind)
    n = snapshot.n_nodes

    active = backend.active_mask()
    n_active = int(active.sum())
    in_deg, out_deg = backend.degrees()

    pagerank = backend.pagerank(damping=damping, weighted=(weight_kind != "none"))
    community = backend.communities(seed=seed)

    # Percentile rank among ACTIVE nodes only. Including the ~92K dormant accounts would
    # pile them all on the same teleport value and compress the informative range.
    rank_pct = np.full(n, np.nan, dtype=np.float64)
    if n_active:
        active_idx = np.flatnonzero(active)
        order = np.argsort(pagerank[active_idx])
        ranks = np.empty(len(active_idx), dtype=np.float64)
        ranks[order] = np.arange(len(active_idx))
        rank_pct[active_idx] = ranks / max(len(active_idx) - 1, 1)

    # Undirected 1-hop neighbourhood for ego and community statistics: "who does this
    # account transact with" is a question about adjacency, not about direction.
    undirected = _undirected(snapshot)
    ego_size = np.asarray(undirected.sum(axis=1)).ravel()
    safe_size = np.maximum(ego_size, 1)
    ego_mean_degree = np.asarray(undirected @ ego_size).ravel() / safe_size
    ego_max_degree = _segment_max(undirected, ego_size)
    neighbour_pr = np.asarray(undirected @ pagerank).ravel() / safe_size

    comm_size, comm_density = _community_stats(community, undirected, n)

    denominator = max(n_active - 1, 1)
    return {
        "pagerank": pagerank,
        "pagerank_rank_pct": rank_pct,
        "in_degree_centrality": in_deg / denominator,
        "out_degree_centrality": out_deg / denominator,
        "community_size": comm_size,
        "community_density": comm_density,
        "ego_size": ego_size,
        "ego_mean_degree": ego_mean_degree,
        "ego_max_degree": ego_max_degree,
        "neighbour_pagerank_mean": neighbour_pr,
        "community": community,
        "active": active,
    }


def _undirected(snapshot) -> sp.csr_matrix:
    """Boolean undirected adjacency, self-loops already excluded upstream."""
    directed = sp.csr_matrix(
        (np.ones(snapshot.n_edges, dtype=np.float32), snapshot.indices, snapshot.indptr),
        shape=(snapshot.n_nodes, snapshot.n_nodes),
    )
    return ((directed + directed.T) > 0).astype(np.float32).tocsr()


def _segment_max(adjacency: sp.csr_matrix, values: np.ndarray) -> np.ndarray:
    """Max of ``values`` over each row's neighbours. Zero for isolated nodes."""
    weighted = adjacency.multiply(values[np.newaxis, :]).tocsr()
    result = np.zeros(adjacency.shape[0], dtype=np.float64)
    if weighted.nnz:
        maxima = weighted.max(axis=1).toarray().ravel()
        result = maxima
    return result


def _community_stats(
    community: np.ndarray, undirected: sp.csr_matrix, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Community size and internal edge density, broadcast back to nodes.

    Density is internal edges over possible internal edges. Laundering rings are small and
    dense, so the pair (small size, high density) is the signature -- either alone is weak.
    """
    sizes = np.bincount(community, minlength=int(community.max()) + 1)

    # Count edges whose endpoints share a community.
    src = np.repeat(np.arange(n, dtype=np.int32), np.diff(undirected.indptr))
    dst = undirected.indices
    internal = community[src] == community[dst]
    internal_edges = np.bincount(
        community[src][internal], minlength=len(sizes)
    ) / 2.0  # each undirected edge counted from both ends

    size_per_node = sizes[community].astype(np.float64)
    possible = size_per_node * (size_per_node - 1) / 2.0
    density = np.divide(
        internal_edges[community],
        possible,
        out=np.zeros(n, dtype=np.float64),
        where=possible > 0,
    )
    return size_per_node, density


def _assign(
    out: pd.DataFrame, rows: np.ndarray, stats: dict, src: np.ndarray, dst: np.ndarray
) -> None:
    """Write one day's rows by indexing the D-1 per-node arrays at both endpoints."""
    active = stats["active"]

    for suffix, _, _ in _PER_NODE:
        values = stats[suffix]
        for side, nodes in (("src", src), ("dst", dst)):
            column = np.asarray(values, dtype=np.float64)[nodes]
            # Dormant accounts have no meaningful graph position; null rather than zero.
            column = np.where(active[nodes], column, np.nan)
            # Cast before assigning: pandas deprecates writing float64 into a float32
            # column and will raise on it in a future version.
            out.iloc[rows, out.columns.get_loc(f"{side}_{suffix}")] = column.astype(np.float32)

    community = stats["community"]
    both_active = active[src] & active[dst]
    same = np.where(both_active, (community[src] == community[dst]).astype(np.float64), np.nan)
    out.iloc[rows, out.columns.get_loc("same_community")] = same.astype(np.float32)


assert isinstance(StructuralBlock(), FeatureBlock)
