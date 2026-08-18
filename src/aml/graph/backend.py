"""Graph compute backends over a snapshot's CSR adjacency.

Two implementations behind one protocol, per architecture.md section 6.1:

* ``IGraphBackend``  -- C-backed PageRank and Louvain. The production path.
* ``NetworkxBackend`` -- the same quantities, computed slowly, on the same inputs.

The second exists to make "we used igraph for speed" an implementation note rather than a
methodological change. Running both on a sampled config and getting the same answer is the
Phase 2 checkpoint; if they ever diverge, that is a bug in our adapter and not a licence to
prefer whichever number looks better.

**Why the backend takes CSR arrays rather than a ``Snapshot``.** Inverting the dependency
stated in section 6.1: ``snapshots.py`` depends on this module, not the reverse. That makes
the backend testable against six-node graphs with hand-checkable PageRank, with no ingest
and no parquet involved, and it keeps the equivalence check honest -- see below.

**What is and is not backend-dependent.** Degree, strength, transaction count, neighbours
and the active mask are exact CSR arithmetic with one correct answer, so they are computed
once in NumPy on the shared base class. Only PageRank and Louvain genuinely differ between
libraries, so those are the only two methods the subclasses implement -- and therefore the
only two the equivalence check has to cover. Verifying ``np.bincount`` against a second
implementation of ``np.bincount`` would be theatre.

Node space
----------
Every backend spans the **full** node space (all 515,088 interned accounts), not just the
accounts active in the snapshot window. Returned arrays are therefore indexed directly by
``node_id`` with no remapping layer.

The cost is that a dormant account receives a small teleport-only PageRank instead of
nothing at all. That is handled by ``active_mask()``, which the feature layer uses to null
dormant nodes explicitly -- not by compacting the graph. A dense, node-id-indexed array
plus a mask has no off-by-one failure mode; a compacted graph with a remap table has a
whole family of them, and they surface as plausible-looking wrong features rather than as
a crash.
"""

from __future__ import annotations

import logging
import random
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp

LOGGER = logging.getLogger("aml.graph.backend")

OUT = "out"
IN = "in"
_DIRECTIONS = (OUT, IN)


# --------------------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class GraphBackend(Protocol):
    """The surface the feature layer is written against.

    Every array returned is length ``n_nodes`` and indexed by ``node_id``.
    """

    n_nodes: int
    n_edges: int

    def pagerank(self, damping: float = 0.85, weighted: bool = True) -> np.ndarray: ...

    def communities(self, seed: int) -> np.ndarray: ...

    def degrees(self) -> tuple[np.ndarray, np.ndarray]: ...

    def strengths(self) -> tuple[np.ndarray, np.ndarray]: ...

    def tx_counts(self) -> tuple[np.ndarray, np.ndarray]: ...

    def neighbors(self, node: int, direction: str = OUT) -> np.ndarray: ...

    def active_mask(self) -> np.ndarray: ...


# --------------------------------------------------------------------------------------
# Shared CSR arithmetic
# --------------------------------------------------------------------------------------


class _CSRBackend:
    """Everything computable exactly from the CSR arrays, shared by both backends.

    The adjacency is **collapsed**: at most one entry per ordered ``(src, dst)`` pair, with
    the parallel transactions between that pair summarised into ``weight`` (total amount)
    and ``tx_count`` (how many transactions). Self-loops are excluded upstream at snapshot
    build time (``cfg.graph.exclude_self_loops``), so this class does not re-filter them --
    but it does assert their absence, because a self-loop reaching this far means the
    snapshot builder is wrong and every degree downstream is quietly inflated.
    """

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weight: np.ndarray,
        tx_count: np.ndarray,
        n_nodes: int,
    ) -> None:
        indptr = np.asarray(indptr)
        indices = np.asarray(indices)

        if indptr.ndim != 1 or len(indptr) != n_nodes + 1:
            raise ValueError(
                f"indptr must have length n_nodes + 1 = {n_nodes + 1}, got {len(indptr)}"
            )
        if indptr[0] != 0:
            raise ValueError(f"indptr must start at 0, got {indptr[0]}")
        n_edges = int(indptr[-1])
        for name, arr in (("indices", indices), ("weight", weight), ("tx_count", tx_count)):
            if len(arr) != n_edges:
                raise ValueError(
                    f"{name} has length {len(arr)} but indptr implies {n_edges} edges"
                )
        if n_edges and (indices.min() < 0 or indices.max() >= n_nodes):
            raise ValueError(
                f"indices out of range [0, {n_nodes}): "
                f"min={indices.min()} max={indices.max()}"
            )

        self.n_nodes = int(n_nodes)
        self.n_edges = n_edges
        self.indptr = indptr.astype(np.int64, copy=False)
        self.indices = indices.astype(np.int32, copy=False)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.tx_count = np.asarray(tx_count, dtype=np.int64)

        # Lazy caches, initialised before the self-loop check below because that check is
        # itself a _sources() caller.
        self._csc: sp.csc_matrix | None = None
        self._src_cache: np.ndarray | None = None

        # A self-loop here means the snapshot builder failed to exclude one. That would
        # inflate degree and strength on exactly the 11.6% of rows that carry no signal
        # (architecture.md section 2.1), so it fails loudly rather than being filtered
        # away silently -- a silent fix here would hide the real bug upstream.
        if n_edges:
            sources = self._sources()
            if np.any(sources == self.indices):
                offenders = np.unique(sources[sources == self.indices])[:5]
                raise ValueError(
                    f"Adjacency contains {int(np.sum(sources == self.indices))} self-loop(s), "
                    f"e.g. at nodes {offenders.tolist()}. Self-loops must be excluded by the "
                    f"snapshot builder; see architecture.md section 2.1."
                )

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_edges(
        cls,
        src: np.ndarray,
        dst: np.ndarray,
        weight: np.ndarray,
        tx_count: np.ndarray,
        n_nodes: int,
    ) -> "_CSRBackend":
        """Build from an already-collapsed, ``src``-sorted edge list."""
        src = np.asarray(src, dtype=np.int64)
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=n_nodes), out=indptr[1:])
        return cls(indptr, dst, weight, tx_count, n_nodes)

    def _sources(self) -> np.ndarray:
        """Expand ``indptr`` back into a per-edge source array.

        Cached: four callers need it and at 4.5M edges the ``repeat`` is not free.
        """
        if self._src_cache is None:
            self._src_cache = np.repeat(
                np.arange(self.n_nodes, dtype=np.int32),
                np.diff(self.indptr).astype(np.int64),
            )
        return self._src_cache

    def _as_csc(self) -> sp.csc_matrix:
        """Transpose view, built once, for in-direction neighbour lookups."""
        if self._csc is None:
            self._csc = sp.csr_matrix(
                (self.weight, self.indices, self.indptr),
                shape=(self.n_nodes, self.n_nodes),
            ).tocsc()
        return self._csc

    # ------------------------------------------------------------------ exact quantities

    def degrees(self) -> tuple[np.ndarray, np.ndarray]:
        """``(in_degree, out_degree)`` as **distinct counterparties**.

        Not transaction counts. Because parallel edges are collapsed, the CSR structure
        counts how many *different* accounts a node sent to or received from, which is the
        quantity the fan-out / fan-in typologies are actually about. Use ``tx_counts()``
        for volume of activity; the two differ by an order of magnitude on hub nodes and
        conflating them is how a "max out-degree" figure ends up meaning two things at once.
        """
        out_deg = np.diff(self.indptr).astype(np.int32)
        in_deg = np.bincount(self.indices, minlength=self.n_nodes).astype(np.int32)
        return in_deg, out_deg

    def strengths(self) -> tuple[np.ndarray, np.ndarray]:
        """``(in_strength, out_strength)`` -- total amount received / sent in the window.

        ``bincount`` on both sides rather than ``add.reduceat`` on the out side: reduceat
        returns the element *at* the index for an empty segment instead of zero, and
        isolated nodes are the common case here, not an edge case.
        """
        if not self.n_edges:
            zeros = np.zeros(self.n_nodes, dtype=np.float64)
            return zeros, zeros.copy()
        out_str = np.bincount(
            self._sources(), weights=self.weight, minlength=self.n_nodes
        ).astype(np.float64)
        in_str = np.bincount(
            self.indices, weights=self.weight, minlength=self.n_nodes
        ).astype(np.float64)
        return in_str, out_str

    def tx_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """``(in_tx, out_tx)`` -- number of transactions, parallel edges expanded back out.

        Not part of the section 6.1 protocol; added because it is free from the collapsed
        ``tx_count`` array and because it is the quantity that disambiguates the R10 hub
        figure (architecture.md section 2.1 quotes an out-degree and an edge share that
        cannot both be the same measurement).
        """
        if not self.n_edges:
            zeros = np.zeros(self.n_nodes, dtype=np.int64)
            return zeros, zeros.copy()
        counts = self.tx_count.astype(np.float64)
        out_tx = np.bincount(self._sources(), weights=counts, minlength=self.n_nodes)
        in_tx = np.bincount(self.indices, weights=counts, minlength=self.n_nodes)
        return in_tx.astype(np.int64), out_tx.astype(np.int64)

    def neighbors(self, node: int, direction: str = OUT) -> np.ndarray:
        """Distinct counterparties of ``node`` in the given direction.

        The motif search (Block D) is the heavy caller. It applies its own branch cap on
        top of this; the backend deliberately does not truncate, so that the caller can
        record ``motif_censored`` rather than silently receiving a short list.
        """
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
        if not 0 <= node < self.n_nodes:
            raise IndexError(f"node {node} out of range [0, {self.n_nodes})")
        if direction == OUT:
            return self.indices[self.indptr[node] : self.indptr[node + 1]]
        csc = self._as_csc()
        return csc.indices[csc.indptr[node] : csc.indptr[node + 1]].astype(np.int32)

    def active_mask(self) -> np.ndarray:
        """True for nodes with at least one edge in the window, in either direction.

        The feature layer uses this to null structural features for dormant accounts
        rather than joining them a teleport-only PageRank that means nothing.
        """
        in_deg, out_deg = self.degrees()
        return (in_deg > 0) | (out_deg > 0)

    # ------------------------------------------------------------------ helpers

    def _edge_array(self) -> np.ndarray:
        """``(n_edges, 2)`` int32 source/target array, the shape both libraries accept."""
        return np.column_stack([self._sources(), self.indices]).astype(np.int32)

    def _pagerank_weights(self, weighted: bool) -> list[float] | None:
        """Edge weights as a plain list.

        igraph interprets a string or numpy array in the ``weights`` slot as an *edge
        attribute name* and raises ``KeyError: 'Attribute does not exist'``, so the
        conversion is not cosmetic -- passing the array through fails outright.
        """
        return self.weight.tolist() if weighted else None

    def _undirected_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Collapse the digraph to an undirected multigraph edge list with summed weights.

        Louvain maximises modularity, which is only defined on undirected graphs, so A->B
        and B->A must merge into a single edge. This is a genuine modelling choice, not an
        API detail: it asserts that a laundering ring is a dense subgraph regardless of
        which way the money moved. Defensible -- and stated in the report rather than
        applied silently.
        """
        edges = self._edge_array()
        if not len(edges):
            return edges, self.weight
        # Canonical ordering (min, max) so the two directions land on the same key; the
        # caller simplifies with combine_edges="sum".
        lo = np.minimum(edges[:, 0], edges[:, 1])
        hi = np.maximum(edges[:, 0], edges[:, 1])
        return np.column_stack([lo, hi]).astype(np.int32), self.weight


# --------------------------------------------------------------------------------------
# igraph -- the production path
# --------------------------------------------------------------------------------------


class IGraphBackend(_CSRBackend):
    """PageRank via PRPACK and Louvain via ``community_multilevel``.

    PRPACK solves the PageRank system directly rather than iterating to a tolerance, so it
    is deterministic and has no convergence knob to get wrong. Louvain is stochastic, so
    the seed is passed through igraph's RNG explicitly -- see ``communities``.
    """

    def pagerank(self, damping: float = 0.85, weighted: bool = True) -> np.ndarray:
        import igraph

        if not self.n_edges:
            # An edgeless graph is uniform by definition. igraph agrees, but saying so
            # here means the empty day-0-with-lookback case is covered by construction.
            return np.full(self.n_nodes, 1.0 / self.n_nodes, dtype=np.float64)

        g = igraph.Graph(n=self.n_nodes, edges=self._edge_array(), directed=True)
        scores = g.pagerank(
            damping=damping,
            weights=self._pagerank_weights(weighted),
            directed=True,
            implementation="prpack",
        )
        return np.asarray(scores, dtype=np.float64)

    def communities(self, seed: int) -> np.ndarray:
        import igraph

        if not self.n_edges:
            return np.arange(self.n_nodes, dtype=np.int32)

        edges, weights = self._undirected_edges()
        g = igraph.Graph(n=self.n_nodes, edges=edges, directed=False)
        g.es["weight"] = weights.tolist()
        # Merge the two directions of each pair into one weighted edge (see
        # _undirected_edges); without this a reciprocal pair counts twice in modularity.
        g.simplify(multiple=True, loops=False, combine_edges={"weight": "sum"})

        # Louvain is stochastic and igraph draws from its own RNG, which by default is the
        # `random` module's global state. Handing it a private, seeded generator makes the
        # community assignment reproducible without depending on whatever else in the
        # process has touched `random` first.
        #
        # igraph exposes no getter for the current generator -- set_random_number_generator
        # returns None -- so we cannot save and restore what was there before. Passing None
        # reverts to igraph's own C-layer PCG32 default, which is the library's initial
        # state. Nothing else in this pipeline installs a generator, so that is a faithful
        # restore; if that ever changes, this is where it will bite.
        igraph.set_random_number_generator(random.Random(seed))
        try:
            clustering = g.community_multilevel(weights="weight")
        finally:
            igraph.set_random_number_generator(None)

        return np.asarray(clustering.membership, dtype=np.int32)


# --------------------------------------------------------------------------------------
# networkx -- the equivalence check
# --------------------------------------------------------------------------------------


class NetworkxBackend(_CSRBackend):
    """The same two quantities via networkx. Correct, and far slower.

    Not a fallback we expect to use in anger: at 515K nodes and 4.5M edges this is minutes
    per snapshot against seconds. It exists so the Phase 2 checkpoint can demonstrate that
    the igraph numbers are not an artifact of the library, on a sampled config.

    **networkx's default PageRank tolerance is not usable at this scale, and that is the
    strongest argument in the project for not taking the brief's "networkx is acceptable"
    at face value.** networkx solves by power iteration and compares its L1 error against
    ``N * tol``, so the accuracy of the result *degrades as the graph grows*. Measured on a
    95,841-node induced subgraph of the day-9 snapshot, against PRPACK's exact solve:

    ==============  ====================  ==========================
    networkx tol    max abs difference    relative error, top node
    ==============  ====================  ==========================
    1e-6 (default)  1.72e-04              **44 %**
    1e-10           5.74e-08              0.015 %
    1e-14           3.10e-12              ~0
    ==============  ====================  ==========================

    It converges cleanly to the igraph answer, so the adapter is faithful -- but a naive
    ``nx.pagerank(g)`` would have produced a silently 44 %-wrong centrality on exactly the
    high-mass accounts this project is about, and nothing would have crashed. ``PAGERANK_TOL``
    below is therefore set far tighter than the library default, deliberately.

    Louvain community *ids* are arbitrary and will not match between runs, let alone
    libraries, so community structure is compared by pairwise agreement, never by label.
    """

    # Far tighter than networkx's 1e-6 default -- see the class docstring. The cost is a
    # few extra power iterations; the alternative is a wrong feature that looks fine.
    PAGERANK_TOL = 1e-12
    PAGERANK_MAX_ITER = 1000

    def _nx_digraph(self, weighted: bool):
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(range(self.n_nodes))
        edges = self._edge_array()
        if weighted:
            g.add_weighted_edges_from(
                zip(edges[:, 0].tolist(), edges[:, 1].tolist(), self.weight.tolist())
            )
        else:
            g.add_edges_from(zip(edges[:, 0].tolist(), edges[:, 1].tolist()))
        return g

    def pagerank(self, damping: float = 0.85, weighted: bool = True) -> np.ndarray:
        import networkx as nx

        if not self.n_edges:
            return np.full(self.n_nodes, 1.0 / self.n_nodes, dtype=np.float64)

        g = self._nx_digraph(weighted)
        scores = nx.pagerank(
            g,
            alpha=damping,
            weight="weight" if weighted else None,
            tol=self.PAGERANK_TOL,
            max_iter=self.PAGERANK_MAX_ITER,
        )
        out = np.zeros(self.n_nodes, dtype=np.float64)
        for node, value in scores.items():
            out[node] = value
        return out

    def communities(self, seed: int) -> np.ndarray:
        import networkx as nx

        if not self.n_edges:
            return np.arange(self.n_nodes, dtype=np.int32)

        edges, weights = self._undirected_edges()
        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))
        for (u, v), w in zip(edges.tolist(), weights.tolist()):
            # Reciprocal pairs collapse onto one key; sum their weights to match the
            # igraph simplify(combine_edges="sum") path exactly.
            if g.has_edge(u, v):
                g[u][v]["weight"] += w
            else:
                g.add_edge(u, v, weight=w)

        communities = nx.community.louvain_communities(g, weight="weight", seed=seed)
        membership = np.zeros(self.n_nodes, dtype=np.int32)
        for cid, members in enumerate(communities):
            for node in members:
                membership[node] = cid
        return membership


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------

_BACKENDS: dict[str, type[_CSRBackend]] = {
    "igraph": IGraphBackend,
    "networkx": NetworkxBackend,
}


def get_backend_class(name: str) -> type[_CSRBackend]:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown graph backend {name!r}; valid: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


def build_backend(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    tx_count: np.ndarray,
    n_nodes: int,
    backend: str = "igraph",
) -> _CSRBackend:
    """Construct the configured backend from a collapsed, ``src``-sorted edge list."""
    cls = get_backend_class(backend)
    obj = cls.from_edges(src, dst, weight, tx_count, n_nodes)
    LOGGER.debug(
        "built %s over %d nodes / %d collapsed edges", backend, obj.n_nodes, obj.n_edges
    )
    return obj
