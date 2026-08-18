"""Guards on the graph backends (architecture.md section 6.1, component 9).

All fixtures are synthetic and small enough that every expected number is derived by hand
in the test itself rather than snapshotted from a previous run. A test that asserts
"whatever it printed last time" would pass just as happily after a sign error.

Two things are actually being guarded here:

1. **The CSR arithmetic is exact.** Degree, strength and transaction count are the inputs
   to every structural feature; if they are quietly wrong, nothing crashes and the
   ablation still produces a number.
2. **igraph and networkx agree.** This is the Phase 2 checkpoint from section 6.1, in
   miniature. If the two libraries disagree, our adapter is wrong -- the answer is not to
   keep whichever number looks better.
"""

from __future__ import annotations

import numpy as np
import pytest

from aml.graph.backend import (
    IN,
    OUT,
    GraphBackend,
    IGraphBackend,
    NetworkxBackend,
    build_backend,
    get_backend_class,
)

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

# Two directed triangles joined by a single weak bridge, plus one isolated node.
#
#     0 -> 1 -> 2 -> 0        (triangle A, weight 10 each)
#                2 -> 3       (bridge, weight 1)
#     3 -> 4 -> 5 -> 3        (triangle B, weight 10 each)
#     6                       (isolated -- exercises the dormant-node path)
#
# Chosen so the community structure is unambiguous by eye: any correct Louvain must put
# {0,1,2} and {3,4,5} in different communities, because the bridge is 10x weaker than
# every intra-triangle edge.
N_NODES = 7
SRC = np.array([0, 1, 2, 2, 3, 4, 5], dtype=np.int32)
DST = np.array([1, 2, 0, 3, 4, 5, 3], dtype=np.int32)
WEIGHT = np.array([10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0], dtype=np.float64)
TX_COUNT = np.array([1, 2, 1, 1, 3, 1, 1], dtype=np.int64)


@pytest.fixture
def ig() -> IGraphBackend:
    return build_backend(SRC, DST, WEIGHT, TX_COUNT, N_NODES, backend="igraph")


@pytest.fixture
def nx_backend() -> NetworkxBackend:
    return build_backend(SRC, DST, WEIGHT, TX_COUNT, N_NODES, backend="networkx")


@pytest.fixture(params=["igraph", "networkx"])
def any_backend(request):
    """Both backends, so shared behaviour is asserted against both implementations."""
    return build_backend(SRC, DST, WEIGHT, TX_COUNT, N_NODES, backend=request.param)


def empty_backend(name: str = "igraph"):
    """A node space with no edges at all -- the day-0-under-lookback case."""
    return build_backend(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64),
        N_NODES,
        backend=name,
    )


# --------------------------------------------------------------------------------------
# Protocol and selection
# --------------------------------------------------------------------------------------


def test_both_backends_satisfy_the_protocol(any_backend):
    assert isinstance(any_backend, GraphBackend)


def test_unknown_backend_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown graph backend"):
        get_backend_class("graph-tool")


def test_backend_reports_its_shape(any_backend):
    assert any_backend.n_nodes == N_NODES
    assert any_backend.n_edges == len(SRC)


# --------------------------------------------------------------------------------------
# Exact CSR arithmetic -- expected values derived by hand from the fixture above
# --------------------------------------------------------------------------------------


def test_degrees_count_distinct_counterparties(any_backend):
    in_deg, out_deg = any_backend.degrees()
    # node 2 sends to {0, 3}; node 3 receives from {2, 5}; node 6 is isolated.
    assert out_deg.tolist() == [1, 1, 2, 1, 1, 1, 0]
    assert in_deg.tolist() == [1, 1, 1, 2, 1, 1, 0]


def test_strengths_sum_edge_amounts(any_backend):
    in_str, out_str = any_backend.strengths()
    # node 2 sends 10 (to 0) + 1 (bridge) = 11; node 3 receives 1 (bridge) + 10 (from 5).
    assert out_str.tolist() == [10.0, 10.0, 11.0, 10.0, 10.0, 10.0, 0.0]
    assert in_str.tolist() == [10.0, 10.0, 10.0, 11.0, 10.0, 10.0, 0.0]


def test_tx_counts_expand_collapsed_parallel_edges(any_backend):
    in_tx, out_tx = any_backend.tx_counts()
    # Distinct from degree: edge 1->2 carries 2 transactions and 3->4 carries 3.
    assert out_tx.tolist() == [1, 2, 2, 3, 1, 1, 0]
    assert in_tx.tolist() == [1, 1, 2, 2, 3, 1, 0]


def test_degree_and_tx_count_are_not_the_same_quantity(any_backend):
    """The R10 ambiguity, pinned as a test.

    architecture.md section 2.1 quotes a max out-degree and an edge share that cannot both
    describe the same measurement. They are different quantities and the backend keeps
    them apart; this asserts they are genuinely distinguishable on a fixture where a
    parallel edge exists.
    """
    _, out_deg = any_backend.degrees()
    _, out_tx = any_backend.tx_counts()
    assert out_deg[1] == 1 and out_tx[1] == 2  # one counterparty, two transactions
    assert out_deg.tolist() != out_tx.tolist()


def test_neighbors_respect_direction(any_backend):
    assert sorted(any_backend.neighbors(2, OUT).tolist()) == [0, 3]
    assert sorted(any_backend.neighbors(3, IN).tolist()) == [2, 5]
    assert any_backend.neighbors(6, OUT).tolist() == []
    assert any_backend.neighbors(6, IN).tolist() == []


def test_neighbors_rejects_bad_direction_and_node(any_backend):
    with pytest.raises(ValueError, match="direction"):
        any_backend.neighbors(0, "sideways")
    with pytest.raises(IndexError):
        any_backend.neighbors(N_NODES, OUT)


def test_active_mask_excludes_dormant_nodes(any_backend):
    assert any_backend.active_mask().tolist() == [True] * 6 + [False]


# --------------------------------------------------------------------------------------
# Input validation -- these failures must be loud
# --------------------------------------------------------------------------------------


def test_self_loops_are_rejected_rather_than_filtered():
    """A self-loop reaching the backend means the snapshot builder is broken.

    Silently dropping it here would hide that bug while inflating degree on exactly the
    11.6% of rows that carry no laundering signal (architecture.md section 2.1).
    """
    with pytest.raises(ValueError, match="self-loop"):
        build_backend(
            np.array([0, 1], dtype=np.int32),
            np.array([1, 1], dtype=np.int32),  # 1 -> 1
            np.array([1.0, 1.0]),
            np.array([1, 1], dtype=np.int64),
            N_NODES,
        )


def test_indices_outside_the_node_space_are_rejected():
    with pytest.raises(ValueError, match="out of range"):
        build_backend(
            np.array([0], dtype=np.int32),
            np.array([99], dtype=np.int32),
            np.array([1.0]),
            np.array([1], dtype=np.int64),
            N_NODES,
        )


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ValueError, match="indptr implies"):
        IGraphBackend(
            indptr=np.array([0, 1, 2]),
            indices=np.array([1, 0]),
            weight=np.array([1.0]),  # too short
            tx_count=np.array([1, 1]),
            n_nodes=2,
        )


# --------------------------------------------------------------------------------------
# PageRank
# --------------------------------------------------------------------------------------


def test_pagerank_is_a_distribution(any_backend):
    scores = any_backend.pagerank(damping=0.85, weighted=True)
    assert scores.shape == (N_NODES,)
    assert scores.min() > 0.0
    assert pytest.approx(1.0, abs=1e-9) == scores.sum()


def test_pagerank_on_an_edgeless_graph_is_uniform():
    scores = empty_backend().pagerank()
    assert np.allclose(scores, 1.0 / N_NODES)


def test_pagerank_concentrates_on_the_bridge_target(ig):
    """Node 3 is the only node fed by two others, so it must outrank its own triangle."""
    scores = ig.pagerank(damping=0.85, weighted=True)
    assert scores[3] > scores[4]
    assert scores[3] > scores[5]


def test_weighted_and_unweighted_pagerank_differ(ig):
    """If the weights argument were being dropped, these would be identical."""
    weighted = ig.pagerank(weighted=True)
    unweighted = ig.pagerank(weighted=False)
    assert not np.allclose(weighted, unweighted)


def test_pagerank_is_deterministic_across_calls(ig):
    assert np.array_equal(ig.pagerank(), ig.pagerank())


# --------------------------------------------------------------------------------------
# Communities
# --------------------------------------------------------------------------------------


def _partition(membership: np.ndarray) -> set[frozenset[int]]:
    """Community ids are arbitrary labels; compare the induced partition instead."""
    groups: dict[int, set[int]] = {}
    for node, cid in enumerate(membership.tolist()):
        groups.setdefault(cid, set()).add(node)
    return {frozenset(g) for g in groups.values()}


def test_communities_separate_the_two_triangles(any_backend):
    membership = any_backend.communities(seed=42)
    assert membership.shape == (N_NODES,)
    assert membership[0] == membership[1] == membership[2]
    assert membership[3] == membership[4] == membership[5]
    assert membership[0] != membership[3]


def test_communities_are_reproducible_under_the_same_seed(any_backend):
    first = any_backend.communities(seed=42)
    second = any_backend.communities(seed=42)
    assert np.array_equal(first, second)


def test_communities_on_an_edgeless_graph_are_all_singletons():
    membership = empty_backend().communities(seed=42)
    assert len(set(membership.tolist())) == N_NODES


def test_isolated_node_is_not_absorbed_into_a_community(any_backend):
    """Node 6 has no edges; it must not be silently attached to a real community."""
    membership = any_backend.communities(seed=42)
    assert membership[6] not in membership[:6].tolist()


# --------------------------------------------------------------------------------------
# The Phase 2 checkpoint: igraph and networkx must agree
# --------------------------------------------------------------------------------------


def test_pagerank_agrees_between_backends(ig, nx_backend):
    """PRPACK solves the system exactly; networkx iterates to a tolerance.

    They therefore agree to roughly that tolerance, not to machine precision -- so this
    compares with an absolute tolerance rather than exactly. A real adapter bug (dropped
    weights, wrong edge direction) moves the scores far more than 1e-5.
    """
    a = ig.pagerank(damping=0.85, weighted=True)
    b = nx_backend.pagerank(damping=0.85, weighted=True)
    assert np.allclose(a, b, atol=1e-5)
    assert np.array_equal(np.argsort(a), np.argsort(b))


def test_communities_agree_between_backends(ig, nx_backend):
    assert _partition(ig.communities(seed=42)) == _partition(nx_backend.communities(seed=42))


def test_exact_quantities_are_identical_between_backends(ig, nx_backend):
    """These are shared NumPy code, so equality here is a guard against the shared base
    class being accidentally overridden in one subclass and not the other."""
    assert np.array_equal(ig.degrees()[0], nx_backend.degrees()[0])
    assert np.array_equal(ig.degrees()[1], nx_backend.degrees()[1])
    assert np.array_equal(ig.strengths()[0], nx_backend.strengths()[0])
    assert np.array_equal(ig.tx_counts()[1], nx_backend.tx_counts()[1])
