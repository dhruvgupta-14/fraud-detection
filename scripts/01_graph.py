"""Stage 01 -- canonical transactions to daily snapshot graphs.

    python scripts/01_graph.py [--force] [--verify-backend] [--log-level DEBUG]

Writes to artifacts/snapshots/<config-hash>/:
    day=NN.npz    CSR adjacency + edge attributes for the window ending on day NN
    meta.json     window settings, currency scale factors, per-day summary

Hashed output: the snapshot content depends on the lookback window, the self-loop policy
and the dataset variant, so the E7 lookback sweep produces genuinely different graphs that
must not overwrite each other.

--verify-backend runs the igraph vs networkx equivalence check from architecture.md 6.1 on
a sampled subgraph. It is the Phase 2 checkpoint and is off by default because networkx is
roughly 10x slower.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from aml.config import load_config  # noqa: E402
from aml.graph.snapshots import (  # noqa: E402
    META_FILE,
    Snapshot,
    open_store,
    snapshot_filename,
    window_bounds,
    write_snapshots,
)
from aml.ingest.transactions import TRANSACTIONS_FILE  # noqa: E402
from aml.io import ArtifactStore, setup_logging, timed  # noqa: E402

LOGGER = logging.getLogger("aml.scripts.graph")

# Only the columns the snapshot builder actually reads. The canonical table is 5.08M rows
# and loading all 16 columns to use 6 of them costs ~800MB for no reason.
NEEDED_COLUMNS = [
    "day_idx",
    "src_node",
    "dst_node",
    "amount_paid",
    "currency_paid",
    "timestamp",
    "is_self_loop",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default=None, help="config/experiments/<name>.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild even if snapshots exist")
    parser.add_argument(
        "--verify-backend",
        action="store_true",
        help="run the igraph vs networkx equivalence check (architecture.md 6.1)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    cfg = load_config(args.experiment)
    cfg.seed_everything()
    store = ArtifactStore(cfg)
    stage = open_store(store)

    expected = [snapshot_filename(d) for d in range(cfg.time.max_day + 1)] + [META_FILE]
    if not args.force and all(stage.exists(name) for name in expected):
        LOGGER.info(
            "all %d snapshots present in %s -- nothing to do (use --force to rebuild)",
            cfg.time.max_day + 1,
            stage.dir,
        )
        summary = stage.read_json(META_FILE)["days"]
    else:
        with timed("01_graph"):
            df = _load_transactions(store, cfg)
            n_nodes = int(store.read_processed_json("ingest_report.json")["transactions"]["nodes"])
            stage, summary = write_snapshots(df, n_nodes, cfg, store)

    _print_summary(cfg, summary, stage.dir)
    _check_invariants(cfg, summary)

    if args.verify_backend:
        _verify_backend_equivalence(cfg, stage)

    return 0


def _load_transactions(store: ArtifactStore, cfg):
    """Read the canonical table, truncated to the modelling window.

    Days beyond max_day are the generator tail (architecture.md 2.1) and must not enter a
    snapshot: they carry a 59% illicit rate and would distort the structure that day 9's
    features are read from.
    """
    df = store.read_processed(TRANSACTIONS_FILE, columns=NEEDED_COLUMNS)
    before = len(df)
    df = df.loc[df["day_idx"] <= cfg.time.max_day]
    LOGGER.info(
        "loaded %d rows, kept %d within max_day=%d (dropped %d generator-tail rows)",
        before,
        len(df),
        cfg.time.max_day,
        before - len(df),
    )
    return df


def _check_invariants(cfg, summary: list[dict]) -> None:
    """Fail loudly on the two things that would silently corrupt every structural feature."""
    days = [row["day_idx"] for row in summary]
    if days != list(range(cfg.time.max_day + 1)):
        raise AssertionError(f"snapshot days are not contiguous 0..{cfg.time.max_day}: {days}")

    # Under a cumulative lookback each snapshot is a superset of the previous one, so edge
    # count must be non-decreasing. A drop means the window filter is wrong -- and a wrong
    # window is exactly the failure that produces plausible numbers rather than a crash.
    if cfg.time.lookback_days is None:
        edges = [row["n_edges"] for row in summary]
        if any(b < a for a, b in zip(edges, edges[1:])):
            raise AssertionError(f"cumulative snapshot edge counts are not monotone: {edges}")
        LOGGER.info("invariant OK: cumulative edge counts are monotone non-decreasing")
    else:
        LOGGER.info(
            "lookback=%dd -- edge counts are not expected to be monotone; skipping that check",
            cfg.time.lookback_days,
        )


def _verify_backend_equivalence(cfg, stage) -> None:
    """The Phase 2 checkpoint: igraph and networkx must produce the same answer.

    Run on a sampled subgraph rather than the full snapshot -- networkx at 515K nodes is
    minutes, and the point is to show the adapter is faithful, not to race the libraries.
    Only PageRank and Louvain are checked because they are the only backend-dependent
    quantities (see backend.py).

    **Sampling random nodes does not work here and the first version of this check was
    worthless because of it.** The day-9 graph has 647K edges over 422K active nodes --
    average degree ~1.5 -- so a uniform sample of 4,000 nodes induced *59* edges. Both
    backends trivially agreed on what was essentially an empty graph, and the check would
    have passed no matter how broken the adapter was.

    Sampling **edges** and inducing the subgraph on their endpoints keeps the density of
    the region it came from: 60,000 sampled edges yield ~96K nodes and ~118K induced edges,
    which is a real graph with real community structure to disagree about.
    """
    from aml.graph.backend import build_backend

    snap = Snapshot.load(stage, cfg.time.max_day)
    rng = np.random.default_rng(cfg.seed)

    sources = np.repeat(np.arange(snap.n_nodes, dtype=np.int32), np.diff(snap.indptr))
    n_sample_edges = min(60_000, snap.n_edges)
    seed_edges = rng.choice(snap.n_edges, size=n_sample_edges, replace=False)
    sample = np.unique(np.concatenate([sources[seed_edges], snap.indices[seed_edges]]))

    keep = np.zeros(snap.n_nodes, dtype=bool)
    keep[sample] = True
    mask = keep[sources] & keep[snap.indices]
    remap = np.full(snap.n_nodes, -1, dtype=np.int32)
    remap[sample] = np.arange(len(sample), dtype=np.int32)

    src = remap[sources[mask]]
    dst = remap[snap.indices[mask]]
    weight = snap.weights("amount_norm")[mask]
    counts = snap.tx_count[mask]
    order = np.argsort(src.astype(np.int64) * len(sample) + dst, kind="stable")

    LOGGER.info(
        "backend equivalence check on %d sampled nodes / %d edges", len(sample), int(mask.sum())
    )
    results = {}
    for name in ("igraph", "networkx"):
        g = build_backend(
            src[order], dst[order], weight[order], counts[order], len(sample), backend=name
        )
        results[name] = (g.pagerank(cfg.graph.pagerank_damping), g.communities(cfg.seed))

    pr_a, comm_a = results["igraph"]
    pr_b, comm_b = results["networkx"]
    max_diff = float(np.abs(pr_a - pr_b).max())
    agree = _partition_agreement(comm_a, comm_b)

    print("\n" + "=" * 72)
    print("BACKEND EQUIVALENCE -- architecture.md 6.1")
    print("=" * 72)
    print(f"  sampled nodes / edges      {len(sample):,} / {int(mask.sum()):,}")
    print(f"  PageRank max abs diff      {max_diff:.3e}   (PRPACK exact vs nx power iteration)")
    print(f"  PageRank rank correlation  {np.corrcoef(pr_a.argsort(), pr_b.argsort())[0, 1]:.6f}")
    print(f"  Louvain pairwise agreement {agree:.4f}")
    print("=" * 72 + "\n")

    if max_diff > 1e-4:
        raise AssertionError(f"PageRank backends disagree by {max_diff:.3e} -- adapter bug")


def _partition_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of sampled node pairs the two partitions place the same way.

    Community *ids* are arbitrary, so labels cannot be compared directly; what must agree
    is whether two nodes land together.
    """
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(2, 20000))
    same_a = a[idx[0]] == a[idx[1]]
    same_b = b[idx[0]] == b[idx[1]]
    return float((same_a == same_b).mean())


def _print_summary(cfg, summary: list[dict], path: Path) -> None:
    lookback = "cumulative" if cfg.time.lookback_days is None else f"{cfg.time.lookback_days}d"
    print("\n" + "=" * 72)
    print("SNAPSHOT SUMMARY -- architecture.md 6.2")
    print("=" * 72)
    print(f"  lookback            {lookback}")
    print(f"  self-loops          {'excluded from edges' if cfg.graph.exclude_self_loops else 'INCLUDED'}")
    print(f"  written to          {path}")
    print("-" * 72)
    print(f"  {'day':>4}  {'window':>9}  {'rows':>11}  {'edges':>10}  {'active':>9}  {'rows/edge':>9}")
    for row in summary:
        lo, hi = window_bounds(row["day_idx"], cfg.time.lookback_days)
        print(
            f"  {row['day_idx']:>4}  {f'{lo}-{hi}':>9}  {row['n_rows']:>11,}  "
            f"{row['n_edges']:>10,}  {row['n_active_nodes']:>9,}  {row['collapse_ratio']:>9.2f}"
        )
    print("=" * 72)
    print("  Note: a transaction on day D reads structural features from snapshot D-1.")
    print(f"        Day 0 has no prior snapshot and is cold-start (architecture.md 6.2).")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
