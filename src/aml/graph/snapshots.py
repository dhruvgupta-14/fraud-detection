"""Daily snapshot graphs -- the mechanism behind the temporal contract (§6.2, §11.2).

A snapshot is the transaction graph as it stood at the end of one day. Structural features
for a transaction on day ``D`` are read from the snapshot built at ``D - 1``, so a
transaction can never contribute to its own features. That lag is enforced by the feature
layer; this module's job is to make the ``D - 1`` object exist, be cheap to rebuild, and be
honest about what is in it.

What a snapshot is
------------------
Parallel edges are **collapsed**: at most one entry per ordered ``(src, dst)`` pair. This
is not lossy in the way it sounds -- the number of transactions between the pair, their
total value and the timestamp of the most recent one all survive as edge attributes. The
collapse is what makes the graph tractable: 4,487,133 non-self-loop rows reduce to 647,939
distinct pairs (§2.1), a 6.9x reduction, because the same accounts transact repeatedly.

Self-loops are excluded from edges but their rows remain scoreable transactions -- "the
graph view and the scoring view are not the same row set" (§5).

Three weights, and why
----------------------
The snapshot stores **three** candidate edge weights rather than picking one, because the
right choice is a feature-layer question (Phase 5) and picking early would bake in an
error that is hard to see later:

* ``weight_amount``      -- raw summed ``amount_paid``. What §6.2 originally specified.
* ``weight_amount_norm`` -- each amount divided by the median amount of its own payment
  currency before summing.
* ``tx_count``           -- how many transactions ran along the pair.

``weight_amount`` has a **unit problem that is easy to miss and severe when missed**. The
dataset carries 15 payment currencies with no FX table, and their median transaction sizes
span six orders of magnitude: Bitcoin 0.07, US Dollar 877, Rupee 65,887, Yen 97,439. Summing
them as though they were one unit -- which is exactly what an amount-weighted PageRank does
-- means a single Yen transaction carries more mass than roughly 1.4 million Bitcoin ones.
The resulting centrality would rank currencies, not accounts.

``weight_amount_norm`` rescales each amount into "typical transaction for its currency"
units, which is dimensionally coherent and label-free. It is a crude stand-in for an FX
conversion and is documented as such; the per-currency medians are computed once over the
modelling window and persisted in the snapshot metadata so the transform is reproducible
and inspectable rather than implicit.

``tx_count`` sidesteps the question entirely and answers a different one -- relationship
intensity rather than value flow.

All three are stored, none is privileged here, and the decision is recorded in the report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from aml.config import Config
from aml.graph.backend import _CSRBackend, get_backend_class
from aml.io import ArtifactStore, StageStore, timed

LOGGER = logging.getLogger("aml.graph.snapshots")

SNAPSHOT_STAGE = "snapshots"
# Snapshot content depends on which rows enter the window (time), whether self-loops are
# dropped (graph), and which dataset variant is loaded. It does NOT depend on the seed --
# nothing here is stochastic; Louvain runs later, at feature time.
SNAPSHOT_SECTIONS = ("dataset", "time", "graph")
META_FILE = "meta.json"

WEIGHT_KINDS = ("amount_norm", "amount", "tx_count", "none")


def snapshot_filename(day_idx: int) -> str:
    return f"day={day_idx:02d}.npz"


# --------------------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """One day's graph, in CSR form over the **full** node space.

    Spanning all interned accounts rather than only the active ones means every returned
    array is indexed directly by ``node_id`` with no remap table. ``active_mask`` records
    which nodes actually participated, so the feature layer can null dormant accounts
    rather than joining them a meaningless teleport-only PageRank (see backend.py).
    """

    day_idx: int
    lookback_days: int | None
    n_nodes: int
    n_rows: int  # source transaction rows in the window, before collapsing

    indptr: np.ndarray  # int64, length n_nodes + 1
    indices: np.ndarray  # int32, destination node per collapsed edge
    weight_amount: np.ndarray  # float64, summed amount_paid (mixed currencies -- see module docstring)
    weight_amount_norm: np.ndarray  # float64, summed currency-normalised amount
    tx_count: np.ndarray  # int64, transactions per collapsed edge
    last_ts: np.ndarray  # int64, epoch seconds of the most recent transaction on the pair

    @property
    def n_edges(self) -> int:
        return int(self.indptr[-1])

    @property
    def n_active(self) -> int:
        return int(self.active_mask().sum())

    def active_mask(self) -> np.ndarray:
        out_deg = np.diff(self.indptr)
        in_deg = np.bincount(self.indices, minlength=self.n_nodes)
        return (out_deg > 0) | (in_deg > 0)

    def weights(self, kind: str = "amount_norm") -> np.ndarray:
        """Select one of the three stored edge weights.

        ``none`` returns unit weights, i.e. structure only. See the module docstring for
        why this is a real choice and not a formatting preference.
        """
        if kind not in WEIGHT_KINDS:
            raise ValueError(f"weight kind must be one of {WEIGHT_KINDS}, got {kind!r}")
        if kind == "amount":
            return self.weight_amount
        if kind == "amount_norm":
            return self.weight_amount_norm
        if kind == "tx_count":
            return self.tx_count.astype(np.float64)
        return np.ones(self.n_edges, dtype=np.float64)

    def backend(self, name: str = "igraph", weight: str = "amount_norm") -> _CSRBackend:
        """Wrap this snapshot in a compute backend."""
        cls = get_backend_class(name)
        return cls(
            indptr=self.indptr,
            indices=self.indices,
            weight=self.weights(weight),
            tx_count=self.tx_count,
            n_nodes=self.n_nodes,
        )

    # ------------------------------------------------------------------ persistence

    def save(self, store: StageStore) -> None:
        store.write_npz(
            snapshot_filename(self.day_idx),
            indptr=self.indptr,
            indices=self.indices,
            weight_amount=self.weight_amount,
            weight_amount_norm=self.weight_amount_norm,
            tx_count=self.tx_count,
            last_ts=self.last_ts,
            scalars=np.array(
                [
                    self.day_idx,
                    -1 if self.lookback_days is None else self.lookback_days,
                    self.n_nodes,
                    self.n_rows,
                ],
                dtype=np.int64,
            ),
        )

    @classmethod
    def load(cls, store: StageStore, day_idx: int) -> "Snapshot":
        data = store.read_npz(snapshot_filename(day_idx))
        day, lookback, n_nodes, n_rows = data["scalars"].tolist()
        return cls(
            day_idx=int(day),
            lookback_days=None if lookback < 0 else int(lookback),
            n_nodes=int(n_nodes),
            n_rows=int(n_rows),
            indptr=data["indptr"],
            indices=data["indices"],
            weight_amount=data["weight_amount"],
            weight_amount_norm=data["weight_amount_norm"],
            tx_count=data["tx_count"],
            last_ts=data["last_ts"],
        )


# --------------------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------------------


def window_bounds(day_idx: int, lookback_days: int | None) -> tuple[int, int]:
    """Inclusive ``(first_day, last_day)`` of the snapshot window.

    **Deviation from §6.2, deliberate.** The spec writes the edge set as
    ``(day_idx - lookback_days) <= d <= day_idx``, which spans ``lookback_days + 1`` days --
    so a config saying ``lookback_days: 3`` would actually build a four-day window. That
    off-by-one would silently mislabel every point in the E7 lookback sweep, which is a
    published result, so the window is defined here as the ``lookback_days`` **most recent
    days inclusive**: ``day_idx - lookback_days + 1 <= d <= day_idx``.

    ``None`` means cumulative -- all history up to and including ``day_idx``.
    """
    if lookback_days is None:
        return 0, day_idx
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1 or None, got {lookback_days}")
    return max(0, day_idx - lookback_days + 1), day_idx


# --------------------------------------------------------------------------------------
# Currency normalisation
# --------------------------------------------------------------------------------------


def currency_medians(df: pd.DataFrame) -> dict[str, float]:
    """Median ``amount_paid`` per payment currency, over the rows given.

    Used to rescale amounts into per-currency "typical transaction" units. This is a
    **unit conversion, not a learned statistic**: it reads no labels and no graph
    structure, so it does not belong to any causality class in §11.1 and is computed once
    over the whole modelling window rather than per snapshot -- a weight that changed
    between days would make cross-day centrality incomparable, which is worse than the
    mild look-ahead of a stable scale factor.

    It is persisted with the snapshots so the transform is inspectable rather than implied.
    """
    medians = (
        df.groupby("currency_paid", observed=True)["amount_paid"].median().to_dict()
    )
    return {str(k): float(v) for k, v in medians.items() if v and v > 0}


def _normalised_amounts(df: pd.DataFrame, medians: dict[str, float]) -> np.ndarray:
    scale = df["currency_paid"].astype(str).map(medians).to_numpy(dtype=np.float64)
    # A currency absent from the median table (possible only on a filtered subset) falls
    # back to 1.0, i.e. no rescale, rather than producing NaN weights.
    np.nan_to_num(scale, copy=False, nan=1.0)
    scale[scale <= 0] = 1.0
    return df["amount_paid"].to_numpy(dtype=np.float64) / scale


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


def collapse_edges(
    src: np.ndarray,
    dst: np.ndarray,
    amount: np.ndarray,
    amount_norm: np.ndarray,
    ts: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, ...]:
    """Collapse parallel edges into one entry per ordered ``(src, dst)`` pair.

    Returns CSR-ordered arrays. The packed key ``src * n_nodes + dst`` sorts by source then
    destination, which *is* CSR order, so no second sort is needed after ``np.unique``.
    """
    if len(src) == 0:
        return (
            np.zeros(n_nodes + 1, dtype=np.int64),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    key = src.astype(np.int64) * n_nodes + dst.astype(np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    n_edges = len(uniq)

    w_amount = np.bincount(inverse, weights=amount, minlength=n_edges)
    w_norm = np.bincount(inverse, weights=amount_norm, minlength=n_edges)
    counts = np.bincount(inverse, minlength=n_edges).astype(np.int64)

    # Most recent timestamp per pair, without ufunc.at (which is slow at this scale):
    # write in ascending-timestamp order so the final write into each slot is the maximum.
    last_ts = np.zeros(n_edges, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    last_ts[inverse[order]] = ts[order]

    src_u = (uniq // n_nodes).astype(np.int64)
    indices = (uniq % n_nodes).astype(np.int32)

    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src_u, minlength=n_nodes), out=indptr[1:])

    return indptr, indices, w_amount, w_norm, counts, last_ts


def build_snapshot(
    df: pd.DataFrame,
    day_idx: int,
    n_nodes: int,
    cfg: Config,
    medians: dict[str, float] | None = None,
) -> Snapshot:
    """Build the snapshot for ``day_idx`` from the canonical transaction frame.

    ``df`` must already be restricted to the modelling window and, if
    ``cfg.graph.exclude_self_loops``, to non-self-loop rows -- the caller does that once
    rather than once per day.
    """
    first_day, last_day = window_bounds(day_idx, cfg.time.lookback_days)
    days = df["day_idx"].to_numpy()
    mask = (days >= first_day) & (days <= last_day)
    window = df.loc[mask]

    medians = medians if medians is not None else currency_medians(window)
    ts = window["timestamp"].to_numpy("datetime64[s]").astype(np.int64)

    indptr, indices, w_amount, w_norm, counts, last_ts = collapse_edges(
        window["src_node"].to_numpy(),
        window["dst_node"].to_numpy(),
        window["amount_paid"].to_numpy(dtype=np.float64),
        _normalised_amounts(window, medians),
        ts,
        n_nodes,
    )

    return Snapshot(
        day_idx=day_idx,
        lookback_days=cfg.time.lookback_days,
        n_nodes=n_nodes,
        n_rows=int(mask.sum()),
        indptr=indptr,
        indices=indices,
        weight_amount=w_amount,
        weight_amount_norm=w_norm,
        tx_count=counts,
        last_ts=last_ts,
    )


def build_all_snapshots(
    df: pd.DataFrame, n_nodes: int, cfg: Config
) -> Iterator[Snapshot]:
    """Yield one snapshot per day in the modelling window, day 0 through ``max_day``.

    Day ``max_day``'s snapshot serves no transaction under the ``D - 1`` rule -- day
    ``max_day + 1`` is the generator tail and is excluded. It is built anyway because it is
    the graph over the full usable window, which is what the exploration notebook plots
    degree distributions from. The feature layer's A1 assertion is what stops it from
    accidentally being joined.
    """
    graph_rows = df
    if cfg.graph.exclude_self_loops:
        graph_rows = df.loc[~df["is_self_loop"]]
        LOGGER.info(
            "excluded %d self-loop rows from edges (%.1f%%); they remain scoreable rows",
            len(df) - len(graph_rows),
            100.0 * (len(df) - len(graph_rows)) / max(len(df), 1),
        )

    medians = currency_medians(graph_rows)
    LOGGER.info(
        "currency scale factors over %d currencies: min median %.4f, max median %.2f",
        len(medians),
        min(medians.values()),
        max(medians.values()),
    )

    for day_idx in range(cfg.time.max_day + 1):
        with timed(f"snapshot day={day_idx:02d}", LOGGER):
            yield build_snapshot(graph_rows, day_idx, n_nodes, cfg, medians=medians)


# --------------------------------------------------------------------------------------
# Stage orchestration
# --------------------------------------------------------------------------------------


def open_store(store: ArtifactStore) -> StageStore:
    return store.stage(SNAPSHOT_STAGE, SNAPSHOT_SECTIONS)


def write_snapshots(
    df: pd.DataFrame, n_nodes: int, cfg: Config, store: ArtifactStore
) -> tuple[StageStore, list[dict]]:
    """Build and persist every snapshot, returning the stage store and a summary table."""
    stage = open_store(store)
    summary: list[dict] = []

    for snap in build_all_snapshots(df, n_nodes, cfg):
        snap.save(stage)
        summary.append(
            {
                "day_idx": snap.day_idx,
                "n_rows": snap.n_rows,
                "n_edges": snap.n_edges,
                "n_active_nodes": snap.n_active,
                "collapse_ratio": round(snap.n_rows / max(snap.n_edges, 1), 2),
            }
        )

    graph_rows = df.loc[~df["is_self_loop"]] if cfg.graph.exclude_self_loops else df
    stage.write_json(
        {
            "lookback_days": cfg.time.lookback_days,
            "max_day": cfg.time.max_day,
            "n_nodes": n_nodes,
            "exclude_self_loops": cfg.graph.exclude_self_loops,
            "currency_medians": currency_medians(graph_rows),
            "days": summary,
        },
        META_FILE,
    )
    return stage, summary


def load_snapshot(cfg: Config, store: ArtifactStore, day_idx: int) -> Snapshot:
    """Read one persisted snapshot. Raises if `01_graph.py` has not been run."""
    return Snapshot.load(open_store(store), day_idx)
