"""Block B -- causal account-state counters (architecture.md 7.2).

One pass over the transaction table in timestamp order, maintaining per-account state.
**Every row reads state before contributing to it.** That ordering is the entire leakage
argument for this block, and it is why the work is done in a single explicit loop rather
than as a set of vectorised ``groupby().cumsum().shift()`` expressions.

The vectorised form would be roughly ten times faster and would also be correct -- but the
guarantee would then be distributed across a dozen separate shift operations, each of which
has to be individually right, and three of these features (``turnover_latency``,
``distinct_counterparties``, the ``unique_*_seen`` masks) do not vectorise cleanly anyway.
Splitting the causality argument across two mechanisms to save seconds inside a 25-minute
budget is a bad trade for a project whose headline claim is that its features do not leak.
Here the read-then-update order is visible in one place and can be checked by eye.

Every column is emitted for **both** endpoints, prefixed ``src_`` and ``dst_``: what matters
about a transaction is the state of the account sending it *and* the account receiving it.

Causality class for all 32 columns: ``causal_streaming``.
"""

from __future__ import annotations

import logging
from math import sqrt

import numpy as np
import pandas as pd

from aml.features.base import Causality, FeatureBlock, FeatureContext, FeatureSpec, register

LOGGER = logging.getLogger("aml.features.streaming")

STREAMING = Causality.CAUSAL_STREAMING

# One entry per emitted quantity: (suffix, null policy, description). Each is produced twice,
# once for the sender and once for the receiver, so the emitted column count is 2x this.
_QUANTITIES: list[tuple[str, str, str]] = [
    ("tx_count_in", "never", "transactions this account had received before this one"),
    ("tx_count_out", "never", "transactions this account had sent before this one"),
    ("volume_in", "never", "total amount received before this transaction"),
    ("volume_out", "never", "total amount sent before this transaction"),
    ("inout_volume_ratio", "cold_start", "volume_in / volume_out -- the pass-through signature"),
    ("secs_since_last_in", "cold_start", "seconds since this account last received"),
    ("secs_since_last_out", "cold_start", "seconds since this account last sent"),
    ("turnover_latency", "cold_start", "the account's most recent receive-to-send gap, in seconds"),
    ("distinct_counterparties_in", "never", "distinct senders seen before, capped"),
    ("distinct_counterparties_out", "never", "distinct receivers seen before, capped"),
    ("unique_currencies_seen", "never", "distinct payment currencies seen before"),
    ("unique_formats_seen", "never", "distinct payment rails seen before"),
    ("mean_amount_in", "cold_start", "mean amount received before this transaction"),
    ("mean_amount_out", "cold_start", "mean amount sent before this transaction"),
    ("amount_zscore_vs_own_history", "cold_start", "this amount against the account's own prior amounts"),
    ("account_age_secs", "never", "seconds since this account was first observed"),
]

_NAN = float("nan")


@register
class StreamingBlock:
    name = "streaming"
    group = "streaming"
    requires_snapshot = False

    def columns(self) -> list[FeatureSpec]:
        return [
            FeatureSpec(f"{side}_{suffix}", STREAMING, f"{label} ({side})", null_policy=policy)
            for suffix, policy, label in _QUANTITIES
            for side in ("src", "dst")
        ]

    def compute(self, ctx: FeatureContext) -> pd.DataFrame:
        df = ctx.transactions
        _assert_time_ordered(df)

        n_nodes = ctx.n_nodes
        cap = int(ctx.cfg.features.streaming["max_tracked_counterparties"])
        names = [spec.name for spec in self.columns()]

        src = df["src_node"].to_numpy(dtype=np.int64)
        dst = df["dst_node"].to_numpy(dtype=np.int64)
        amount = df["amount_paid"].to_numpy(dtype=np.float64)
        ts = df["timestamp"].to_numpy("datetime64[s]").astype(np.int64)
        currency = _codes(df["currency_paid"])
        fmt = _codes(df["payment_format"])

        LOGGER.info("streaming pass over %d rows / %d accounts (cap=%d)", len(df), n_nodes, cap)
        values = _run_pass(src, dst, amount, ts, currency, fmt, n_nodes, cap)

        return pd.DataFrame(values, columns=names, index=df.index)


# --------------------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------------------


def _run_pass(
    src: np.ndarray,
    dst: np.ndarray,
    amount: np.ndarray,
    ts: np.ndarray,
    currency: np.ndarray,
    fmt: np.ndarray,
    n_nodes: int,
    cap: int,
) -> np.ndarray:
    """Walk the table once, emitting pre-update state for both endpoints of every row."""
    n_rows = len(src)
    out = np.empty((n_rows, 2 * len(_QUANTITIES)), dtype=np.float32)

    # Per-account state as plain Python lists indexed by node id. The interner exists so
    # this can be a positional lookup rather than a dict of string keys.
    #
    # Lists rather than numpy arrays deliberately: this loop does 5.08M x ~40 *scalar*
    # accesses, and every numpy scalar read boxes a np.float64/np.int64 object. Lists hold
    # native Python numbers and are roughly 2-3x faster here. numpy wins on vectorised
    # work; this is the opposite case.
    n_in = [0] * n_nodes
    n_out = [0] * n_nodes
    vol_in = [0.0] * n_nodes
    vol_out = [0.0] * n_nodes
    last_in = [-1] * n_nodes
    last_out = [-1] * n_nodes
    turnover = [_NAN] * n_nodes
    first_ts = [-1] * n_nodes

    # 15 currencies and 7 formats both fit in an integer bitmask, so "how many distinct
    # values has this account used" is a popcount rather than a per-account set. These must
    # be Python ints: int.bit_count does not accept a numpy scalar.
    cur_mask = [0] * n_nodes
    fmt_mask = [0] * n_nodes

    # Running moments, for the z-score of an amount against the account's own history.
    amt_n = [0] * n_nodes
    amt_sum = [0.0] * n_nodes
    amt_sumsq = [0.0] * n_nodes

    # Counterparty identity genuinely needs sets. Created lazily: most accounts have a
    # handful of counterparties, and 515K pre-allocated empty sets would cost more than the
    # contents. Capped per config -- beyond the cap the count is a lower bound, which is
    # the documented behaviour rather than a silent truncation to a wrong number.
    cp_in: dict[int, set] = {}
    cp_out: dict[int, set] = {}

    # Iterate over Python lists rather than numpy arrays for the same boxing reason.
    src_l, dst_l = src.tolist(), dst.tolist()
    amount_l, ts_l = amount.tolist(), ts.tolist()
    currency_l, fmt_l = currency.tolist(), fmt.tolist()

    for i in range(n_rows):
        s = src_l[i]
        d = dst_l[i]
        a = amount_l[i]
        t = ts_l[i]

        for col, node in ((0, s), (1, d)):
            base = col
            ni, no = n_in[node], n_out[node]
            vi, vo = vol_in[node], vol_out[node]
            li, lo = last_in[node], last_out[node]
            an = amt_n[node]
            first = first_ts[node]

            out[i, base + 0] = ni
            out[i, base + 2] = no
            out[i, base + 4] = vi
            out[i, base + 6] = vo
            out[i, base + 8] = (vi / vo) if vo > 0 else _NAN
            out[i, base + 10] = (t - li) if li >= 0 else _NAN
            out[i, base + 12] = (t - lo) if lo >= 0 else _NAN
            out[i, base + 14] = turnover[node]
            out[i, base + 16] = len(cp_in.get(node, ()))
            out[i, base + 18] = len(cp_out.get(node, ()))
            out[i, base + 20] = cur_mask[node].bit_count()
            out[i, base + 22] = fmt_mask[node].bit_count()
            out[i, base + 24] = (vi / ni) if ni else _NAN
            out[i, base + 26] = (vo / no) if no else _NAN

            if an >= 2:
                mean = amt_sum[node] / an
                var = amt_sumsq[node] / an - mean * mean
                out[i, base + 28] = (a - mean) / sqrt(var) if var > 1e-12 else 0.0
            else:
                out[i, base + 28] = _NAN
            out[i, base + 30] = (t - first) if first >= 0 else 0.0

        # ---- state update, strictly after both reads ----
        if first_ts[s] < 0:
            first_ts[s] = t
        if first_ts[d] < 0:
            first_ts[d] = t

        # The sender is turning money over: record the gap since it last received. This is
        # what makes turnover_latency distinct from secs_since_last_in -- the emitted value
        # is the account's *previous* measured turnover, not the current one.
        if last_in[s] >= 0:
            turnover[s] = t - last_in[s]

        n_out[s] += 1
        vol_out[s] += a
        last_out[s] = t
        n_in[d] += 1
        vol_in[d] += a
        last_in[d] = t

        outs = cp_out.setdefault(s, set())
        if len(outs) < cap:
            outs.add(d)
        ins = cp_in.setdefault(d, set())
        if len(ins) < cap:
            ins.add(s)

        cur_mask[s] |= 1 << currency[i]
        cur_mask[d] |= 1 << currency[i]
        fmt_mask[s] |= 1 << fmt[i]
        fmt_mask[d] |= 1 << fmt[i]

        amt_n[s] += 1
        amt_sum[s] += a
        amt_sumsq[s] += a * a
        amt_n[d] += 1
        amt_sum[d] += a
        amt_sumsq[d] += a * a

    return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _assert_time_ordered(df: pd.DataFrame) -> None:
    """The whole block is meaningless if the rows are not in timestamp order.

    Out of order, an account's "prior state" would include transactions from its future --
    the exact leak this block is built to avoid -- and nothing would fail. The ingest
    contract guarantees the ordering; this asserts it rather than trusting it.
    """
    ts = df["timestamp"].to_numpy("datetime64[s]").astype(np.int64)
    if np.any(np.diff(ts) < 0):
        first = int(np.argmax(np.diff(ts) < 0))
        raise AssertionError(
            f"transactions are not sorted by timestamp (row {first} goes backwards); "
            f"streaming features would read future state"
        )


def _codes(series: pd.Series) -> np.ndarray:
    categories = pd.Index(sorted(series.dropna().unique()))
    if len(categories) > 62:
        raise ValueError(
            f"{series.name} has {len(categories)} levels; the bitmask holds at most 62"
        )
    return categories.get_indexer(series).astype(np.int64)


assert isinstance(StreamingBlock(), FeatureBlock)
