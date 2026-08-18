"""The leakage guards (architecture.md 7.6, 11.1).

This is the most important test file in the project, because it is the only one guarding a
failure mode that **cannot be seen in the output**. A leaked feature column looks exactly
like a good one: same dtype, same range, no nulls, no warning. The only symptom is an AUPRC
that is too high, which is indistinguishable from success.

So the properties are asserted on hand-built fixtures where the correct answer is known by
construction, not by running the pipeline and checking the numbers look sensible.

The scenario throughout: three accounts, four transactions, at known times.

    row 0   t=0     0 -> 1   100
    row 1   t=100   1 -> 2    50
    row 2   t=300   0 -> 1   200
    row 3   t=400   1 -> 0    25
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aml.features  # noqa: F401  (registers blocks)
from aml.config import load_config
from aml.features.base import Causality, FeatureContext, build_manifest, enabled_blocks
from aml.features.streaming import StreamingBlock

N_NODES = 3


def make_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["t", "src", "dst", "amount"])
    return pd.DataFrame(
        {
            "tx_id": np.arange(len(df), dtype=np.int64),
            "timestamp": pd.to_datetime("2022-09-01") + pd.to_timedelta(df["t"], unit="s"),
            "day_idx": np.zeros(len(df), dtype=np.int16),
            "src_node": df["src"].to_numpy(np.int32),
            "dst_node": df["dst"].to_numpy(np.int32),
            "amount_paid": df["amount"].to_numpy(float),
            "amount_received": df["amount"].to_numpy(float),
            "currency_paid": "US Dollar",
            "currency_received": "US Dollar",
            "payment_format": "Wire",
            "is_self_loop": df["src"].to_numpy() == df["dst"].to_numpy(),
            "is_cross_currency": False,
            "is_cross_bank": True,
            "label": np.zeros(len(df), dtype=np.int8),
        }
    )


SCENARIO = [(0, 0, 1, 100.0), (100, 1, 2, 50.0), (300, 0, 1, 200.0), (400, 1, 0, 25.0)]


@pytest.fixture
def streamed() -> pd.DataFrame:
    cfg = load_config()
    ctx = FeatureContext(transactions=make_frame(SCENARIO), cfg=cfg, n_nodes=N_NODES)
    return StreamingBlock().compute(ctx)


# --------------------------------------------------------------------------------------
# The core property: a row never sees itself
# --------------------------------------------------------------------------------------


def test_a_transaction_does_not_count_itself(streamed):
    """Row 0 is the first transaction ever. Both endpoints must read a clean slate.

    If state were updated before the read, the receiver would already show one received
    transaction — the account would be counting the very row being scored.
    """
    assert streamed["src_tx_count_out"][0] == 0
    assert streamed["src_volume_out"][0] == 0
    assert streamed["dst_tx_count_in"][0] == 0
    assert streamed["dst_volume_in"][0] == 0


def test_a_transaction_never_sees_a_later_one(streamed):
    """Account 0 sends at t=0 and again at t=300.

    At t=0 it must show no prior sends. At t=300 it must show exactly one — the t=0 send,
    and *not* the t=400 receipt from account 1, which has not happened yet.
    """
    assert streamed["src_tx_count_out"][0] == 0
    assert streamed["src_tx_count_out"][2] == 1
    assert streamed["src_volume_out"][2] == 100.0
    # The t=400 inbound has not happened at t=300.
    assert streamed["src_tx_count_in"][2] == 0


def test_state_accumulates_only_from_the_past(streamed):
    """Row 3: account 1 has received twice (t=0, t=300) and sent once (t=100)."""
    assert streamed["src_tx_count_in"][3] == 2
    assert streamed["src_volume_in"][3] == 300.0
    assert streamed["src_tx_count_out"][3] == 1
    assert streamed["src_volume_out"][3] == 50.0
    assert streamed["src_inout_volume_ratio"][3] == pytest.approx(300.0 / 50.0)


def test_elapsed_times_measure_backwards_only(streamed):
    """Row 3 at t=400; account 1 last received at t=300 and last sent at t=100."""
    assert streamed["src_secs_since_last_in"][3] == 100.0
    assert streamed["src_secs_since_last_out"][3] == 300.0


def test_first_appearance_is_null_not_zero(streamed):
    """"No history" and "history of zero" are different facts.

    Emitting 0.0 for an account's first transaction would tell a tree that the gap since
    its last transfer was instantaneous, which is the opposite of the truth. LightGBM
    handles NaN natively, so the honest value is available at no cost.
    """
    assert np.isnan(streamed["src_secs_since_last_in"][0])
    assert np.isnan(streamed["src_secs_since_last_out"][0])
    assert np.isnan(streamed["dst_mean_amount_in"][0])
    assert np.isnan(streamed["src_turnover_latency"][0])


def test_turnover_latency_reports_a_completed_turnover_not_the_current_one(streamed):
    """Account 1 receives at t=0 and passes it on at t=100 — a 100s turnover.

    That turnover is only *known* after row 1, so row 1 itself must not report it; row 3,
    the account's next appearance, is the first row entitled to see it. This is what makes
    the column distinct from secs_since_last_in rather than a duplicate of it.
    """
    assert np.isnan(streamed["src_turnover_latency"][1])
    assert streamed["src_turnover_latency"][3] == 100.0


def test_counterparty_and_alphabet_counts_exclude_the_current_row(streamed):
    # Row 2, receiver is account 1: it has seen sender {0} and receiver {2} before.
    assert streamed["dst_distinct_counterparties_in"][2] == 1
    assert streamed["dst_distinct_counterparties_out"][2] == 1
    # Row 0 is account 0's first transaction: no currency or format observed yet.
    assert streamed["src_unique_currencies_seen"][0] == 0
    assert streamed["src_unique_formats_seen"][0] == 0
    assert streamed["src_unique_currencies_seen"][2] == 1


def test_account_age_is_zero_on_first_sight_and_grows(streamed):
    assert streamed["src_account_age_secs"][0] == 0.0
    assert streamed["src_account_age_secs"][2] == 300.0


# --------------------------------------------------------------------------------------
# Ordering -- the precondition the whole block rests on
# --------------------------------------------------------------------------------------


def test_out_of_order_input_is_rejected():
    """Unsorted rows would let an account read state from its own future, silently."""
    scrambled = make_frame([(400, 0, 1, 100.0), (0, 1, 2, 50.0)])
    ctx = FeatureContext(transactions=scrambled, cfg=load_config(), n_nodes=N_NODES)
    with pytest.raises(AssertionError, match="not sorted by timestamp"):
        StreamingBlock().compute(ctx)


def test_simultaneous_transactions_are_allowed():
    """Equal timestamps are ordinary here -- the source has minute resolution, so ~26K
    distinct minutes carry 5.08M rows. Only a *backwards* step is an error."""
    tied = make_frame([(0, 0, 1, 10.0), (0, 1, 2, 20.0), (0, 2, 0, 30.0)])
    ctx = FeatureContext(transactions=tied, cfg=load_config(), n_nodes=N_NODES)
    out = StreamingBlock().compute(ctx)
    assert len(out) == 3


def test_self_loop_reads_one_consistent_prior_state():
    """When sender and receiver are the same account, both endpoint reads happen before
    either update, so the two sides must agree rather than one seeing the other's write."""
    frame = make_frame([(0, 0, 1, 100.0), (100, 0, 0, 50.0)])
    ctx = FeatureContext(transactions=frame, cfg=load_config(), n_nodes=N_NODES)
    out = StreamingBlock().compute(ctx)
    assert out["src_tx_count_out"][1] == out["dst_tx_count_out"][1] == 1
    assert out["src_volume_out"][1] == out["dst_volume_out"][1] == 100.0


# --------------------------------------------------------------------------------------
# The manifest is the audit trail
# --------------------------------------------------------------------------------------


def test_every_column_declares_a_causality_class():
    """Architecture 11.1: there is no fourth class, and no unclassified column ships."""
    cfg = load_config()
    manifest = build_manifest(enabled_blocks(cfg), cfg)
    valid = {c.value for c in Causality}
    assert manifest["columns"], "no columns declared"
    for col in manifest["columns"]:
        assert col["causality"] in valid, f"{col['column']} has class {col['causality']!r}"


def test_streaming_columns_are_declared_causal_not_row_local():
    """A mislabelled class would make the report's leakage table wrong even though the
    computation is right -- the manifest is what the report is generated from."""
    for spec in StreamingBlock().columns():
        assert spec.causality is Causality.CAUSAL_STREAMING


def test_tabular_only_config_declares_no_streaming_columns():
    """The E1 control arm must genuinely contain no account history."""
    cfg = load_config(experiment="ablation_tabular")
    manifest = build_manifest(enabled_blocks(cfg), cfg)
    classes = {c["causality"] for c in manifest["columns"]}
    assert classes == {Causality.ROW_LOCAL.value}
