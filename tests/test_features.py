"""Block correctness and the assembly invariants (architecture.md 7.1, 7.6).

The causality guards live in test_causality.py. This file covers the other half: that the
row-local features compute what they claim to, and that the A3/A4/A5 invariants actually
fire when violated -- an assertion that never triggers is decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aml.features  # noqa: F401
from aml.config import load_config
from aml.features.assemble import PASSTHROUGH, _check_block_output, assemble_features
from aml.features.base import Causality, FeatureSpec, build_manifest, enabled_blocks
from aml.features.tabular import REPORTING_THRESHOLD, TabularBlock
from aml.features.base import FeatureContext


def frame_from(amounts, times=None, currencies=None, received=None) -> pd.DataFrame:
    n = len(amounts)
    times = times or ["2022-09-01 12:00"] * n
    return pd.DataFrame(
        {
            "tx_id": np.arange(n, dtype=np.int64),
            "timestamp": pd.to_datetime(times),
            "day_idx": np.zeros(n, dtype=np.int16),
            "src_node": np.arange(n, dtype=np.int32) % 3,
            "dst_node": (np.arange(n, dtype=np.int32) + 1) % 3,
            "amount_paid": np.asarray(amounts, dtype=float),
            "amount_received": np.asarray(received if received is not None else amounts, dtype=float),
            "currency_paid": currencies or ["US Dollar"] * n,
            "currency_received": currencies or ["US Dollar"] * n,
            "payment_format": ["Wire"] * n,
            "is_self_loop": np.zeros(n, dtype=bool),
            "is_cross_currency": np.zeros(n, dtype=bool),
            "is_cross_bank": np.ones(n, dtype=bool),
            "label": np.zeros(n, dtype=np.int8),
        }
    )


def tabular(df: pd.DataFrame) -> pd.DataFrame:
    return TabularBlock().compute(FeatureContext(df, load_config(), n_nodes=3))


# --------------------------------------------------------------------------------------
# Block A -- row-local
# --------------------------------------------------------------------------------------


def test_round_number_flags():
    out = tabular(frame_from([100.0, 250.0, 1000.0, 1234.56]))
    assert out["is_round_100"].tolist() == [1, 0, 1, 0]
    assert out["is_round_1000"].tolist() == [0, 0, 1, 0]


def test_trailing_zero_count():
    out = tabular(frame_from([1.0, 50.0, 500.0, 50_000.0, 1234.0]))
    assert out["trailing_zero_count"].tolist() == [0, 1, 2, 4, 0]


def test_threshold_distance_is_signed():
    """9,900 and 10,100 are 100 either side of the reporting line.

    An unsigned distance would make them identical to the model, erasing the only thing
    that matters: structuring sits *underneath* the threshold.
    """
    out = tabular(frame_from([REPORTING_THRESHOLD - 100, REPORTING_THRESHOLD + 100]))
    assert out["threshold_distance"][0] < 0
    assert out["threshold_distance"][1] > 0
    assert out["threshold_distance"][0] == pytest.approx(-out["threshold_distance"][1], rel=1e-5)


def test_below_threshold_band_catches_only_the_approach_from_under():
    out = tabular(frame_from([8_000.0, 9_500.0, 9_999.0, 10_000.0, 10_500.0]))
    assert out["below_threshold_band"].tolist() == [0, 1, 1, 0, 0]


def test_amount_mismatch_ratio_and_the_zero_guard():
    out = tabular(frame_from([100.0, 0.0], received=[200.0, 50.0]))
    assert out["amount_mismatch_ratio"][0] == pytest.approx(2.0)
    # A zero-amount row must produce NaN, never inf: LightGBM reads NaN meaningfully and
    # inf silently corrupts split thresholds.
    assert np.isnan(out["amount_mismatch_ratio"][1])


def test_off_hours_flag():
    times = ["2022-09-01 03:00", "2022-09-01 09:00", "2022-09-01 18:00", "2022-09-01 23:00"]
    out = tabular(frame_from([1.0] * 4, times=times))
    assert out["is_off_hours"].tolist() == [1, 0, 1, 1]


def test_categorical_codes_are_stable_under_row_order():
    """Codes are assigned in sorted category order, so they are a pure function of the
    value set -- reshuffling rows must not renumber the currencies."""
    currencies = ["Euro", "US Dollar", "Bitcoin", "Euro"]
    a = tabular(frame_from([1.0] * 4, currencies=currencies))
    b = tabular(frame_from([1.0] * 4, currencies=currencies[::-1]))
    assert a["currency_paid_code"].tolist() == b["currency_paid_code"].tolist()[::-1]


def test_tabular_emits_float32_and_no_infinities():
    out = tabular(frame_from([0.0, 1.0, 1e12]))
    assert all(str(dt) == "float32" for dt in out.dtypes)
    assert not np.isinf(out.to_numpy()).any()


# --------------------------------------------------------------------------------------
# Assembly invariants -- each must actually fire
# --------------------------------------------------------------------------------------


class _UndeclaredColumnBlock:
    name, group, requires_snapshot = "bad", "tabular", False

    def columns(self):
        return [FeatureSpec("declared", Causality.ROW_LOCAL, "d")]

    def compute(self, ctx):
        return pd.DataFrame({"declared": np.ones(len(ctx.transactions), dtype=np.float32),
                             "sneaky": np.ones(len(ctx.transactions), dtype=np.float32)})


class _WrongRowCountBlock(_UndeclaredColumnBlock):
    name = "short"

    def compute(self, ctx):
        return pd.DataFrame({"declared": np.ones(len(ctx.transactions) - 1, dtype=np.float32)})


def test_a5_undeclared_column_is_rejected():
    df = frame_from([1.0, 2.0])
    with pytest.raises(AssertionError, match="undeclared"):
        _check_block_output(_UndeclaredColumnBlock(), _UndeclaredColumnBlock().compute(
            FeatureContext(df, load_config(), 3)), df)


def test_a3_row_count_change_is_rejected():
    df = frame_from([1.0, 2.0])
    with pytest.raises(AssertionError, match="rows for"):
        _check_block_output(_WrongRowCountBlock(), _WrongRowCountBlock().compute(
            FeatureContext(df, load_config(), 3)), df)


def test_assembly_preserves_keys_and_row_count():
    df = frame_from([100.0, 200.0, 300.0])
    out, manifest = assemble_features(df, load_config(), n_nodes=3)
    assert len(out) == len(df)
    assert out["tx_id"].tolist() == df["tx_id"].tolist()
    assert all(col in out.columns for col in PASSTHROUGH)


def test_assembly_emits_exactly_the_manifest_columns():
    df = frame_from([100.0, 200.0])
    out, manifest = assemble_features(df, load_config(), n_nodes=3)
    declared = {c["column"] for c in manifest["columns"]}
    assert set(out.columns) - set(PASSTHROUGH) == declared
    assert manifest["n_columns"] == len(declared)


def test_label_is_passthrough_not_a_feature():
    """The single most costly possible bug: training on the target."""
    df = frame_from([100.0, 200.0])
    _, manifest = assemble_features(df, load_config(), n_nodes=3)
    assert "label" not in {c["column"] for c in manifest["columns"]}


# --------------------------------------------------------------------------------------
# The ablation is a config diff
# --------------------------------------------------------------------------------------


def test_tabular_arm_is_a_strict_subset_of_the_full_arm():
    """E1 and E2 must be the same code path with different config, not two codebases."""
    e1 = build_manifest(enabled_blocks(load_config(experiment="ablation_tabular")),
                        load_config(experiment="ablation_tabular"))
    e2 = build_manifest(enabled_blocks(load_config(experiment="ablation_graph")),
                        load_config(experiment="ablation_graph"))
    a = {c["column"] for c in e1["columns"]}
    b = {c["column"] for c in e2["columns"]}
    assert a < b, "tabular arm must be a proper subset of the full arm"


def test_disabling_every_group_is_an_error_not_an_empty_matrix():
    cfg = load_config(overrides={"features": {"enabled_groups": []}})
    with pytest.raises(ValueError, match="no feature blocks enabled"):
        assemble_features(frame_from([1.0]), cfg, n_nodes=3)
