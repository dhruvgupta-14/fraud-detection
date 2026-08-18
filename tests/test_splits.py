"""Split and sampling guards (architecture.md 8.2, 8.3).

Both failures guarded here produce a *higher* score rather than an error:

* a ring appearing in train and test lets the model memorise instead of generalise;
* sub-sampling negatives from test inflates AUPRC by construction, because AUPRC depends
  directly on prevalence.

Neither leaves a trace in the output, so both are asserted on fixtures where the right
answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.config import load_config
from aml.models.sampling import positive_weight, subsample_negatives
from aml.models.splits import assert_no_attempt_straddles, temporal_split


def cfg_for(purge: bool = True):
    return load_config(
        overrides={
            "time": {
                "max_day": 3,
                "train_days": [0, 1],
                "val_days": [2, 2],
                "test_days": [3, 3],
                "purge_straddling_attempts": purge,
            }
        }
    )


def make_features(rows) -> pd.DataFrame:
    """rows: (day_idx, label) tuples. Timestamps are ordered within and across days."""
    df = pd.DataFrame(rows, columns=["day_idx", "label"])
    return pd.DataFrame(
        {
            "tx_id": np.arange(len(df), dtype=np.int64),
            "day_idx": df["day_idx"].to_numpy(np.int16),
            "timestamp": pd.to_datetime("2022-09-01")
            + pd.to_timedelta(df["day_idx"] * 86400 + np.arange(len(df)), unit="s"),
            "label": df["label"].to_numpy(np.int8),
        }
    )


def make_typology(pairs) -> pd.DataFrame:
    """pairs: (tx_id, attempt_id)."""
    return pd.DataFrame(pairs, columns=["tx_id", "attempt_id"]).astype(
        {"tx_id": "int64", "attempt_id": "int32"}
    )


# A ring (attempt 0) straddles train and test: rows 0 (day 0) and 6 (day 3).
# A second ring (attempt 1) sits entirely inside train: rows 1 and 2.
BASE_ROWS = [
    (0, 1),  # 0  attempt 0  -- train side of a straddling ring
    (0, 1),  # 1  attempt 1  -- train only
    (1, 1),  # 2  attempt 1  -- train only
    (1, 0),  # 3  negative, train
    (2, 1),  # 4  val positive
    (2, 0),  # 5  negative, val
    (3, 1),  # 6  attempt 0  -- test side of the straddling ring
    (3, 0),  # 7  negative, test
]
BASE_TYPOLOGY = [(0, 0), (1, 1), (2, 1), (4, 2), (6, 0)]


@pytest.fixture
def features():
    return make_features(BASE_ROWS)


@pytest.fixture
def typology():
    return make_typology(BASE_TYPOLOGY)


# --------------------------------------------------------------------------------------
# Purging
# --------------------------------------------------------------------------------------


def test_purge_removes_the_train_side_of_a_straddling_ring(features, typology):
    split = temporal_split(features, typology, cfg_for(purge=True))
    assert 0 not in split.train.tolist()  # attempt 0's train-side row is gone
    assert 6 in split.test.tolist()  # its test-side row is untouched
    assert split.report["purged_rows"] == 1
    assert split.report["purged_positives"] == 1


def test_purge_keeps_rings_that_do_not_cross_a_boundary(features, typology):
    """Attempt 1 lives entirely inside train and must survive -- purging it would throw
    away training signal for no leakage benefit."""
    split = temporal_split(features, typology, cfg_for(purge=True))
    assert {1, 2} <= set(split.train.tolist())


def test_purge_never_modifies_val_or_test(features, typology):
    """The whole point of purging rather than reassigning: evaluation sets stay intact."""
    purged = temporal_split(features, typology, cfg_for(purge=True))
    unpurged = temporal_split(features, typology, cfg_for(purge=False))
    assert purged.val.tolist() == unpurged.val.tolist()
    assert purged.test.tolist() == unpurged.test.tolist()


def test_disabling_the_purge_leaves_the_ring_on_both_sides(features, typology):
    """Confirms the assertion below is actually testing something."""
    split = temporal_split(features, typology, cfg_for(purge=False))
    assert 0 in split.train.tolist() and 6 in split.test.tolist()
    with pytest.raises(AssertionError, match="appear in both train"):
        assert_no_attempt_straddles(features, typology, split)


def test_purged_split_passes_the_no_straddle_assertion(features, typology):
    split = temporal_split(features, typology, cfg_for(purge=True))
    assert_no_attempt_straddles(features, typology, split)  # must not raise


def test_unannotated_rows_are_never_purged(features, typology):
    """The documented residual leak: rows with no attempt_id cannot be traced to a ring.

    Row 3 is an unannotated train row; it must survive the purge. Asserting this pins the
    limitation in place so it stays a *known* gap rather than drifting into an unknown one.
    """
    split = temporal_split(features, typology, cfg_for(purge=True))
    assert 3 in split.train.tolist()


# --------------------------------------------------------------------------------------
# Temporal contract
# --------------------------------------------------------------------------------------


def test_splits_are_time_ordered_and_disjoint(features, typology):
    split = temporal_split(features, typology, cfg_for())
    ts = features["timestamp"].to_numpy()
    assert ts[split.train].max() < ts[split.val].min()
    assert ts[split.val].max() < ts[split.test].min()
    combined = np.concatenate([split.train, split.val, split.test])
    assert len(np.unique(combined)) == len(combined)


def test_a_split_without_positives_is_rejected():
    """AUPRC is undefined with no positives; failing here beats reporting NaN later."""
    rows = [(0, 1), (1, 1), (2, 0), (3, 1)]  # val day has no positive
    with pytest.raises(AssertionError, match="no positives"):
        temporal_split(make_features(rows), make_typology([]), cfg_for())


def test_days_beyond_max_day_are_rejected():
    """The generator tail must be dropped upstream, not silently split into test."""
    rows = [(0, 1), (1, 1), (2, 1), (3, 1), (9, 1)]
    with pytest.raises(ValueError, match="beyond max_day"):
        temporal_split(make_features(rows), make_typology([]), cfg_for())


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def test_sampling_refuses_any_split_but_train():
    """The guard rail from 8.3, as a hard failure rather than a comment."""
    labels = np.array([0, 1, 0, 0])
    for split in ("val", "test", "TRAIN", ""):
        with pytest.raises(ValueError, match="train-only"):
            subsample_negatives(labels, np.arange(4), ratio=2, seed=1, split=split)


def test_sampling_keeps_every_positive():
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    kept = subsample_negatives(labels, np.arange(10), ratio=2, seed=1, split="train")
    assert set(np.flatnonzero(labels)) <= set(kept.tolist())


def test_sampling_respects_the_ratio():
    labels = np.array([1] + [0] * 99)
    kept = subsample_negatives(labels, np.arange(100), ratio=10, seed=1, split="train")
    assert len(kept) == 11  # 1 positive + 10 negatives


def test_sampling_is_capped_by_available_negatives():
    labels = np.array([1, 1, 0, 0])
    kept = subsample_negatives(labels, np.arange(4), ratio=50, seed=1, split="train")
    assert len(kept) == 4


def test_sampling_is_reproducible_under_the_same_seed():
    labels = np.array([1] + [0] * 99)
    a = subsample_negatives(labels, np.arange(100), ratio=5, seed=42, split="train")
    b = subsample_negatives(labels, np.arange(100), ratio=5, seed=42, split="train")
    assert a.tolist() == b.tolist()


def test_sampling_output_stays_sorted():
    """Row order must remain time-ordered; an unsorted index would scramble the frame."""
    labels = np.array([1] + [0] * 99)
    kept = subsample_negatives(labels, np.arange(100), ratio=20, seed=3, split="train")
    assert kept.tolist() == sorted(kept.tolist())


def test_positive_weight_is_negatives_per_positive():
    labels = np.array([1, 0, 0, 0, 0])
    assert positive_weight(labels, np.arange(5)) == pytest.approx(4.0)
