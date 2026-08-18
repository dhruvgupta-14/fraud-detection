"""Purged temporal split (architecture.md 8.2).

Two things go wrong with a naive split on this dataset, and both produce a *better* number
rather than an error:

1. **A random split** scatters one laundering ring across train and test, so the model
   memorises the ring instead of generalising. Never used here; the split is temporal.
2. **A temporal split alone is not enough**, because 305 of 370 attempts span more than one
   day. A boundary drawn on time cuts through rings, putting near-identical rows on both
   sides.

The fix is a **purge**: the boundary stays on the transaction timestamp, so val and test keep
every row genuinely inside their window, and what gets dropped is the *train-side* remainder
of any attempt that reaches forward across a boundary. Val and test are never modified.

The rejected alternative was assigning each straddling attempt wholly to the earlier split.
That is the more intuitive rule and it is unusable here: it would leave the test window with
59 annotated rows across 29 attempts and destroy the per-typology exhibit (F3). Purging
costs 535 training positives instead, and training positives are the more replaceable
resource.

**Known residual leak, stated rather than hidden.** The purge keys on ``attempt_id``, which
only the 62 %-annotated illicit rows carry. The 1,968 UNANNOTATED illicit rows cannot be
traced to a ring, so ring overlap involving them is invisible here. This is real, bounded,
and named in the report's Limitations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aml.config import Config

LOGGER = logging.getLogger("aml.models.splits")

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """Positional indices into the feature frame, one array per split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    report: dict = field(default_factory=dict)

    def indices(self, name: str) -> np.ndarray:
        if name not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {name!r}")
        return getattr(self, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Split(train={len(self.train):,}, val={len(self.val):,}, test={len(self.test):,})"


def temporal_split(features: pd.DataFrame, typology_map: pd.DataFrame, cfg: Config) -> Split:
    """Split by day window, then purge straddling attempts out of train.

    ``features`` needs only the passthrough columns (``tx_id``, ``day_idx``, ``timestamp``,
    ``label``); ``typology_map`` supplies ``tx_id -> attempt_id``.
    """
    days = features["day_idx"].to_numpy()
    time_cfg = cfg.time

    if days.max() > time_cfg.max_day:
        raise ValueError(
            f"features contain day {days.max()} beyond max_day={time_cfg.max_day}; the "
            f"generator tail must be dropped before splitting (architecture.md 2.1)"
        )

    masks = {
        "train": _window_mask(days, time_cfg.train_days),
        "val": _window_mask(days, time_cfg.val_days),
        "test": _window_mask(days, time_cfg.test_days),
    }
    before = {name: int(m.sum()) for name, m in masks.items()}

    purged = 0
    purged_positives = 0
    if time_cfg.purge_straddling_attempts:
        masks["train"], purged, purged_positives = _purge(features, typology_map, masks)

    split = Split(
        train=np.flatnonzero(masks["train"]),
        val=np.flatnonzero(masks["val"]),
        test=np.flatnonzero(masks["test"]),
        report={
            "rows": {name: int(m.sum()) for name, m in masks.items()},
            "positives": {
                name: int(features.loc[m, "label"].sum()) for name, m in masks.items()
            },
            "rows_before_purge": before,
            "purged_rows": purged,
            "purged_positives": purged_positives,
            "purge_enabled": time_cfg.purge_straddling_attempts,
        },
    )

    _assert_contract(features, split, masks)
    LOGGER.info(
        "split: train=%d val=%d test=%d (purged %d train rows, %d positives)",
        len(split.train),
        len(split.val),
        len(split.test),
        purged,
        purged_positives,
    )
    return split


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _window_mask(days: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    return (days >= window[0]) & (days <= window[1])


def _purge(
    features: pd.DataFrame, typology_map: pd.DataFrame, masks: dict[str, np.ndarray]
) -> tuple[np.ndarray, int, int]:
    """Drop train rows belonging to any attempt that also appears in val or test."""
    attempt_of = pd.Series(
        typology_map["attempt_id"].to_numpy(),
        index=typology_map["tx_id"].to_numpy(),
    )
    tx_id = features["tx_id"].to_numpy()
    attempt = attempt_of.reindex(tx_id).to_numpy()  # NaN where the row is unannotated

    later = masks["val"] | masks["test"]
    forward_attempts = pd.unique(attempt[later & ~pd.isna(attempt)])

    train = masks["train"]
    # Unannotated rows have NaN and are never purged -- the documented residual leak.
    offending = train & np.isin(attempt, forward_attempts)
    kept = train & ~offending

    LOGGER.info(
        "purge: %d attempt(s) reach into val/test, dropping %d train rows",
        len(forward_attempts),
        int(offending.sum()),
    )
    return kept, int(offending.sum()), int(features.loc[offending, "label"].sum())


def _assert_contract(features: pd.DataFrame, split: Split, masks: dict) -> None:
    """The four properties that make the evaluation trustworthy. All fail loudly."""
    ts = features["timestamp"].to_numpy()

    # 1. Strict temporal ordering between splits.
    if len(split.train) and len(split.val):
        if ts[split.train].max() >= ts[split.val].min():
            raise AssertionError("train overlaps val in time")
    if len(split.val) and len(split.test):
        if ts[split.val].max() >= ts[split.test].min():
            raise AssertionError("val overlaps test in time")

    # 2. No split is empty -- an empty val silently disables threshold selection.
    for name in SPLIT_NAMES:
        if len(split.indices(name)) == 0:
            raise AssertionError(f"split {name!r} is empty")

    # 3. Every split must contain positives, or its AUPRC is undefined.
    for name in SPLIT_NAMES:
        idx = split.indices(name)
        if features["label"].to_numpy()[idx].sum() == 0:
            raise AssertionError(f"split {name!r} contains no positives")

    # 4. The splits must be disjoint.
    total = len(split.train) + len(split.val) + len(split.test)
    if len(np.unique(np.concatenate([split.train, split.val, split.test]))) != total:
        raise AssertionError("splits overlap")


def assert_no_attempt_straddles(
    features: pd.DataFrame, typology_map: pd.DataFrame, split: Split
) -> None:
    """No annotated attempt may appear in train and also in val or test.

    Separated from ``_assert_contract`` because it is the specific claim the report makes
    about ring isolation, and ``tests/test_splits.py`` asserts it directly.
    """
    attempt_of = pd.Series(
        typology_map["attempt_id"].to_numpy(), index=typology_map["tx_id"].to_numpy()
    )
    attempt = attempt_of.reindex(features["tx_id"].to_numpy()).to_numpy()

    train_attempts = set(pd.unique(attempt[split.train][~pd.isna(attempt[split.train])]))
    later = np.concatenate([split.val, split.test])
    later_attempts = set(pd.unique(attempt[later][~pd.isna(attempt[later])]))

    overlap = train_attempts & later_attempts
    if overlap:
        raise AssertionError(
            f"{len(overlap)} attempt(s) appear in both train and val/test: "
            f"{sorted(overlap)[:5]}... -- the purge did not run or is broken"
        )
