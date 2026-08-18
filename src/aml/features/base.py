"""Feature block protocol, registry, and the manifest that backs the leakage claim.

Every feature column in the project is produced by exactly one *block*, and every block
declares its columns up front with a **causality class** (architecture.md 11.1):

    row_local          derived only from the row itself
    causal_streaming   derived from state strictly before this row's timestamp
    lagged_snapshot    derived from the D-1 graph snapshot

There is no fourth class. A feature that cannot be assigned one of these three does not
ship -- that rule is what makes "our features are causal" a checkable claim rather than an
assurance.

``feature_manifest.json`` is written from these declarations, and the report's "which
columns could leak" table is generated from the manifest rather than written by hand. If
the two ever disagree, the manifest is right and the prose is wrong.

Blocks are switchable by group, which is how the headline ablation is a config diff
(``enabled_groups: [tabular]`` vs. the full four) rather than a second codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import pandas as pd

from aml.config import Config

LOGGER = logging.getLogger("aml.features")

# Tree models do not need float64, and halving the matrix halves LightGBM's binning copy.
# At 5.08M rows this is the difference between a 3.25 GB and a 1.63 GB frame (R9).
DEFAULT_DTYPE = "float32"


class Causality(str, Enum):
    ROW_LOCAL = "row_local"
    CAUSAL_STREAMING = "causal_streaming"
    LAGGED_SNAPSHOT = "lagged_snapshot"


@dataclass(frozen=True)
class FeatureSpec:
    """One emitted column, and everything the manifest needs to say about it."""

    name: str
    causality: Causality
    description: str
    dtype: str = DEFAULT_DTYPE
    # Where nulls are legitimate. "never" is asserted at assembly; the others are the two
    # places a null is meaningful rather than a bug (architecture.md 6.2, 7.6-A4).
    null_policy: str = "never"  # never | cold_start | dormant

    def __post_init__(self) -> None:
        if self.null_policy not in {"never", "cold_start", "dormant"}:
            raise ValueError(f"unknown null_policy {self.null_policy!r} on {self.name}")


@dataclass
class FeatureContext:
    """Everything a block is allowed to read.

    Deliberately narrow: a block receives the transaction frame and the config, and nothing
    that would let it reach forward in time. Snapshot-reading blocks (Phase 5) take their
    snapshots through this object too, so the D-1 lag is applied in one place.
    """

    transactions: pd.DataFrame
    cfg: Config
    n_nodes: int
    snapshots: dict[int, object] = field(default_factory=dict)


@runtime_checkable
class FeatureBlock(Protocol):
    name: str
    group: str
    requires_snapshot: bool

    def columns(self) -> list[FeatureSpec]: ...

    def compute(self, ctx: FeatureContext) -> pd.DataFrame: ...


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator; makes a block discoverable by name."""
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate feature block name {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def get_block(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown feature block {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def enabled_blocks(cfg: Config) -> list:
    """Instantiate every registered block whose group is enabled in config.

    Import order matters only in that a block must have been imported to be registered;
    ``aml.features`` imports them all, so callers do not have to think about it.
    """
    blocks = [cls() for cls in _REGISTRY.values() if cfg.features.enabled(cls.group)]
    return sorted(blocks, key=lambda b: b.name)


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def build_manifest(blocks: list, cfg: Config) -> dict:
    """The auditable record behind the temporal-integrity claim (architecture.md 11.1)."""
    columns = []
    for block in blocks:
        for spec in block.columns():
            columns.append(
                {
                    "column": spec.name,
                    "block": block.name,
                    "group": block.group,
                    "causality": spec.causality.value,
                    "dtype": spec.dtype,
                    "null_policy": spec.null_policy,
                    "description": spec.description,
                }
            )

    by_causality: dict[str, int] = {}
    for col in columns:
        by_causality[col["causality"]] = by_causality.get(col["causality"], 0) + 1

    return {
        "enabled_groups": list(cfg.features.enabled_groups),
        "blocks": [b.name for b in blocks],
        "n_columns": len(columns),
        "columns_by_causality": by_causality,
        "columns": columns,
    }


def manifest_columns(manifest: dict) -> list[str]:
    return [c["column"] for c in manifest["columns"]]
