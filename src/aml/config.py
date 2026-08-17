"""Typed configuration loading and content-addressed hashing.

Every stage of the pipeline reads its settings from here, and every artifact on disk
is keyed by a hash of the config subset that produced it. That keying is what lets the
two ablation arms (E1 tabular-only, E2 tabular+graph) coexist without overwriting each
other's features, and what stops a stale cache from silently poisoning a result.

Config is exposed as frozen dataclasses rather than raw dicts on purpose: a typo like
``cfg.time.train_dayz`` raises AttributeError immediately, whereas ``cfg["time"]["train_dayz"]``
on a dict-of-dicts tends to surface hours later as a confusing KeyError deep in a fit call.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

CONFIG_DIR_NAME = "config"
DEFAULT_CONFIG_NAME = "default.yaml"
EXPERIMENTS_DIR_NAME = "experiments"

# Length of the truncated sha256 used in artifact directory names. 12 hex chars is
# 48 bits -- collision risk is nil at the handful-of-experiments scale, and short
# enough that the paths stay readable when debugging.
HASH_LEN = 12


def repo_root() -> Path:
    """Locate the project root by walking up from this file until config/ appears."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / CONFIG_DIR_NAME / DEFAULT_CONFIG_NAME).is_file():
            return candidate
    raise RuntimeError(
        f"Could not locate repo root: no {CONFIG_DIR_NAME}/{DEFAULT_CONFIG_NAME} "
        f"found in any parent of {here}"
    )


# --------------------------------------------------------------------------------------
# Section dataclasses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PathsConfig:
    raw: str
    processed: str
    artifacts: str


@dataclass(frozen=True)
class DatasetConfig:
    variant: str
    expected_rows: int
    expected_illicit: int
    expected_nodes: int
    expected_pattern_rows: int
    expected_attempts: int
    origin_date: str


@dataclass(frozen=True)
class TimeConfig:
    snapshot_granularity: str
    lookback_days: int | None
    train_days: tuple[int, int]
    val_days: tuple[int, int]
    test_days: tuple[int, int]

    def __post_init__(self) -> None:
        # The temporal contract is the project's headline methodological claim, so the
        # split boundaries are validated at load time rather than at fit time.
        if not (self.train_days[1] < self.val_days[0] < self.val_days[1] < self.test_days[0]):
            raise ValueError(
                f"Split windows must be strictly ordered and non-overlapping, got "
                f"train={self.train_days} val={self.val_days} test={self.test_days}"
            )
        for name, window in (
            ("train_days", self.train_days),
            ("val_days", self.val_days),
            ("test_days", self.test_days),
        ):
            if window[0] > window[1]:
                raise ValueError(f"{name} is inverted: {window}")
        if self.lookback_days is not None and self.lookback_days < 1:
            raise ValueError(f"lookback_days must be >= 1 or null, got {self.lookback_days}")

    @property
    def n_days(self) -> int:
        return self.test_days[1] + 1


@dataclass(frozen=True)
class GraphConfig:
    backend: str
    exclude_self_loops: bool
    pagerank_damping: float
    pagerank_weighted: bool

    def __post_init__(self) -> None:
        if self.backend not in {"igraph", "networkx"}:
            raise ValueError(f"Unknown graph backend: {self.backend!r}")


@dataclass(frozen=True)
class FeaturesConfig:
    enabled_groups: tuple[str, ...]
    streaming: Mapping[str, Any]
    motifs: Mapping[str, Any]
    community_prior: Mapping[str, Any]

    VALID_GROUPS = ("tabular", "streaming", "structural", "motif", "reference")

    def __post_init__(self) -> None:
        unknown = set(self.enabled_groups) - set(self.VALID_GROUPS)
        if unknown:
            raise ValueError(
                f"Unknown feature group(s) {sorted(unknown)}; valid: {list(self.VALID_GROUPS)}"
            )

    def enabled(self, group: str) -> bool:
        return group in self.enabled_groups

    @property
    def needs_snapshots(self) -> bool:
        """True when any enabled group reads from a graph snapshot."""
        return bool({"structural", "motif"} & set(self.enabled_groups))


@dataclass(frozen=True)
class SamplingConfig:
    negative_ratio: int
    account_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.account_fraction <= 1.0:
            raise ValueError(f"account_fraction must be in (0, 1], got {self.account_fraction}")


@dataclass(frozen=True)
class ModelsConfig:
    recall_target: float
    stack_cv: str
    lightgbm: Mapping[str, Any]
    random_forest: Mapping[str, Any]
    logistic_regression: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not 0.0 < self.recall_target < 1.0:
            raise ValueError(f"recall_target must be in (0, 1), got {self.recall_target}")
        if self.stack_cv != "temporal":
            # sklearn's default StratifiedKFold is out-of-fold but NOT causal; using it
            # would leak future rows into the meta-learner's training data.
            raise ValueError(
                f"stack_cv must be 'temporal' -- random CV leaks across time. Got {self.stack_cv!r}"
            )


@dataclass(frozen=True)
class EvaluateConfig:
    bootstrap_iterations: int
    walkforward_blocks: int


@dataclass(frozen=True)
class Config:
    """Fully resolved configuration for one pipeline run."""

    seed: int
    paths: PathsConfig
    dataset: DatasetConfig
    time: TimeConfig
    graph: GraphConfig
    features: FeaturesConfig
    sampling: SamplingConfig
    models: ModelsConfig
    evaluate: EvaluateConfig

    # The merged dict this was built from. Kept so hashing operates on exactly what was
    # loaded, including any keys added to the YAML after these dataclasses were written.
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)
    experiment: str | None = None

    # ---------------------------------------------------------------- construction

    @classmethod
    def load(
        cls,
        experiment: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        config_dir: Path | None = None,
    ) -> "Config":
        """Load default.yaml, overlay an optional experiment file, then dict overrides.

        ``experiment`` names a file in config/experiments/ with or without the .yaml
        suffix. Overlays are deep-merged, so an experiment file only needs to state the
        keys it actually changes -- which is what keeps the ablation a config diff
        rather than a duplicated config.
        """
        config_dir = config_dir or (repo_root() / CONFIG_DIR_NAME)
        merged = _read_yaml(config_dir / DEFAULT_CONFIG_NAME)

        if experiment:
            name = experiment if experiment.endswith(".yaml") else f"{experiment}.yaml"
            exp_path = config_dir / EXPERIMENTS_DIR_NAME / name
            if not exp_path.is_file():
                available = sorted(
                    p.stem for p in (config_dir / EXPERIMENTS_DIR_NAME).glob("*.yaml")
                )
                raise FileNotFoundError(
                    f"No experiment config {exp_path}. Available: {available}"
                )
            merged = _deep_merge(merged, _read_yaml(exp_path))

        if overrides:
            merged = _deep_merge(merged, overrides)

        return cls.from_dict(merged, experiment=experiment)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], experiment: str | None = None) -> "Config":
        return cls(
            seed=int(d["seed"]),
            paths=_build(PathsConfig, d["paths"]),
            dataset=_build(DatasetConfig, d["dataset"]),
            time=_build(TimeConfig, d["time"], tuple_fields=("train_days", "val_days", "test_days")),
            graph=_build(GraphConfig, d["graph"]),
            features=_build(FeaturesConfig, d["features"], tuple_fields=("enabled_groups",)),
            sampling=_build(SamplingConfig, d["sampling"]),
            models=_build(ModelsConfig, d["models"]),
            evaluate=_build(EvaluateConfig, d["evaluate"]),
            raw=d,
            experiment=experiment,
        )

    # ---------------------------------------------------------------- paths

    @property
    def root(self) -> Path:
        return repo_root()

    @property
    def raw_dir(self) -> Path:
        return self.root / self.paths.raw

    @property
    def processed_dir(self) -> Path:
        return self.root / self.paths.processed

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.paths.artifacts

    # ---------------------------------------------------------------- hashing

    def hash_for(self, *sections: str) -> str:
        """Stable short hash over the named top-level config sections.

        Scoped deliberately: the feature artifact is keyed on the sections that actually
        affect features, so changing ``models.recall_target`` does not invalidate a
        25-minute feature build. Callers name their true dependencies.
        """
        if not sections:
            raise ValueError("hash_for() requires at least one section name")
        subset: dict[str, Any] = {}
        for section in sections:
            if section not in self.raw:
                raise KeyError(f"No config section {section!r}; have {sorted(self.raw)}")
            subset[section] = self.raw[section]
        payload = json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LEN]

    def describe(self, *sections: str) -> dict[str, Any]:
        """The exact config subset behind a hash, for writing next to the artifact."""
        return {s: self.raw[s] for s in sections}

    # ---------------------------------------------------------------- determinism

    def seed_everything(self) -> None:
        """Seed every global RNG we rely on.

        Estimators still receive the seed explicitly (sklearn and LightGBM do not read
        the global numpy RNG reliably); this covers the incidental uses -- sampling,
        shuffles, and igraph's Louvain.
        """
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:  # numpy absent only in a bare config-only environment
            pass


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return loaded


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base. Scalars and lists are replaced, not extended.

    Lists are replaced so that ``enabled_groups: [tabular]`` in an ablation file means
    exactly that, rather than appending to the default four.
    """
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, values: Mapping[str, Any], tuple_fields: Iterable[str] = ()) -> Any:
    """Instantiate a section dataclass, rejecting unknown keys.

    Rejecting unknowns is the point: a misspelled YAML key would otherwise be silently
    ignored and the run would quietly use the default, which is the kind of thing that
    invalidates an ablation without anyone noticing.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in config section '{cls.__name__}'; "
            f"expected any of {sorted(known)}"
        )
    missing = known - set(values)
    if missing:
        raise ValueError(f"Missing key(s) {sorted(missing)} in config section '{cls.__name__}'")

    kwargs = dict(values)
    for name in tuple_fields:
        if name in kwargs and isinstance(kwargs[name], list):
            kwargs[name] = tuple(kwargs[name])
    return cls(**kwargs)


def load_config(
    experiment: str | None = None, overrides: Mapping[str, Any] | None = None
) -> Config:
    """Convenience wrapper matching the CLI entry points in scripts/."""
    return Config.load(experiment=experiment, overrides=overrides)
