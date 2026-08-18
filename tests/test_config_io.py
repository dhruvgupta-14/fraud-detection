"""Phase 0 checkpoint tests: config loads and hashes correctly, artifacts round-trip.

These are cheap and they guard the two properties the rest of the build leans on:
scoped hashing (so an unrelated edit does not invalidate an expensive artifact) and
atomic writes (so an interrupted stage never leaves a valid-looking cache entry).
"""

from __future__ import annotations

import pandas as pd
import pytest

from aml.config import Config, load_config
from aml.io import ArtifactStore


# ------------------------------------------------------------------ config loading


def test_default_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.dataset.expected_rows == 5_078_345
    assert cfg.dataset.expected_illicit == 5_177
    # Modelling window is days 0-9; days 10-17 are the generator tail (58% illicit on
    # 0.02% of rows) and are excluded. See architecture.md 2.1.
    assert cfg.time.max_day == 9
    assert cfg.time.train_days == (0, 5)
    assert cfg.time.val_days == (6, 6)
    assert cfg.time.test_days == (7, 9)
    assert cfg.time.purge_straddling_attempts is True
    assert cfg.time.n_days == 10
    assert cfg.graph.exclude_self_loops is True


def test_experiment_overlay_replaces_lists_rather_than_extending():
    base = load_config()
    tabular = load_config("ablation_tabular")
    assert set(base.features.enabled_groups) == {"tabular", "streaming", "structural", "motif"}
    assert tabular.features.enabled_groups == ("tabular",)
    # Deep merge must preserve untouched sibling keys.
    assert tabular.seed == base.seed
    assert tabular.time.train_days == base.time.train_days


def test_needs_snapshots_reflects_enabled_groups():
    assert load_config("ablation_graph").features.needs_snapshots is True
    assert load_config("ablation_tabular").features.needs_snapshots is False


def test_unknown_config_key_is_rejected():
    # A silently ignored typo would let a run use a default nobody intended.
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(overrides={"graph": {"pagerank_dampening": 0.9}})


def test_out_of_order_split_windows_are_rejected():
    with pytest.raises(ValueError, match="strictly ordered"):
        load_config(overrides={"time": {"train_days": [0, 14]}})


def test_random_stack_cv_is_rejected():
    # Random CV is out-of-fold but not causal; refusing it here keeps the temporal
    # contract from being weakened by a one-word config edit.
    with pytest.raises(ValueError, match="leaks across time"):
        load_config(overrides={"models": {"stack_cv": "stratified"}})


# ------------------------------------------------------------------ hashing


def test_hash_is_scoped_to_named_sections():
    base = load_config()
    other = load_config(overrides={"models": {"recall_target": 0.8}})

    feature_sections = ("seed", "time", "graph", "features")
    # A model-only change must not invalidate the ~25 minute feature build.
    assert base.hash_for(*feature_sections) == other.hash_for(*feature_sections)
    assert base.hash_for("models") != other.hash_for("models")


def test_ablation_arms_hash_differently():
    a = load_config("ablation_tabular")
    b = load_config("ablation_graph")
    assert a.hash_for("features") != b.hash_for("features")


def test_hash_is_stable_across_loads():
    assert load_config().hash_for("features") == load_config().hash_for("features")


# ------------------------------------------------------------------ artifact store


@pytest.fixture()
def tmp_cfg(tmp_path) -> Config:
    cfg = load_config()
    return Config.from_dict(
        {
            **cfg.raw,
            "paths": {
                "raw": str(tmp_path / "raw"),
                "processed": str(tmp_path / "processed"),
                "artifacts": str(tmp_path / "artifacts"),
            },
        }
    )


def test_stage_roundtrip_and_sidecar(tmp_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("aml.config.repo_root", lambda: tmp_path)
    monkeypatch.setattr(Config, "root", property(lambda self: tmp_path))

    store = ArtifactStore(tmp_cfg)
    stage = store.stage("features", sections=("seed", "features"))

    df = pd.DataFrame({"tx_id": [0, 1, 2], "score": [0.1, 0.2, 0.3]})
    stage.write_frame(df, "features.parquet")

    pd.testing.assert_frame_equal(stage.read_frame("features.parquet"), df)
    # The sidecar is what lets a bare hash directory be traced back to its settings.
    sidecar = stage.read_json("_config.json")
    assert sidecar["stage"] == "features"
    assert sidecar["sections"] == ["seed", "features"]


def test_cached_frame_builds_once(tmp_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("aml.config.repo_root", lambda: tmp_path)
    monkeypatch.setattr(Config, "root", property(lambda self: tmp_path))

    store = ArtifactStore(tmp_cfg)
    stage = store.stage("features", sections=("seed",))
    calls = {"n": 0}

    def build() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"tx_id": [0], "x": [1.0]})

    stage.cached_frame("f.parquet", build)
    stage.cached_frame("f.parquet", build)
    assert calls["n"] == 1, "second call should be a cache hit"

    stage.cached_frame("f.parquet", build, force=True)
    assert calls["n"] == 2, "force=True must rebuild"


def test_missing_artifact_error_names_the_producing_stage(tmp_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("aml.config.repo_root", lambda: tmp_path)
    monkeypatch.setattr(Config, "root", property(lambda self: tmp_path))

    store = ArtifactStore(tmp_cfg)
    with pytest.raises(FileNotFoundError, match="00_ingest"):
        store.read_processed("transactions.parquet")


def test_test_window_may_not_reach_into_the_generator_tail():
    """Days beyond max_day have an inverted class balance and must not be scored on."""
    with pytest.raises(ValueError, match="max_day"):
        load_config(overrides={"time": {"test_days": [7, 12]}})


def test_split_windows_must_stay_ordered_and_disjoint():
    with pytest.raises(ValueError, match="strictly ordered"):
        load_config(overrides={"time": {"train_days": [0, 6], "val_days": [6, 6]}})


def test_single_day_validation_window_is_allowed():
    """val_days [6, 6] is one day wide; the ordering check must not reject that."""
    cfg = load_config(overrides={"time": {"val_days": [6, 6]}})
    assert cfg.time.val_days == (6, 6)
