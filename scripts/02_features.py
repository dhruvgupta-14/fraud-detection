"""Stage 02 -- canonical transactions to the feature matrix.

    python scripts/02_features.py [--experiment NAME] [--force] [--log-level DEBUG]

Writes to artifacts/features/<config-hash>/:
    features.parquet        one row per transaction, keyed by tx_id
    feature_manifest.json   per-column block, group, causality class, null policy

The two ablation arms are the same code path with a different config:

    python scripts/02_features.py --experiment ablation_tabular   # E1, Block A only
    python scripts/02_features.py --experiment ablation_graph     # E2, all blocks

They land in different hashed directories, so neither can overwrite the other.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aml.features  # noqa: E402,F401  (registers the blocks)
from aml.config import load_config  # noqa: E402
from aml.features.assemble import (  # noqa: E402
    FEATURE_SECTIONS,
    FEATURE_STAGE,
    FEATURES_FILE,
    MANIFEST_FILE,
    assemble_features,
    null_summary,
)
from aml.ingest.transactions import TRANSACTIONS_FILE  # noqa: E402
from aml.io import ArtifactStore, setup_logging, timed  # noqa: E402

LOGGER = logging.getLogger("aml.scripts.features")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default=None, help="config/experiments/<name>.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    cfg = load_config(args.experiment)
    cfg.seed_everything()
    store = ArtifactStore(cfg)
    stage = store.stage(FEATURE_STAGE, FEATURE_SECTIONS)

    if not args.force and stage.exists(FEATURES_FILE) and stage.exists(MANIFEST_FILE):
        LOGGER.info("features present in %s -- nothing to do (use --force)", stage.dir)
        features = stage.read_frame(FEATURES_FILE)
        manifest = stage.read_json(MANIFEST_FILE)
    else:
        with timed("02_features"):
            transactions = _load_transactions(store, cfg)
            n_nodes = int(
                store.read_processed_json("ingest_report.json")["transactions"]["nodes"]
            )
            snapshots = _load_snapshots(cfg, store)
            features, manifest = assemble_features(transactions, cfg, n_nodes, snapshots)

        stage.write_frame(features, FEATURES_FILE)
        stage.write_json(manifest, MANIFEST_FILE)

    _print_summary(cfg, features, manifest, stage.dir)
    return 0


def _load_transactions(store: ArtifactStore, cfg):
    """Read the canonical table, truncated to the modelling window.

    Days beyond max_day are the generator tail (architecture.md 2.1): 1,108 rows at a 59%
    illicit rate. They are excluded here rather than at train time so that no streaming
    counter ever accumulates state from them.
    """
    df = store.read_processed(TRANSACTIONS_FILE)
    before = len(df)
    df = df.loc[df["day_idx"] <= cfg.time.max_day].reset_index(drop=True)
    LOGGER.info(
        "loaded %d rows, kept %d within max_day=%d (dropped %d generator-tail rows)",
        before,
        len(df),
        cfg.time.max_day,
        before - len(df),
    )
    return df


def _load_snapshots(cfg, store: ArtifactStore) -> dict:
    """Snapshots, if any enabled block needs them.

    Phase 3 enables only tabular and streaming, so this returns empty and stage 01 is not a
    prerequisite yet. It becomes one in Phase 5.
    """
    if not cfg.features.needs_snapshots:
        return {}
    from aml.graph.snapshots import Snapshot, open_store

    stage = open_store(store)
    return {day: Snapshot.load(stage, day) for day in range(cfg.time.max_day + 1)}


def _print_summary(cfg, features, manifest, path: Path) -> None:
    nulls = null_summary(features, manifest)
    by_causality = manifest["columns_by_causality"]

    print("\n" + "=" * 76)
    print("FEATURE SUMMARY -- architecture.md 7.6 / 11.1")
    print("=" * 76)
    print(f"  experiment          {cfg.experiment or 'default'}")
    print(f"  enabled groups      {', '.join(cfg.features.enabled_groups)}")
    print(f"  blocks              {', '.join(manifest['blocks'])}")
    print(f"  matrix              {len(features):,} rows x {manifest['n_columns']} feature columns")
    print(f"  memory              {features.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    print(f"  written to          {path}")
    print("-" * 76)
    print("  causality classes (the leakage audit):")
    for klass, count in sorted(by_causality.items()):
        print(f"    {klass:<20} {count:>4} columns")
    print("-" * 76)
    top = nulls.sort_values("null_rate", ascending=False).head(8)
    if top["null_rate"].max() > 0:
        print("  highest null rates (all within declared policy):")
        for _, row in top.iterrows():
            if row["null_rate"] > 0:
                print(f"    {row['column']:<36} {row['null_rate']:>7.2%}  [{row['null_policy']}]")
    else:
        print("  no nulls emitted")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
