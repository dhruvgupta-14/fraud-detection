"""Stage 03 -- feature matrix to trained models and scored predictions.

    python scripts/03_train.py [--experiment NAME] [--models lightgbm,random_forest] [--force]

Writes to artifacts/models/<config-hash>/:
    <model>/model.pkl            fitted estimator (joblib)
    <model>/predictions.parquet  tx_id -> score, for val and test
    metrics.json                 every reported number, all models
    split_report.json            row and positive counts, purge cost

This is the pivotal checkpoint (architecture.md 16, Phase 4): after it runs, there is a real
baseline AUPRC and everything downstream is measured improvement rather than plan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402

from aml.config import load_config  # noqa: E402
from aml.features.assemble import (  # noqa: E402
    FEATURE_SECTIONS,
    FEATURE_STAGE,
    FEATURES_FILE,
    MANIFEST_FILE,
    feature_columns,
)
from aml.io import ArtifactStore, setup_logging, timed  # noqa: E402
from aml.models.registry import model_names  # noqa: E402
from aml.models.splits import assert_no_attempt_straddles, temporal_split  # noqa: E402
from aml.models.train import MODEL_SECTIONS, MODEL_STAGE, train_and_score  # noqa: E402

LOGGER = logging.getLogger("aml.scripts.train")

METRICS_FILE = "metrics.json"
SPLIT_REPORT_FILE = "split_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--models", default=None, help="comma-separated subset to train")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    cfg = load_config(args.experiment)
    cfg.seed_everything()
    store = ArtifactStore(cfg)

    feature_store = store.stage(FEATURE_STAGE, FEATURE_SECTIONS)
    if not feature_store.exists(FEATURES_FILE):
        raise SystemExit(
            f"No features at {feature_store.dir}. Run scripts/02_features.py first"
            + (f" --experiment {args.experiment}" if args.experiment else "")
        )

    model_store = store.stage(MODEL_STAGE, MODEL_SECTIONS)
    wanted = model_names(args.models.split(",") if args.models else None)

    if not args.force and model_store.exists(METRICS_FILE):
        existing = model_store.read_json(METRICS_FILE)
        if all(name in existing for name in wanted):
            LOGGER.info("metrics present in %s -- nothing to do (use --force)", model_store.dir)
            _print_summary(cfg, existing, model_store.read_json(SPLIT_REPORT_FILE), model_store.dir)
            return 0

    with timed("03_train"):
        features = feature_store.read_frame(FEATURES_FILE)
        manifest = feature_store.read_json(MANIFEST_FILE)
        columns = feature_columns(manifest)
        typology = store.read_processed("typology_map.parquet")

        split = temporal_split(features, typology, cfg)
        assert_no_attempt_straddles(features, typology, split)
        model_store.write_json(split.report, SPLIT_REPORT_FILE)

        all_metrics = {}
        for name in wanted:
            with timed(f"train:{name}"):
                model, predictions, metrics = train_and_score(
                    name, features, columns, split, cfg
                )
            target = model_store.path(name) / "model.pkl"
            target.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, target)
            predictions.to_parquet(
                model_store.path(name) / "predictions.parquet", index=False
            )
            all_metrics[name] = metrics

        model_store.write_json(all_metrics, METRICS_FILE)

    _print_summary(cfg, all_metrics, split.report, model_store.dir)
    return 0


def _print_summary(cfg, metrics: dict, split_report: dict, path: Path) -> None:
    print("\n" + "=" * 88)
    print("TRAINING SUMMARY -- architecture.md 8, 9.1")
    print("=" * 88)
    print(f"  experiment          {cfg.experiment or 'default'}")
    print(f"  written to          {path}")
    print("-" * 88)
    rows = split_report["rows"]
    pos = split_report["positives"]
    print(
        f"  split               train {rows['train']:>9,} ({pos['train']:>5,} pos)  "
        f"val {rows['val']:>9,} ({pos['val']:>4,})  test {rows['test']:>9,} ({pos['test']:>5,})"
    )
    if split_report["purge_enabled"]:
        print(
            f"  purge cost          {split_report['purged_rows']:,} train rows, "
            f"{split_report['purged_positives']:,} positives "
            f"({split_report['purged_positives'] / max(pos['train'] + split_report['purged_positives'], 1):.0%} of train positives)"
        )
    print("-" * 88)
    print(f"  {'model':<22}{'val AUPRC':>11}{'test AUPRC':>12}{'95% CI':>20}{'lift':>8}{'alerts/day':>12}")
    for name, m in metrics.items():
        test = m["test"]
        lo, hi = test.get("auprc_ci95", (float('nan'), float('nan')))
        print(
            f"  {name:<22}{m['val']['auprc']:>11.4f}{test['auprc']:>12.4f}"
            f"{f'[{lo:.4f}, {hi:.4f}]':>20}{test['auprc_lift_over_random']:>7.0f}x"
            f"{test['alerts_per_day']:>12,.0f}"
        )
    print("-" * 88)
    any_metrics = next(iter(metrics.values()))
    print(
        f"  test prevalence     {any_metrics['test']['prevalence']:.4%}  "
        f"-- a random ranker scores AUPRC = prevalence; 'lift' is the multiple over that"
    )
    print(
        f"  threshold policy    chosen on VALIDATION at recall >= {any_metrics['recall_target']:.0%}, "
        f"applied unchanged to test"
    )
    print("=" * 88 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
