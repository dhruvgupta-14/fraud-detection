"""Stage 04 -- trained models to report figures and the metrics summary.

    python scripts/04_evaluate.py [--skip-shap] [--log-level DEBUG]

Reads every ablation arm's model directory, so all three must have been trained first:

    for arm in ablation_tabular ablation_streaming ablation_graph; do
        python scripts/02_features.py --experiment $arm
        python scripts/03_train.py    --experiment $arm
    done

Writes:
    artifacts/figures/F1..F5*.png     report exhibits (10.1)
    artifacts/metrics/summary.json    every reported number, one file
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aml.config import load_config  # noqa: E402
from aml.evaluate import figures  # noqa: E402
from aml.evaluate.typology import asymmetry_check, compare_arms_at_budget  # noqa: E402
from aml.io import ArtifactStore, setup_logging, timed  # noqa: E402
from aml.models.train import MODEL_SECTIONS, MODEL_STAGE  # noqa: E402

LOGGER = logging.getLogger("aml.scripts.evaluate")

# The three arms of the ablation, in the order they are reported.
ARMS = {"E1": "ablation_tabular", "E3": "ablation_streaming", "E2": "ablation_graph"}
HEADLINE_MODEL = "lightgbm"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-shap", action="store_true", help="skip F5 (needs the E2 features)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    base = load_config()
    store = ArtifactStore(base)
    figures_dir = base.artifacts_dir / "figures"
    metrics_dir = base.artifacts_dir / "metrics"

    metrics, predictions = _load_arms(store)
    typology_map = store.read_processed("typology_map.parquet")

    with timed("04_evaluate"):
        summary = {"arms": {}, "figures": []}

        # Shared alert budget = the tightest arm's own operating cost, so every arm is
        # compared at the same analyst workload (see evaluate/typology.py).
        budget = metrics["E2"][HEADLINE_MODEL]["test"]["n_flagged"]
        test_days = base.time.test_days[1] - base.time.test_days[0] + 1

        comparison = compare_arms_at_budget(predictions, typology_map, budget)
        summary["typology"] = json.loads(comparison.to_json(orient="records"))
        summary["mechanism_check"] = {
            "E1_to_E2": asymmetry_check(comparison, "E1", "E2"),
            "E3_to_E2": asymmetry_check(comparison, "E3", "E2"),
        }
        summary["shared_alert_budget"] = {"flagged": int(budget), "per_day": budget / test_days}

        for arm, arm_metrics in metrics.items():
            summary["arms"][arm] = {
                name: {k: v for k, v in m["test"].items() if k != "pr_curve"}
                for name, m in arm_metrics.items()
            }

        summary["figures"].append(str(figures.f1_pr_curves(metrics, figures_dir)))
        summary["figures"].append(str(figures.f2_auprc_by_rung(metrics, figures_dir)))
        summary["figures"].append(
            str(figures.f3_typology_recall(comparison, figures_dir, budget / test_days))
        )
        summary["figures"].append(
            str(figures.f4_alert_budget_curve(predictions, figures_dir, test_days))
        )

        if not args.skip_shap:
            shap_summary = _shap(store, figures_dir)
            if shap_summary:
                summary["shap"] = shap_summary["table"]
                summary["figures"].append(shap_summary["figure"])

        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    _print_summary(summary, comparison, metrics, budget / test_days)
    return 0


def _load_arms(store: ArtifactStore):
    metrics, predictions = {}, {}
    for arm, experiment in ARMS.items():
        cfg = load_config(experiment)
        stage = ArtifactStore(cfg).stage(MODEL_STAGE, MODEL_SECTIONS)
        if not stage.exists("metrics.json"):
            raise SystemExit(
                f"Arm {arm} ({experiment}) not trained. Missing {stage.dir}/metrics.json"
            )
        metrics[arm] = stage.read_json("metrics.json")
        predictions[arm] = pd.read_parquet(stage.path(HEADLINE_MODEL) / "predictions.parquet")
        LOGGER.info("loaded arm %s from %s", arm, stage.dir.name)
    return metrics, predictions


def _shap(store: ArtifactStore, figures_dir: Path) -> dict | None:
    """F5 on the E2 LightGBM model. Needs the E2 feature matrix, so it is skippable."""
    import joblib

    from aml.evaluate.explain import shap_by_group
    from aml.features.assemble import (
        FEATURE_SECTIONS,
        FEATURE_STAGE,
        FEATURES_FILE,
        MANIFEST_FILE,
        feature_columns,
    )
    from aml.models.splits import temporal_split

    cfg = load_config(ARMS["E2"])
    arm_store = ArtifactStore(cfg)
    feature_stage = arm_store.stage(FEATURE_STAGE, FEATURE_SECTIONS)
    model_stage = arm_store.stage(MODEL_STAGE, MODEL_SECTIONS)

    if not feature_stage.exists(FEATURES_FILE):
        LOGGER.warning("E2 features missing; skipping F5")
        return None

    with timed("shap"):
        features = feature_stage.read_frame(FEATURES_FILE)
        manifest = feature_stage.read_json(MANIFEST_FILE)
        columns = feature_columns(manifest)
        typology_map = arm_store.read_processed("typology_map.parquet")
        split = temporal_split(features, typology_map, cfg)
        model = joblib.load(model_stage.path(HEADLINE_MODEL) / "model.pkl")

        per_column, per_group, n_sampled = shap_by_group(
            model,
            features,
            columns,
            manifest,
            split.test,
            features["label"].to_numpy(),
            cfg.seed,
        )

    path = figures.f5_shap_by_group(per_group, per_column, figures_dir, n_sampled)
    return {
        "figure": str(path),
        "table": {
            "n_sampled": n_sampled,
            "by_group": json.loads(per_group.to_json(orient="records")),
            "top_columns": json.loads(per_column.head(20).to_json(orient="records")),
        },
    }


def _print_summary(summary: dict, comparison: pd.DataFrame, metrics: dict, per_day: float) -> None:
    print("\n" + "=" * 84)
    print("EVALUATION SUMMARY -- architecture.md 9, 10.1")
    print("=" * 84)
    print(f"  {'arm':<6}{'test AUPRC':>12}{'95% CI':>22}{'alerts/day':>13}{'% traffic':>11}")
    for arm in ("E1", "E3", "E2"):
        t = metrics[arm][HEADLINE_MODEL]["test"]
        lo, hi = t["auprc_ci95"]
        print(
            f"  {arm:<6}{t['auprc']:>12.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}"
            f"{t['alerts_per_day']:>13,.0f}{t['alert_rate']:>10.1%}"
        )
    print("-" * 84)
    print(f"  Per-typology recall at a shared budget of {per_day:,.0f} alerts/day:")
    pivot = comparison.pivot_table(index="typology", columns="arm", values="recall", observed=True)
    counts = comparison[comparison["arm"] == "E2"].set_index("typology")["n_positives"]
    for typology in counts.sort_values(ascending=False).index:
        row = pivot.loc[typology]
        print(
            f"    {typology:<16} n={int(counts[typology]):>4}   "
            f"E1 {row.get('E1', float('nan')):.3f}   E3 {row.get('E3', float('nan')):.3f}   "
            f"E2 {row.get('E2', float('nan')):.3f}"
        )
    print("-" * 84)
    check = summary["mechanism_check"]["E3_to_E2"]
    print("  MECHANISM CHECK (structured typologies vs RANDOM):")
    print(f"    structured  {check['structured_pooled']}   CI {_fmt_ci(check['structured_ci95'])}")
    print(f"    RANDOM      {check['random_pooled']}   CI {_fmt_ci(check['random_ci95'])}")
    print(f"    verdict     {check['verdict']}")
    if "shap" in summary:
        print("-" * 84)
        print("  SHAP attribution by feature group:")
        for row in summary["shap"]["by_group"]:
            print(
                f"    {row['group']:<14}{row['share']:>7.1%}  "
                f"({int(row['n_columns'])} cols, {row['share_per_column']:.2%}/col)"
            )
    print("-" * 84)
    for path in summary["figures"]:
        print(f"    {Path(path).name}")
    print("=" * 84 + "\n")


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


if __name__ == "__main__":
    raise SystemExit(main())
