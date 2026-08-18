"""Stage 06 -- build the small data bundle the Streamlit viewer reads.

    python scripts/06_demo_bundle.py [--top 300] [--force]

Writes to artifacts/demo/<config-hash>/:
    alerts.parquet       top-scoring test transactions + a sample, with readable columns
    alert_shap.parquet   per-row SHAP contributions for those transactions
    demo_summary.json    headline metrics for all three arms

Reads only what Phases 1-6 already produced. Nothing is re-scored and no model is refitted.
The bundle exists because the E2 feature matrix is 559 MB and the prediction table is 1.8M
rows -- far too much to load in an app that displays a few hundred alerts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from aml.config import load_config  # noqa: E402
from aml.features.assemble import (  # noqa: E402
    FEATURE_SECTIONS,
    FEATURE_STAGE,
    FEATURES_FILE,
    MANIFEST_FILE,
    feature_columns,
)
from aml.graph.interner import NODE_INDEX_FILE  # noqa: E402
from aml.ingest.transactions import TRANSACTIONS_FILE  # noqa: E402
from aml.io import ArtifactStore, setup_logging, timed  # noqa: E402
from aml.models.train import MODEL_SECTIONS, MODEL_STAGE  # noqa: E402
from aml.viz.demo import (  # noqa: E402
    ALERTS_FILE,
    BUNDLE_STAGE,
    DEFAULT_SAMPLE_OTHERS,
    DEFAULT_TOP_ALERTS,
    SHAP_FILE,
    SUMMARY_FILE,
    alert_shap,
    build_alert_table,
)

LOGGER = logging.getLogger("aml.scripts.demo")

ARMS = {"E1": "ablation_tabular", "E3": "ablation_streaming", "E2": "ablation_graph"}
DEMO_ARM = "ablation_graph"
MODEL = "lightgbm"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_ALERTS)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_OTHERS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    cfg = load_config(DEMO_ARM)
    cfg.seed_everything()
    store = ArtifactStore(cfg)
    bundle = store.stage(BUNDLE_STAGE, ("dataset", "time", "graph", "features", "models"))

    if not args.force and bundle.exists(ALERTS_FILE) and bundle.exists(SUMMARY_FILE):
        LOGGER.info("bundle present at %s -- nothing to do (use --force)", bundle.dir)
        _print_summary(bundle.read_json(SUMMARY_FILE), bundle.dir)
        return 0

    with timed("06_demo_bundle"):
        feature_stage = store.stage(FEATURE_STAGE, FEATURE_SECTIONS)
        model_stage = store.stage(MODEL_STAGE, MODEL_SECTIONS)
        for stage, name in ((feature_stage, "02_features"), (model_stage, "03_train")):
            if not stage.dir.exists() or not any(stage.dir.iterdir()):
                raise SystemExit(f"Missing {stage.dir}. Run scripts/{name}.py --experiment {DEMO_ARM}")

        metrics = model_stage.read_json("metrics.json")
        threshold = metrics[MODEL]["test"]["threshold"]

        predictions = pd.read_parquet(model_stage.path(MODEL) / "predictions.parquet")
        transactions = store.read_processed(TRANSACTIONS_FILE)
        typology_map = store.read_processed("typology_map.parquet")
        node_index = store.read_processed(NODE_INDEX_FILE)

        alerts = build_alert_table(
            predictions, transactions, typology_map, node_index,
            threshold, args.top, args.sample, cfg.seed,
        )
        LOGGER.info("selected %d transactions (%d above threshold)", len(alerts), int(alerts["alert"].sum()))

        features = feature_stage.read_frame(FEATURES_FILE)
        manifest = feature_stage.read_json(MANIFEST_FILE)
        model = joblib.load(model_stage.path(MODEL) / "model.pkl")
        shap_long = alert_shap(
            model, features, feature_columns(manifest), alerts["tx_id"].to_numpy(), manifest
        )

        bundle.write_frame(alerts, ALERTS_FILE)
        bundle.write_frame(shap_long, SHAP_FILE)
        bundle.write_json(_summary(store, threshold, alerts), SUMMARY_FILE)

    _print_summary(bundle.read_json(SUMMARY_FILE), bundle.dir)
    return 0


def _summary(store: ArtifactStore, threshold: float, alerts: pd.DataFrame) -> dict:
    """Headline numbers for all three arms, so the app states them from measured output."""
    arms = {}
    for label, experiment in ARMS.items():
        cfg = load_config(experiment)
        stage = ArtifactStore(cfg).stage(MODEL_STAGE, MODEL_SECTIONS)
        if not stage.exists("metrics.json"):
            LOGGER.warning("arm %s not trained; omitting from the summary", label)
            continue
        test = stage.read_json("metrics.json")[MODEL]["test"]
        arms[label] = {
            "auprc": test["auprc"],
            "auprc_ci95": test["auprc_ci95"],
            "alerts_per_day": test["alerts_per_day"],
            "alert_rate": test["alert_rate"],
            "n_positives": test["n_positives"],
            "prevalence": test["prevalence"],
        }

    reduction = None
    if "E3" in arms and "E2" in arms:
        reduction = 1 - arms["E2"]["alerts_per_day"] / arms["E3"]["alerts_per_day"]

    return {
        "arms": arms,
        "threshold": threshold,
        "alert_reduction_e3_to_e2": reduction,
        "n_bundled": int(len(alerts)),
        "n_above_threshold": int(alerts["alert"].sum()),
    }


def _print_summary(summary: dict, path: Path) -> None:
    print("\n" + "=" * 70)
    print("DEMO BUNDLE -- architecture.md 10.2")
    print("=" * 70)
    print(f"  written to      {path}")
    print(f"  transactions    {summary['n_bundled']:,} ({summary['n_above_threshold']:,} above threshold)")
    print("-" * 70)
    for arm, values in summary["arms"].items():
        print(f"  {arm}  AUPRC {values['auprc']:.4f}   alerts/day {values['alerts_per_day']:>9,.0f}")
    if summary.get("alert_reduction_e3_to_e2"):
        print(f"  alert reduction E3 -> E2: {summary['alert_reduction_e3_to_e2']:.1%}")
    print("-" * 70)
    print("  launch:  streamlit run app/streamlit_app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
