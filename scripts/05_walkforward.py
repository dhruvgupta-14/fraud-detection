"""Stage 05 -- rolling-origin walk-forward evaluation.

    python scripts/05_walkforward.py [--experiment ablation_graph] [--force]

Trains LightGBM over 5 sequential 2-day blocks in two arms -- retrained each block versus
frozen after the first -- and writes F6 plus the per-block numbers.

Defaults to the E2 arm, since the point is to characterise the model we would actually
deploy (architecture.md 9.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.config import load_config  # noqa: E402
from aml.evaluate import figures  # noqa: E402
from aml.evaluate.walkforward import make_blocks, run_walkforward, summarise  # noqa: E402
from aml.features.assemble import (  # noqa: E402
    FEATURE_SECTIONS,
    FEATURE_STAGE,
    FEATURES_FILE,
    MANIFEST_FILE,
    feature_columns,
)
from aml.io import ArtifactStore, setup_logging, timed  # noqa: E402

LOGGER = logging.getLogger("aml.scripts.walkforward")

RESULTS_FILE = "walkforward.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default="ablation_graph")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    cfg = load_config(args.experiment)
    cfg.seed_everything()
    store = ArtifactStore(cfg)

    feature_stage = store.stage(FEATURE_STAGE, FEATURE_SECTIONS)
    if not feature_stage.exists(FEATURES_FILE):
        raise SystemExit(
            f"No features at {feature_stage.dir}. Run scripts/02_features.py "
            f"--experiment {args.experiment} first"
        )

    metrics_dir = cfg.artifacts_dir / "metrics"
    figures_dir = cfg.artifacts_dir / "figures"
    results_path = metrics_dir / RESULTS_FILE

    if not args.force and results_path.is_file():
        LOGGER.info("results present at %s -- nothing to do (use --force)", results_path)
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        import pandas as pd

        results = pd.DataFrame(payload["blocks"])
    else:
        with timed("05_walkforward"):
            features = feature_stage.read_frame(FEATURES_FILE)
            manifest = feature_stage.read_json(MANIFEST_FILE)
            typology_map = store.read_processed("typology_map.parquet")
            results = run_walkforward(
                features, typology_map, feature_columns(manifest), cfg
            )

        metrics_dir.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(
                {
                    "experiment": args.experiment,
                    "blocks": json.loads(results.to_json(orient="records")),
                    "summary": summarise(results),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    figure = figures.f6_walkforward(results, figures_dir, summarise(results))
    _print_summary(cfg, results, args.experiment, figure)
    return 0


def _print_summary(cfg, results, experiment: str, figure: Path) -> None:
    blocks = make_blocks(cfg)
    summary = summarise(results)

    print("\n" + "=" * 78)
    print("WALK-FORWARD -- architecture.md 9.5")
    print("=" * 78)
    print(f"  experiment          {experiment}")
    print(f"  blocks              {len(blocks)} x {cfg.evaluate.walkforward_block_days} days: {blocks}")
    print("-" * 78)
    print(f"  {'block':>6}{'days':>8}{'positives':>11}{'retrained':>24}{'frozen':>24}")
    for block in sorted(results["block"].unique()):
        rows = results[results["block"] == block].set_index("arm")
        first = rows.iloc[0]
        cells = []
        for arm in ("retrained", "frozen"):
            if arm in rows.index:
                r = rows.loc[arm]
                cells.append(f"{r['auprc']:.4f} [{r['auprc_lo']:.3f},{r['auprc_hi']:.3f}]")
            else:
                cells.append("-")
        print(
            f"  {block:>6}{f'{int(first.first_day)}-{int(first.last_day)}':>8}"
            f"{int(first.n_positives):>11,}{cells[0]:>24}{cells[1]:>24}"
        )
    print("-" * 78)
    print(f"  mean gap (retrained - frozen)   {summary.get('mean_gap', float('nan')):+.4f}")
    print(f"  verdict                         {summary.get('verdict')}")
    print("-" * 78)
    print("  Caveat: 10 usable days, 4 evaluation points, ~900 positives per block.")
    print("          We report the trend and refuse to quote a decay rate from this.")
    print(f"  figure  {figure.name}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
