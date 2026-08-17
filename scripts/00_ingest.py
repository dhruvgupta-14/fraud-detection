"""Stage 00 -- raw files to canonical parquet.

    python scripts/00_ingest.py [--force] [--log-level DEBUG]

Writes to data/processed/:
    transactions.parquet   canonical transaction table (architecture.md 5.1)
    node_index.parquet     (bank, acct) <-> node_id
    typology_map.parquet   tx_id -> (attempt_id, typology)
    ingest_report.json     the measured facts, for the report and the EDA notebook

Unhashed output: these artifacts depend only on the raw files and the dataset section of
the config, so there is exactly one correct version of them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml.config import load_config  # noqa: E402
from aml.graph.interner import NODE_INDEX_FILE, NodeIndex  # noqa: E402
from aml.ingest.patterns import TYPOLOGY_MAP_FILE, ingest_patterns  # noqa: E402
from aml.ingest.transactions import TRANSACTIONS_FILE, ingest_transactions  # noqa: E402
from aml.io import ArtifactStore, setup_logging, timed  # noqa: E402

LOGGER = logging.getLogger("aml.scripts.ingest")

REPORT_FILE = "ingest_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, help="config/experiments/<name>.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    cfg = load_config(args.experiment)
    cfg.seed_everything()
    store = ArtifactStore(cfg)

    outputs = (TRANSACTIONS_FILE, NODE_INDEX_FILE, TYPOLOGY_MAP_FILE, REPORT_FILE)
    if not args.force and all(store.has_processed(name) for name in outputs):
        LOGGER.info("all outputs present in %s -- nothing to do (use --force to rebuild)", cfg.processed_dir)
        return 0

    with timed("00_ingest"):
        with timed("transactions"):
            transactions, node_index, summary = ingest_transactions(cfg)

        with timed("patterns"):
            typology_map, coverage = ingest_patterns(cfg, transactions, node_index)

        _check_coverage(cfg, coverage)

        store.write_processed(transactions, TRANSACTIONS_FILE)
        node_index.save(store.processed(NODE_INDEX_FILE))
        store.write_processed(typology_map, TYPOLOGY_MAP_FILE)
        store.write_processed_json({"transactions": summary, "typology": coverage}, REPORT_FILE)

    _print_summary(summary, coverage)
    return 0


def _check_coverage(cfg, coverage: dict) -> None:
    """Warn loudly if the typology join drifts from the recorded baseline.

    Not fatal: the join is heuristic by nature (Patterns.txt has no id) and a small drift
    after a parser change is worth seeing rather than crashing on. A large drift means the
    per-typology breakdown is measuring the wrong rows.
    """
    ds = cfg.dataset
    if coverage["pattern_rows"] != ds.expected_pattern_rows:
        LOGGER.warning(
            "parsed %d pattern rows, expected %d", coverage["pattern_rows"], ds.expected_pattern_rows
        )
    if coverage["attempts"] != ds.expected_attempts:
        LOGGER.warning("parsed %d attempts, expected %d", coverage["attempts"], ds.expected_attempts)
    if coverage["unmatched"]:
        LOGGER.warning(
            "%d pattern rows did not match any transaction (%d had unknown accounts)",
            coverage["unmatched"],
            coverage["unknown_node_rows"],
        )


def _print_summary(summary: dict, coverage: dict) -> None:
    print("\n" + "=" * 72)
    print("INGEST SUMMARY -- compare against architecture.md section 2")
    print("=" * 72)
    print(f"  rows                {summary['rows']:>12,}")
    print(f"  illicit             {summary['illicit']:>12,}  ({summary['illicit_rate']:.4%})")
    print(f"  nodes               {summary['nodes']:>12,}")
    print(f"  banks               {summary['banks']:>12,}")
    print(f"  span                {summary['first_timestamp']} -> {summary['last_timestamp']}")
    print(f"  days                {summary['n_days']:>12,}")
    print(
        f"  self-loops          {summary['self_loops']:>12,}  "
        f"({summary['self_loops'] / summary['rows']:.1%}, {summary['self_loops_illicit']} illicit)"
    )
    print(f"  cross-currency      {summary['cross_currency']:>12,}")
    print(f"  cross-bank          {summary['cross_bank']:>12,}")
    print("-" * 72)
    print(f"  attempts            {coverage['attempts']:>12,}")
    print(f"  annotated rows      {coverage['matched']:>12,}  of {coverage['pattern_rows']:,} parsed")
    print(
        f"  coverage of illicit {coverage['annotation_coverage']:>11.1%}  "
        f"({coverage['illicit_unannotated']:,} rows are UNANNOTATED)"
    )
    print("-" * 72)
    print("  attempts per typology:")
    for name, count in sorted(coverage["attempts_per_typology"].items(), key=lambda kv: -kv[1]):
        rows = coverage["rows_per_typology"].get(name, 0)
        print(f"    {name:<16} {count:>4} attempts  {rows:>6,} rows")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
