"""Layer 1 -- raw files to canonical, typed, joined tables (§5).

Artifact filenames live with the module that produces them (``TRANSACTIONS_FILE`` in
transactions.py, ``TYPOLOGY_MAP_FILE`` in patterns.py, ``NODE_INDEX_FILE`` in
graph/interner.py). The ingest *report* has no single producing module -- it merges the
transaction summary with the typology coverage report -- so its name is declared here,
where both the CLI stage and the EDA notebook can import it rather than each hard-coding
the string.
"""

INGEST_REPORT_FILE = "ingest_report.json"

__all__ = ["INGEST_REPORT_FILE"]
