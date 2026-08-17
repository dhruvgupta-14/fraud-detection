"""Patterns.txt -> per-transaction typology annotations.

The file is a block format::

    BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
    <transaction rows, same column order as Trans.csv, no header>
    END LAUNDERING ATTEMPT - FAN-OUT

It carries no transaction id, so rows must be matched back to the canonical table on a
natural key. Two things make that delicate and are handled explicitly rather than left
to a plain ``merge``:

* **The key is not unique.** A single attempt can legitimately contain two transactions
  with identical timestamp, endpoints, amount, currency and format. Matching is therefore
  done by occurrence rank within each key, so the k-th duplicate pattern row binds to the
  k-th duplicate transaction rather than fanning out into a cartesian product.
* **Coverage is partial.** Only ~3,209 of the 5,177 illicit rows are annotated. Illicit
  rows with no block are labelled ``UNANNOTATED``; per-typology recall is reported over
  the annotated subset with the coverage rate stated. Silently treating the annotated set
  as "all positives" would misreport that metric.

Amounts join as integer cents rather than floats, so the match never depends on floating
point equality.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from aml.config import Config
from aml.graph.interner import NodeIndex
from aml.ingest.transactions import RAW_COLUMNS

LOGGER = logging.getLogger("aml.ingest.patterns")

TYPOLOGY_MAP_FILE = "typology_map.parquet"
UNANNOTATED = "UNANNOTATED"

BEGIN_PREFIX = "BEGIN LAUNDERING ATTEMPT"
END_PREFIX = "END LAUNDERING ATTEMPT"

KNOWN_TYPOLOGIES = (
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "BIPARTITE",
    "STACK",
    "RANDOM",
)

_INT_RE = re.compile(r"(\d+)")

# The natural key. src/dst are node ids rather than (bank, account) strings so the join
# runs on integers; the pattern rows are interned through the same NodeIndex.
KEY_COLUMNS = ["ts_i64", "src_node", "dst_node", "amount_cents", "currency_code", "format_code"]


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _parse_header(line: str) -> tuple[str, int | None]:
    """``'BEGIN ... - FAN-OUT:  Max 16-degree Fan-Out'`` -> ``('FAN-OUT', 16)``."""
    try:
        body = line.split(" - ", 1)[1]
    except IndexError as exc:
        raise ValueError(f"Malformed block header: {line!r}") from exc

    family, _, param_text = body.partition(":")
    family = family.strip().upper()
    if family not in KNOWN_TYPOLOGIES:
        raise ValueError(
            f"Unknown typology {family!r} in header {line!r}; known: {list(KNOWN_TYPOLOGIES)}"
        )
    match = _INT_RE.search(param_text)
    return family, int(match.group(1)) if match else None


def parse_patterns(path: Path) -> pd.DataFrame:
    """Parse the block file into one row per annotated transaction.

    Adds ``attempt_id`` -- the grouping key that keeps a single laundering ring from
    being split across a train/test boundary (see models/splits.py).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Download the dataset per the README.")

    records: list[list[str]] = []
    attempt_ids: list[int] = []
    typologies: list[str] = []
    params: list[int | None] = []

    attempt_id = -1
    current: tuple[str, int | None] | None = None

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(BEGIN_PREFIX):
                if current is not None:
                    raise ValueError(f"{path.name}:{lineno}: nested BEGIN without END")
                attempt_id += 1
                current = _parse_header(line)
                continue

            if line.startswith(END_PREFIX):
                if current is None:
                    raise ValueError(f"{path.name}:{lineno}: END without BEGIN")
                closing, _ = _parse_header(line)
                if closing != current[0]:
                    raise ValueError(
                        f"{path.name}:{lineno}: END {closing!r} does not match BEGIN {current[0]!r}"
                    )
                current = None
                continue

            if current is None:
                raise ValueError(f"{path.name}:{lineno}: transaction row outside any block")

            fields = line.split(",")
            if len(fields) != len(RAW_COLUMNS):
                raise ValueError(
                    f"{path.name}:{lineno}: expected {len(RAW_COLUMNS)} fields, got {len(fields)}"
                )
            records.append(fields)
            attempt_ids.append(attempt_id)
            typologies.append(current[0])
            params.append(current[1])

    if current is not None:
        raise ValueError(f"{path.name}: file ended inside an unterminated block")

    df = pd.DataFrame(records, columns=RAW_COLUMNS)
    df["attempt_id"] = np.asarray(attempt_ids, dtype=np.int32)
    df["typology"] = typologies
    df["typology_param"] = pd.array(params, dtype="Int16")
    df["label"] = df["label"].astype(np.int8)

    n_attempts = attempt_id + 1
    LOGGER.info("parsed %d annotated rows across %d attempts", len(df), n_attempts)
    if (df["label"] != 1).any():
        raise ValueError("Patterns.txt contains rows not labelled as laundering")
    return df


# --------------------------------------------------------------------------------------
# Linking
# --------------------------------------------------------------------------------------


def _shared_codes(tx_col: pd.Series, pat_col: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Encode a transaction column and a pattern column against one category set.

    Values present in Patterns.txt but absent from the transaction table encode to -1 and
    can therefore never match, which is what surfaces them in the unmatched report rather
    than letting them silently disappear.
    """
    if not isinstance(tx_col.dtype, pd.CategoricalDtype):
        tx_col = tx_col.astype("category")
    categories = tx_col.cat.categories
    return (
        tx_col.cat.codes.to_numpy(),
        pd.Categorical(pat_col.astype(str), categories=categories).codes,
    )


def _to_cents(values: pd.Series) -> np.ndarray:
    """Amounts as integer cents -- exact join key, no float equality."""
    return np.rint(pd.to_numeric(values).to_numpy() * 100).astype(np.int64)


def link_patterns_to_transactions(
    patterns: pd.DataFrame, transactions: pd.DataFrame, node_index: NodeIndex
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Match annotated rows to ``tx_id``. Returns (typology_map, coverage_report)."""
    pat = patterns.copy()
    pat["_row"] = np.arange(len(pat), dtype=np.int64)

    src_node = node_index.encode(pat["src_bank"], pat["src_acct"])
    dst_node = node_index.encode(pat["dst_bank"], pat["dst_acct"])
    unknown_nodes = int(((src_node < 0) | (dst_node < 0)).sum())

    # Endpoint pre-filter. A pattern row can only match a transaction whose src and dst
    # node both appear among the pattern endpoints, and the key contains both, so this is
    # an exact narrowing rather than a heuristic. It matters because materialising six key
    # columns across all 5.08M rows costs ~200 MB, which this pipeline does not have.
    cand_mask = np.isin(
        transactions["src_node"].to_numpy(), np.unique(src_node[src_node >= 0])
    ) & np.isin(transactions["dst_node"].to_numpy(), np.unique(dst_node[dst_node >= 0]))
    sub = transactions.loc[cand_mask]
    LOGGER.info(
        "endpoint pre-filter: %d transactions -> %d candidate rows", len(transactions), len(sub)
    )

    # Category alignment uses the full category set even though sub is filtered, because
    # pandas retains unused categories on a sliced categorical.
    currency_tx, currency_pat = _shared_codes(sub["currency_paid"], pat["currency_paid"])
    format_tx, format_pat = _shared_codes(sub["payment_format"], pat["payment_format"])

    pat["ts_i64"] = (
        pd.to_datetime(pat["timestamp"], format="%Y/%m/%d %H:%M")
        .astype("datetime64[s]")
        .astype(np.int64)
    )
    pat["src_node"] = src_node
    pat["dst_node"] = dst_node
    pat["amount_cents"] = _to_cents(pat["amount_paid"])
    pat["currency_code"] = currency_pat
    pat["format_code"] = format_pat

    tx_keys = pd.DataFrame(
        {
            "ts_i64": sub["timestamp"].astype("datetime64[s]").astype(np.int64).to_numpy(),
            "src_node": sub["src_node"].to_numpy(),
            "dst_node": sub["dst_node"].to_numpy(),
            "amount_cents": _to_cents(sub["amount_paid"]),
            "currency_code": currency_tx,
            "format_code": format_tx,
            "tx_id": sub["tx_id"].to_numpy(),
            "label": sub["label"].to_numpy(),
        }
    )

    # The candidate pool is the labelled-illicit rows only. Every Patterns.txt row is a
    # laundering transaction by construction (asserted at parse time), so a match against
    # a licit row would be definitionally wrong -- and the natural key is demonstrably
    # non-identifying, so without this restriction a duplicate key can bind the annotation
    # to the licit twin. This is using known ground truth, not peeking: the typology map
    # is evaluation metadata and never becomes a feature.
    illicit_keys = tx_keys.loc[tx_keys["label"] == 1]

    unique_keys = pat[KEY_COLUMNS].drop_duplicates()
    candidates = illicit_keys.merge(unique_keys, on=KEY_COLUMNS, how="inner").sort_values(
        "tx_id", kind="stable", ignore_index=True
    )

    # Occurrence-rank matching: the k-th pattern row with a given key binds to the k-th
    # illicit transaction with that key, in tx_id order. Deterministic, and it neither
    # duplicates nor drops legitimately repeated transactions.
    candidates["occ"] = candidates.groupby(KEY_COLUMNS, sort=False).cumcount()
    pat["occ"] = pat.groupby(KEY_COLUMNS, sort=False).cumcount()

    merged = pat.merge(
        candidates[KEY_COLUMNS + ["occ", "tx_id", "label"]],
        on=KEY_COLUMNS + ["occ"],
        how="left",
        suffixes=("", "_tx"),
    )

    matched_mask = merged["tx_id"].notna()
    n_matched = int(matched_mask.sum())

    typology_map = (
        merged.loc[matched_mask, ["tx_id", "attempt_id", "typology", "typology_param"]]
        .assign(tx_id=lambda d: d["tx_id"].astype(np.int64))
        .sort_values("tx_id", ignore_index=True)
    )
    if typology_map["tx_id"].duplicated().any():
        raise ValueError("A transaction was assigned to more than one laundering attempt")

    # Keys where more illicit rows exist than the pattern blocks claim. We bind the
    # earliest, which is deterministic, but the count belongs in the report.
    key_counts = candidates.groupby(KEY_COLUMNS, sort=False).size().rename("n_tx").reset_index()
    pat_counts = pat.groupby(KEY_COLUMNS, sort=False).size().rename("n_pat").reset_index()
    ambiguity = pat_counts.merge(key_counts, on=KEY_COLUMNS, how="left").fillna({"n_tx": 0})
    n_ambiguous_keys = int((ambiguity["n_tx"] > ambiguity["n_pat"]).sum())

    # Diagnostic on key quality: how many annotated rows have a natural key that also
    # occurs on a licit transaction. A non-zero count is exactly why the candidate pool is
    # restricted above, and it is a number worth stating in the methodology section.
    licit_keys = tx_keys.loc[tx_keys["label"] == 0, KEY_COLUMNS].drop_duplicates()
    licit_keys["_hit"] = True
    collisions = pat[KEY_COLUMNS].merge(licit_keys, on=KEY_COLUMNS, how="left")
    n_licit_collisions = int(collisions["_hit"].notna().sum())

    total_illicit = int(transactions["label"].sum())
    report: dict[str, object] = {
        "pattern_rows": len(pat),
        "attempts": int(pat["attempt_id"].nunique()),
        "matched": n_matched,
        "unmatched": len(pat) - n_matched,
        "unknown_node_rows": unknown_nodes,
        "ambiguous_keys": n_ambiguous_keys,
        "key_collides_with_licit": n_licit_collisions,
        "illicit_total": total_illicit,
        "illicit_annotated": n_matched,
        "illicit_unannotated": total_illicit - n_matched,
        "annotation_coverage": n_matched / total_illicit if total_illicit else 0.0,
        "rows_per_typology": typology_map["typology"].value_counts().to_dict(),
        "attempts_per_typology": (
            typology_map.drop_duplicates("attempt_id")["typology"].value_counts().to_dict()
        ),
    }

    LOGGER.info(
        "typology join: %d/%d matched, %d unmatched, %d ambiguous keys",
        n_matched,
        len(pat),
        len(pat) - n_matched,
        n_ambiguous_keys,
    )
    LOGGER.info(
        "annotation coverage: %d of %d illicit rows (%.1f%%) -- remainder is %s",
        n_matched,
        total_illicit,
        100 * report["annotation_coverage"],
        UNANNOTATED,
    )
    return typology_map, report


def attach_typology(transactions: pd.DataFrame, typology_map: pd.DataFrame) -> pd.Series:
    """Per-transaction typology label for evaluation: family, UNANNOTATED, or NA.

    Licit rows get NA; illicit rows with no block get UNANNOTATED. Keeping those two
    distinct is what stops the per-typology breakdown from quietly absorbing negatives.
    """
    typ = pd.Series(pd.NA, index=transactions.index, dtype="object")
    illicit = transactions["label"].to_numpy() == 1
    typ[illicit] = UNANNOTATED
    mapping = typology_map.set_index("tx_id")["typology"]
    known = transactions["tx_id"].map(mapping)
    typ[known.notna().to_numpy()] = known[known.notna()]
    return typ.astype("category")


def ingest_patterns(
    cfg: Config, transactions: pd.DataFrame, node_index: NodeIndex
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = cfg.raw_dir / f"{cfg.dataset.variant}_Patterns.txt"
    patterns = parse_patterns(path)
    return link_patterns_to_transactions(patterns, transactions, node_index)
