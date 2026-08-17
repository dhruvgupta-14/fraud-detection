"""Raw CSV -> canonical transaction table.

The raw file has two columns literally named ``Account`` (sender and receiver), amounts
in received-then-paid order, and timestamps as ``YYYY/MM/DD HH:MM`` strings. All of that
is resolved exactly once here so no downstream module ever parses a string or has to
remember which ``Account`` is which.

Memory note: the file is ~480 MB / 5.08M rows. Every low-cardinality column is read
directly as a categorical, including the timestamp -- there are only 25,920 distinct
minutes in the 18-day span, so parsing the categories rather than the column turns 5M
datetime parses into ~26K.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aml.config import Config
from aml.graph.interner import NodeIndex, aligned_codes

LOGGER = logging.getLogger("aml.ingest.transactions")

TRANSACTIONS_FILE = "transactions.parquet"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

# Positional names for the raw file. Supplied explicitly because the header has a
# duplicate 'Account' and because the amount columns arrive received-before-paid,
# which is easy to transpose by accident.
RAW_COLUMNS = [
    "timestamp",
    "src_bank",
    "src_acct",
    "dst_bank",
    "dst_acct",
    "amount_received",
    "currency_received",
    "amount_paid",
    "currency_paid",
    "payment_format",
    "label",
]

RAW_HEADER = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

READ_DTYPES = {
    "timestamp": "category",
    "src_bank": "category",
    "src_acct": "category",
    "dst_bank": "category",
    "dst_acct": "category",
    "amount_received": "float64",
    "currency_received": "category",
    "amount_paid": "float64",
    "currency_paid": "category",
    "payment_format": "category",
    "label": "int8",
}

CANONICAL_COLUMNS = [
    "tx_id",
    "timestamp",
    "day_idx",
    "src_bank",
    "dst_bank",
    "src_node",
    "dst_node",
    "amount_paid",
    "amount_received",
    "currency_paid",
    "currency_received",
    "payment_format",
    "is_self_loop",
    "is_cross_currency",
    "is_cross_bank",
    "label",
]


def read_transactions_csv(path: Path) -> pd.DataFrame:
    """Read the raw CSV into typed columns, sorted, with ``tx_id`` assigned.

    Sorting is stable on timestamp with original file order as the implicit tie-break, so
    ``tx_id`` is fully determined by the input file. Many rows share a minute, and an
    unstable sort would make ``tx_id`` differ between runs -- which would in turn make the
    streaming features (which walk rows in order) non-reproducible.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download the dataset per the README (Kaggle: "
            f"ealtman2019/ibm-transactions-for-anti-money-laundering-aml)."
        )

    # Read the header line as raw text rather than via pandas: the file has two columns
    # literally named 'Account', and pandas de-duplicates the second to 'Account.1', so a
    # parsed comparison would not be checking what the file actually says.
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    if header != RAW_HEADER:
        raise ValueError(
            f"Unexpected header in {path.name}.\n  expected: {RAW_HEADER}\n  found:    {header}"
        )

    df = pd.read_csv(path, header=0, names=RAW_COLUMNS, dtype=READ_DTYPES)
    LOGGER.info("read %d raw rows from %s", len(df), path.name)

    df["timestamp"] = _parse_timestamp_categories(df["timestamp"])
    df = df.sort_values("timestamp", kind="stable", ignore_index=True)
    df.insert(0, "tx_id", np.arange(len(df), dtype=np.int64))
    return df


def _parse_timestamp_categories(series: pd.Series) -> pd.Series:
    """Parse only the distinct timestamp strings, then fan them back out by code."""
    categories = pd.to_datetime(series.cat.categories, format=TIMESTAMP_FORMAT)
    parsed = categories.to_numpy()[series.cat.codes.to_numpy()]
    return pd.Series(parsed, index=series.index).astype("datetime64[s]")


def build_canonical(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, NodeIndex]:
    """Attach node ids and derived flags, then narrow to the canonical schema.

    Mutates ``df`` in place and returns it. Holding a separate canonical copy alongside
    the raw frame doubles peak memory for no benefit, and this pipeline runs on a machine
    where that margin does not exist.
    """
    node_index, src_node, dst_node = NodeIndex.build(
        df["src_bank"], df["src_acct"], df["dst_bank"], df["dst_acct"]
    )
    # Account strings are consumed by the interner and never needed again: src_node /
    # dst_node plus node_index.parquet recover them losslessly.
    df.drop(columns=["src_acct", "dst_acct"], inplace=True)
    gc.collect()

    df["src_node"] = src_node
    df["dst_node"] = dst_node
    del src_node, dst_node

    origin = pd.Timestamp(cfg.dataset.origin_date)
    day = (df["timestamp"].dt.normalize() - origin).dt.days
    if (day < 0).any():
        raise ValueError(
            f"Transactions predate origin_date {cfg.dataset.origin_date}; "
            f"earliest is {df['timestamp'].min()}"
        )
    df["day_idx"] = day.astype(np.int16)

    # Compared as aligned integer codes rather than strings: .astype(str) on a 5M-row
    # categorical materialises 5M Python objects and exhausts memory.
    df["is_self_loop"] = (df["src_node"] == df["dst_node"]).to_numpy()
    (cur_paid, cur_recv), _ = aligned_codes(df["currency_paid"], df["currency_received"])
    df["is_cross_currency"] = cur_paid != cur_recv
    (src_bank_code, dst_bank_code), _ = aligned_codes(df["src_bank"], df["dst_bank"])
    df["is_cross_bank"] = src_bank_code != dst_bank_code

    # Bank stays, as a categorical, because it is cheap and useful for error analysis.
    gc.collect()
    return df[CANONICAL_COLUMNS], node_index


def validate(df: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Assert the dataset is the expected variant and return a summary for the report.

    A row-count mismatch almost always means the wrong variant (LI-Small, HI-Medium) was
    downloaded, and every measured fact in architecture.md section 2 would then be wrong.
    """
    ds = cfg.dataset
    n, illicit = len(df), int(df["label"].sum())
    nodes = int(
        np.unique(np.concatenate([df["src_node"].to_numpy(), df["dst_node"].to_numpy()])).size
    )
    (src_bank_code, dst_bank_code), _ = aligned_codes(df["src_bank"], df["dst_bank"])
    n_banks = int(np.unique(np.concatenate([src_bank_code, dst_bank_code])).size)

    problems = []
    if n != ds.expected_rows:
        problems.append(f"row count {n} != expected {ds.expected_rows}")
    if illicit != ds.expected_illicit:
        problems.append(f"illicit count {illicit} != expected {ds.expected_illicit}")
    if nodes != ds.expected_nodes:
        problems.append(f"node count {nodes} != expected {ds.expected_nodes}")
    if problems:
        raise ValueError(
            f"Dataset does not match the {ds.variant} variant recorded in config:\n  "
            + "\n  ".join(problems)
        )

    for col in ("timestamp", "src_node", "dst_node", "amount_paid", "label"):
        if df[col].isna().any():
            raise ValueError(f"Nulls found in required column {col!r}")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Canonical table must be sorted by timestamp")

    summary = {
        "rows": n,
        "illicit": illicit,
        "illicit_rate": illicit / n,
        "nodes": nodes,
        "banks": n_banks,
        "first_timestamp": str(df["timestamp"].min()),
        "last_timestamp": str(df["timestamp"].max()),
        "n_days": int(df["day_idx"].max()) + 1,
        "self_loops": int(df["is_self_loop"].sum()),
        "self_loops_illicit": int(df.loc[df["is_self_loop"], "label"].sum()),
        "cross_currency": int(df["is_cross_currency"].sum()),
        "cross_bank": int(df["is_cross_bank"].sum()),
        "payment_formats": df["payment_format"].value_counts().to_dict(),
        "rows_per_day": df["day_idx"].value_counts().sort_index().to_dict(),
    }

    # Not fatal, but the graph layer excludes self-loops on the strength of this number,
    # so a large shift would mean that decision needs revisiting.
    LOGGER.info(
        "self-loops: %d rows (%.1f%%), %d illicit",
        summary["self_loops"],
        100 * summary["self_loops"] / n,
        summary["self_loops_illicit"],
    )
    return summary


def ingest_transactions(cfg: Config) -> tuple[pd.DataFrame, NodeIndex, dict[str, object]]:
    """Full ingest: read, canonicalise, validate."""
    df = read_transactions_csv(cfg.raw_dir / f"{cfg.dataset.variant}_Trans.csv")
    canonical, node_index = build_canonical(df, cfg)
    del df
    gc.collect()
    summary = validate(canonical, cfg)
    return canonical, node_index, summary
