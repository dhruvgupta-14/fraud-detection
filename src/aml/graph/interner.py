"""Account identity: ``(Bank, Account)`` -> contiguous ``int32`` node id.

Two reasons this exists rather than using the account string directly.

1. **Correctness.** Account numbers are not globally unique in this dataset: there are
   515,080 distinct account strings but 515,088 distinct ``(bank, account)`` pairs, so
   eight account numbers are reused across banks. Keying on the string alone would merge
   those accounts and silently corrupt their degree, volume and centrality features.

2. **Tractability.** A contiguous integer id is what lets the graph layer hold snapshots
   as scipy CSR matrices instead of a dict-of-dicts, which at 4.5M edges is roughly a
   20x difference in memory and a much larger one in PageRank runtime.

Ids are assigned in ascending order of the packed ``(bank_code, acct_code)`` key. That is
a pure function of the set of accounts in the file, so it is reproducible across runs and
-- unlike first-appearance ordering -- does not depend on row order at all.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("aml.interner")

NODE_INDEX_FILE = "node_index.parquet"


@dataclass
class NodeIndex:
    """Bidirectional map between ``(bank, acct)`` and ``node_id``."""

    frame: pd.DataFrame  # columns: node_id (int32), bank (str), acct (str)

    def __post_init__(self) -> None:
        expected = ["node_id", "bank", "acct"]
        if list(self.frame.columns) != expected:
            raise ValueError(f"NodeIndex frame must have columns {expected}, got {list(self.frame.columns)}")
        self._lookup: dict[tuple[str, str], int] | None = None

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def size(self) -> int:
        return len(self.frame)

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        src_bank: pd.Series,
        src_acct: pd.Series,
        dst_bank: pd.Series,
        dst_acct: pd.Series,
    ) -> tuple["NodeIndex", np.ndarray, np.ndarray]:
        """Intern both endpoints of every transaction in one pass.

        Returns the index plus the ``src_node`` / ``dst_node`` arrays, because computing
        them separately would mean factorizing twice over 10M values.
        """
        for name, s in (
            ("src_bank", src_bank),
            ("src_acct", src_acct),
            ("dst_bank", dst_bank),
            ("dst_acct", dst_acct),
        ):
            if s.isna().any():
                raise ValueError(f"{name} contains nulls; cannot intern")

        # Align the two sides onto shared category sets so a bank/account means the same
        # integer code whether it appeared as a sender or a receiver.
        bank_codes, bank_cats = aligned_codes(src_bank, dst_bank)
        acct_codes, acct_cats = aligned_codes(src_acct, dst_acct)
        n_acct = len(acct_cats)

        # Pack (bank, acct) into a single int64 so uniqueness is one integer operation.
        # 30.5K banks x 515K accounts is ~1.6e10, comfortably inside int64.
        n = len(src_bank)
        src_key = bank_codes[0].astype(np.int64) * n_acct + acct_codes[0]
        dst_key = bank_codes[1].astype(np.int64) * n_acct + acct_codes[1]
        del bank_codes, acct_codes

        # Deliberately not a factorize over the concatenated 10.2M endpoints: that builds
        # a 10M-entry hash table and is the single largest allocation in the whole ingest.
        # Reducing each side to its own ~500K uniques first, then merging, keeps peak
        # memory an order of magnitude lower for an identical result.
        uniques = np.union1d(np.unique(src_key), np.unique(dst_key))
        src_node = np.searchsorted(uniques, src_key).astype(np.int32)
        dst_node = np.searchsorted(uniques, dst_key).astype(np.int32)
        del src_key, dst_key
        gc.collect()

        frame = pd.DataFrame(
            {
                "node_id": np.arange(len(uniques), dtype=np.int32),
                "bank": pd.Categorical.from_codes(
                    (uniques // n_acct).astype(np.int32), categories=bank_cats
                ).astype(str),
                "acct": pd.Categorical.from_codes(
                    (uniques % n_acct).astype(np.int32), categories=acct_cats
                ).astype(str),
            }
        )
        LOGGER.info(
            "interned %d nodes from %d transactions (%d distinct account strings)",
            len(frame),
            n,
            frame["acct"].nunique(),
        )
        return cls(frame), src_node, dst_node

    # ------------------------------------------------------------------ lookup

    def encode(self, banks: pd.Series, accts: pd.Series) -> np.ndarray:
        """Map ``(bank, acct)`` pairs to node ids; unknown pairs become -1.

        Used to bring the Patterns.txt rows onto the same node space as the transaction
        table so the typology join can happen on integers rather than string tuples.
        """
        if self._lookup is None:
            self._lookup = {
                (b, a): int(n)
                for b, a, n in zip(
                    self.frame["bank"].to_numpy(),
                    self.frame["acct"].to_numpy(),
                    self.frame["node_id"].to_numpy(),
                )
            }
        lookup = self._lookup
        return np.fromiter(
            (lookup.get((b, a), -1) for b, a in zip(banks.astype(str), accts.astype(str))),
            dtype=np.int32,
            count=len(banks),
        )

    def decode(self, node_ids: np.ndarray | pd.Series) -> pd.DataFrame:
        """Recover ``(bank, acct)`` for node ids -- used by error analysis and the viewer."""
        return self.frame.set_index("node_id").loc[np.asarray(node_ids)].reset_index()

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(path, compression="snappy", index=False)
        return path

    @classmethod
    def load(cls, path: Path) -> "NodeIndex":
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run scripts/00_ingest.py first.")
        return cls(pd.read_parquet(path))


def aligned_codes(
    left: pd.Series, right: pd.Series
) -> tuple[tuple[np.ndarray, np.ndarray], pd.Index]:
    """Encode two series against a single shared category set.

    Shared by the interner and by the derived-flag comparisons in ingest. Comparing two
    5M-row categoricals via ``.astype(str)`` materialises 10M Python strings and runs the
    machine out of memory; comparing aligned integer codes is exact and allocates 40 MB.
    """
    left_cat = left if isinstance(left.dtype, pd.CategoricalDtype) else left.astype("category")
    right_cat = right if isinstance(right.dtype, pd.CategoricalDtype) else right.astype("category")
    union = pd.api.types.union_categoricals([left_cat, right_cat])
    categories = union.categories
    n = len(left_cat)
    return (union.codes[:n].astype(np.int64), union.codes[n:].astype(np.int64)), categories
