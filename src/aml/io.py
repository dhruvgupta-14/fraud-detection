"""Artifact persistence: content-addressed directories, atomic writes, cache reuse.

Two storage areas, matching Appendix A of architecture.md:

* ``data/processed/`` -- canonical ingest outputs. One version, no hash, because they
  depend only on the raw files and the dataset section of the config.
* ``artifacts/<stage>/<config-hash>/`` -- everything downstream. Hashed, because the
  ablation arms and the lookback sweep produce genuinely different content from the
  same code and must not collide.

Writes are atomic (temp file + ``os.replace``). The feature stage runs for ~25 minutes;
a Ctrl-C halfway through must not leave a truncated parquet that the next run happily
treats as a valid cache hit.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from aml.config import Config

LOGGER = logging.getLogger("aml")

# Sidecar written into every hashed stage directory so a bare hash on disk can be
# traced back to the settings that produced it without re-deriving it.
CONFIG_SIDECAR = "_config.json"


def setup_logging(level: int = logging.INFO) -> None:
    """Console logging with timings. Called by every script in scripts/."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@contextmanager
def timed(label: str, logger: logging.Logger = LOGGER) -> Iterator[None]:
    """Log wall-clock for a block, so the stage budgets in architecture.md stay honest."""
    start = time.perf_counter()
    logger.info("START  %s", label)
    try:
        yield
    finally:
        logger.info("DONE   %s (%.1fs)", label, time.perf_counter() - start)


@contextmanager
def _atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temp path alongside ``path`` and move it into place on clean exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


# --------------------------------------------------------------------------------------
# Stage store
# --------------------------------------------------------------------------------------


class StageStore:
    """Read/write access to one hashed stage directory."""

    def __init__(self, root: Path, stage: str, config_hash: str) -> None:
        self.root = root
        self.stage = stage
        self.config_hash = config_hash
        self.dir = root / stage / config_hash

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StageStore({self.stage!r}, hash={self.config_hash!r})"

    def path(self, name: str) -> Path:
        return self.dir / name

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    # ------------------------------------------------------------------ frames

    def write_frame(self, df: pd.DataFrame, name: str) -> Path:
        """Write a parquet file atomically.

        ``index=False`` throughout: keys such as ``tx_id`` are carried as real columns so
        that a join never silently depends on index alignment.
        """
        target = self.path(name)
        with _atomic_path(target) as tmp:
            df.to_parquet(tmp, compression="snappy", index=False)
        LOGGER.info(
            "wrote %s (%d rows x %d cols, %s)",
            target.relative_to(self.root.parent),
            len(df),
            df.shape[1],
            human_bytes(target.stat().st_size),
        )
        return target

    def read_frame(self, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        target = self.path(name)
        if not target.is_file():
            raise FileNotFoundError(f"Missing artifact {target}. Run the producing stage first.")
        return pd.read_parquet(target, columns=list(columns) if columns else None)

    # ------------------------------------------------------------------ json

    def write_json(self, obj: Any, name: str) -> Path:
        target = self.path(name)
        with _atomic_path(target) as tmp:
            tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return target

    def read_json(self, name: str) -> Any:
        target = self.path(name)
        if not target.is_file():
            raise FileNotFoundError(f"Missing artifact {target}. Run the producing stage first.")
        return json.loads(target.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ arrays

    def write_npz(self, name: str, **arrays: Any) -> Path:
        """Compressed .npz -- used for CSR snapshot components in Phase 2."""
        import numpy as np

        target = self.path(name)
        with _atomic_path(target) as tmp:
            np.savez_compressed(tmp, **arrays)
            # numpy appends .npz when the target lacks the suffix; normalise it back.
            if not tmp.exists() and tmp.with_suffix(tmp.suffix + ".npz").exists():
                tmp.with_suffix(tmp.suffix + ".npz").replace(tmp)
        return target

    def read_npz(self, name: str) -> Any:
        import numpy as np

        target = self.path(name)
        if not target.is_file():
            raise FileNotFoundError(f"Missing artifact {target}. Run the producing stage first.")
        return np.load(target, allow_pickle=False)

    # ------------------------------------------------------------------ caching

    def cached_frame(
        self, name: str, build: Callable[[], pd.DataFrame], force: bool = False
    ) -> pd.DataFrame:
        """Return the cached frame, or build, persist and return it.

        A stage that has already run with identical config is a no-op. ``force=True``
        rebuilds -- the escape hatch for when the code changed but the config did not,
        which the hash cannot detect on its own.
        """
        if not force and self.exists(name):
            LOGGER.info("cache HIT  %s/%s/%s", self.stage, self.config_hash, name)
            return self.read_frame(name)
        LOGGER.info("cache MISS %s/%s/%s -- building", self.stage, self.config_hash, name)
        with timed(f"{self.stage}:{name}"):
            df = build()
        self.write_frame(df, name)
        return df


class ArtifactStore:
    """Entry point for all persistence. Constructed once per script run."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.artifacts_root = cfg.artifacts_dir
        self.processed_root = cfg.processed_dir

    # ------------------------------------------------------------------ processed

    def processed(self, name: str) -> Path:
        """Path to a canonical ingest artifact (unhashed)."""
        self.processed_root.mkdir(parents=True, exist_ok=True)
        return self.processed_root / name

    def write_processed(self, df: pd.DataFrame, name: str) -> Path:
        target = self.processed(name)
        with _atomic_path(target) as tmp:
            df.to_parquet(tmp, compression="snappy", index=False)
        LOGGER.info(
            "wrote %s (%d rows x %d cols, %s)",
            target.relative_to(self.cfg.root),
            len(df),
            df.shape[1],
            human_bytes(target.stat().st_size),
        )
        return target

    def read_processed(self, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        target = self.processed(name)
        if not target.is_file():
            raise FileNotFoundError(
                f"Missing {target}. Run scripts/00_ingest.py first."
            )
        return pd.read_parquet(target, columns=list(columns) if columns else None)

    def has_processed(self, name: str) -> bool:
        return self.processed(name).is_file()

    def write_processed_json(self, obj: Any, name: str) -> Path:
        target = self.processed(name)
        with _atomic_path(target) as tmp:
            tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
        LOGGER.info("wrote %s", target.relative_to(self.cfg.root))
        return target

    def read_processed_json(self, name: str) -> Any:
        target = self.processed(name)
        if not target.is_file():
            raise FileNotFoundError(f"Missing {target}. Run scripts/00_ingest.py first.")
        return json.loads(target.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ stages

    def stage(self, stage: str, sections: Sequence[str]) -> StageStore:
        """Open a hashed stage directory, recording the config subset that keys it.

        ``sections`` are the config sections this stage genuinely depends on. Naming them
        narrowly is what keeps an unrelated hyperparameter edit from invalidating an
        expensive upstream artifact.
        """
        config_hash = self.cfg.hash_for(*sections)
        store = StageStore(self.artifacts_root, stage, config_hash)
        store.dir.mkdir(parents=True, exist_ok=True)

        sidecar = store.path(CONFIG_SIDECAR)
        if not sidecar.is_file():
            store.write_json(
                {
                    "stage": stage,
                    "sections": list(sections),
                    "config": self.cfg.describe(*sections),
                    "experiment": self.cfg.experiment,
                },
                CONFIG_SIDECAR,
            )
        return store
