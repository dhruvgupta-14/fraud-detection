"""Model registry -- name to estimator factory (architecture.md 8.1).

The progression is a **controlled comparison**, not a search for the best number. Each rung
adds one idea, and each must be earned by a measured validation improvement:

    rung 1   logistic_regression, decision_tree   interpretable floor
    rung 2   random_forest                        variance reduction from bagging
    rung 3   lightgbm                             boosting; expected strongest single model

**Rung 4 (stacking) is deliberately not built yet.** It is the one component that needs a
temporal CV splitter wired into ``StackingClassifier`` and refits every base model k times,
and R5 already pre-commits to dropping it if the lift is under 0.005 AUPRC. Building it
before there is a baseline to compare against would be the definition of premature. It is
added in Phase 6 only if rungs 1-3 leave something on the table.

All hyperparameters come from config so a change is a diff on one file, and every estimator
receives the seed explicitly -- sklearn and LightGBM do not read the global numpy RNG
reliably.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from aml.config import Config

LOGGER = logging.getLogger("aml.models.registry")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    rung: int
    description: str
    build: Callable[[Config, float], Any]


def _logistic_regression(cfg: Config, pos_weight: float):
    """Linear floor. Needs the two things trees do not: imputation and scaling.

    Wrapped in a Pipeline rather than pre-transforming the matrix, so the imputer and scaler
    are fitted on train only and carried with the model. Fitting a scaler on the full frame
    before splitting is a quiet leak that a Pipeline makes structurally impossible.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(cfg.models.logistic_regression)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=cfg.seed,
                    n_jobs=-1,
                    **params,
                ),
            ),
        ]
    )


def _decision_tree(cfg: Config, pos_weight: float):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(
        class_weight="balanced", random_state=cfg.seed, **dict(cfg.models.decision_tree)
    )


def _random_forest(cfg: Config, pos_weight: float):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=cfg.seed,
        n_jobs=-1,
        **dict(cfg.models.random_forest),
    )


def _lightgbm(cfg: Config, pos_weight: float):
    """Expected strongest single model: native nulls, native imbalance handling.

    ``deterministic`` and ``force_row_wise`` come from config and are not optional --
    LightGBM's default multithreaded histogram build is not bit-reproducible, and
    reproducibility is explicitly judged (architecture.md 13.2).
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        scale_pos_weight=pos_weight,
        random_state=cfg.seed,
        n_jobs=-1,
        verbose=-1,
        **dict(cfg.models.lightgbm),
    )


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("logistic_regression", 1, "linear floor, scaled + imputed", _logistic_regression),
        ModelSpec("decision_tree", 1, "single tree, interpretable floor", _decision_tree),
        ModelSpec("random_forest", 2, "bagging -- variance reduction", _random_forest),
        ModelSpec("lightgbm", 3, "boosting -- native nulls and imbalance", _lightgbm),
    )
}

DEFAULT_ORDER = ["logistic_regression", "decision_tree", "random_forest", "lightgbm"]


def get_model(name: str, cfg: Config, pos_weight: float = 1.0):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    return MODELS[name].build(cfg, pos_weight)


def model_names(only: list[str] | None = None) -> list[str]:
    if not only:
        return list(DEFAULT_ORDER)
    unknown = set(only) - set(MODELS)
    if unknown:
        raise KeyError(f"unknown model(s) {sorted(unknown)}; have {sorted(MODELS)}")
    return [n for n in DEFAULT_ORDER if n in set(only)]
