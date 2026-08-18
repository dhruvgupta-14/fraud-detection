"""Report figures (architecture.md 10.1).

One function per exhibit, every figure written to ``artifacts/figures/`` at 300 dpi. No
hand-made charts: a figure that cannot be regenerated from the pipeline cannot be trusted,
and "we regenerated it" is what makes the ablation believable.

    F1  PR curves, all three arms          the headline
    F2  AUPRC by model rung and arm        the progression
    F3  Per-typology recall at a shared alert budget
    F4  Alerts/day vs recall trade-off     the business framing
    F5  SHAP importance by feature group   the mechanism
    F6  Walk-forward AUPRC, retrained vs frozen
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless run; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOGGER = logging.getLogger("aml.evaluate.figures")

# One colour per arm, used consistently across every figure so the reader learns it once.
ARM_COLOURS = {"E1": "#a0aec0", "E3": "#4299e1", "E2": "#2b6cb0"}
ARM_LABELS = {
    "E1": "E1 · tabular only",
    "E3": "E3 · + account counters",
    "E2": "E2 · + graph structure",
}
ACCENT, WARN = "#2b6cb0", "#c05621"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------------------


def f1_pr_curves(metrics: dict, out_dir: Path, model: str = "lightgbm") -> Path:
    """F1 -- precision-recall curves for all three arms. The headline exhibit."""
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for arm in ("E1", "E3", "E2"):
        if arm not in metrics:
            continue
        test = metrics[arm][model]["test"]
        curve = test["pr_curve"]
        lo, hi = test["auprc_ci95"]
        ax.plot(
            curve["recall"],
            curve["precision"],
            color=ARM_COLOURS[arm],
            lw=2.0 if arm == "E2" else 1.5,
            label=f"{ARM_LABELS[arm]}  AUPRC {test['auprc']:.4f} [{lo:.3f}, {hi:.3f}]",
        )

    prevalence = metrics["E2"][model]["test"]["prevalence"]
    ax.axhline(prevalence, color="k", ls=":", lw=1)
    ax.annotate(
        f"random ranker = prevalence = {prevalence:.4%}",
        xy=(0.55, prevalence),
        xytext=(0.4, prevalence * 3),
        fontsize=7.5,
        color="k",
    )

    ax.set_yscale("log")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision (log scale)")
    ax.set_title("F1 · Precision–recall, test window (days 7–9)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    return _save(fig, out_dir, "F1_pr_curves.png")


def f2_auprc_by_rung(metrics: dict, out_dir: Path) -> Path:
    """F2 -- AUPRC by model rung, grouped by arm, with bootstrap intervals."""
    _style()
    rungs = ["logistic_regression", "decision_tree", "random_forest", "lightgbm"]
    labels = ["LogReg", "Decision tree", "Random forest", "LightGBM"]
    arms = [a for a in ("E1", "E3", "E2") if a in metrics]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / len(arms)
    x = np.arange(len(rungs))

    for i, arm in enumerate(arms):
        values, errs = [], [[], []]
        for rung in rungs:
            test = metrics[arm][rung]["test"]
            values.append(test["auprc"])
            lo, hi = test["auprc_ci95"]
            errs[0].append(max(test["auprc"] - lo, 0))
            errs[1].append(max(hi - test["auprc"], 0))
        ax.bar(
            x + i * width - 0.4 + width / 2,
            values,
            width,
            yerr=errs,
            capsize=2.5,
            color=ARM_COLOURS[arm],
            label=ARM_LABELS[arm],
            error_kw={"lw": 0.9},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("test AUPRC")
    ax.set_title("F2 · AUPRC by model rung and feature arm (95 % bootstrap CI)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, "F2_auprc_by_rung.png")


def f3_typology_recall(comparison: pd.DataFrame, out_dir: Path, budget_per_day: float) -> Path:
    """F3 -- per-typology recall at a shared alert budget, with Wilson intervals.

    Plotted at a fixed *budget* rather than a fixed recall: at each arm's own 90 %-recall
    threshold every arm scores 0.97-1.00 on every family, so that version of the chart is
    flat by construction (see evaluate/typology.py).
    """
    _style()
    arms = [a for a in ("E1", "E3", "E2") if a in set(comparison["arm"])]
    order = (
        comparison[comparison["arm"] == arms[-1]]
        .sort_values("n_positives", ascending=False)["typology"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    height = 0.8 / len(arms)
    y = np.arange(len(order))

    for i, arm in enumerate(arms):
        sub = comparison[comparison["arm"] == arm].set_index("typology").loc[order]
        errs = [
            (sub["recall"] - sub["recall_lo"]).clip(lower=0),
            (sub["recall_hi"] - sub["recall"]).clip(lower=0),
        ]
        ax.barh(
            y + i * height - 0.4 + height / 2,
            sub["recall"],
            height,
            xerr=errs,
            capsize=2,
            color=ARM_COLOURS[arm],
            label=ARM_LABELS[arm],
            error_kw={"lw": 0.8},
        )

    counts = comparison[comparison["arm"] == arms[-1]].set_index("typology").loc[order]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  (n={int(counts.loc[t, 'n_positives'])})" for t in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("recall at a shared alert budget")
    ax.set_title(
        f"F3 · Per-typology recall at {budget_per_day:,.0f} alerts/day\n"
        f"only 913 of 1,495 test positives carry an annotation (61 %)",
        fontsize=9.5,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, out_dir, "F3_typology_recall.png")


def f4_alert_budget_curve(
    predictions: dict[str, pd.DataFrame], out_dir: Path, test_days: int
) -> Path:
    """F4 -- recall against alerts per day. The business framing (9.2).

    Converts an abstract AUPRC gap into a staffing statement: at any alert volume an
    analyst team can actually work, how much laundering does each arm catch?
    """
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for arm in ("E1", "E3", "E2"):
        if arm not in predictions:
            continue
        rows = predictions[arm]
        rows = rows[rows["split"] == "test"]
        scores = rows["score"].to_numpy()
        labels = rows["label"].to_numpy()
        order = np.argsort(-scores)
        caught = np.cumsum(labels[order])
        recall = caught / labels.sum()
        alerts = np.arange(1, len(scores) + 1) / test_days
        step = max(len(alerts) // 3000, 1)
        ax.plot(
            alerts[::step],
            recall[::step],
            color=ARM_COLOURS[arm],
            lw=2.0 if arm == "E2" else 1.5,
            label=ARM_LABELS[arm],
        )

    ax.axhline(0.9, color="k", ls=":", lw=1)
    ax.annotate("90 % recall target", xy=(1e3, 0.91), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("alerts per day (log scale)")
    ax.set_ylabel("recall")
    ax.set_title("F4 · What each arm costs an analyst team")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, out_dir, "F4_alert_budget.png")


def f5_shap_by_group(per_group: pd.DataFrame, per_column: pd.DataFrame, out_dir: Path,
                     n_sampled: int) -> Path:
    """F5 -- SHAP importance aggregated by feature group, plus the top columns."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax = axes[0]
    colours = {"tabular": "#a0aec0", "streaming": "#4299e1",
               "structural": "#2b6cb0", "motif": "#c05621"}
    ax.barh(
        per_group["group"],
        per_group["share"],
        color=[colours.get(g, "#888") for g in per_group["group"]],
    )
    for i, row in per_group.iterrows():
        ax.text(row["share"] + 0.005, i, f"{row['share']:.1%} ({int(row['n_columns'])} cols)",
                va="center", fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, per_group["share"].max() * 1.35)
    ax.set_xlabel("share of total mean |SHAP|")
    ax.set_title("F5a · Attribution by feature group")

    ax = axes[1]
    top = per_column.head(15).iloc[::-1]
    ax.barh(top["column"], top["mean_abs_shap"],
            color=[colours.get(g, "#888") for g in top["group"]])
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("F5b · Top 15 individual columns")
    ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(f"F5 · SHAP on the E2 LightGBM model ({n_sampled:,} sampled test rows)",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "F5_shap_by_group.png")


def f6_walkforward(results: pd.DataFrame, out_dir: Path, summary: dict | None = None) -> Path:
    """F6 -- walk-forward AUPRC per block: retrained versus frozen.

    The title is set from the measured verdict rather than from the question we set out to
    ask. §9.5 framed this as a decay study, but on this data the frozen arm is flat and the
    gap comes from the retrained arm accumulating data — so a chart titled "does the model
    decay?" would invite exactly the wrong reading of its own contents.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for arm, colour, marker in (("retrained", ACCENT, "o"), ("frozen", WARN, "s")):
        sub = results[results["arm"] == arm].sort_values("block")
        errs = [
            (sub["auprc"] - sub["auprc_lo"]).clip(lower=0),
            (sub["auprc_hi"] - sub["auprc"]).clip(lower=0),
        ]
        ax.errorbar(
            sub["block"], sub["auprc"], yerr=errs, color=colour, marker=marker,
            capsize=3, lw=1.6, label=arm, ms=5,
        )

    declined = bool(summary.get("frozen_declined")) if summary else None
    if declined is False:
        title = "F6 · Retrained vs frozen — the gap is training-data volume, not decay"
        note = (
            "The frozen arm is flat across all four blocks (intervals overlap): it does not\n"
            "degrade. The retrained arm improves as account history and graph structure\n"
            "accumulate — block 1 trains on days 0–1, where structure is still cold-start."
        )
    elif declined is True:
        title = "F6 · Model decay: the frozen arm degrades, retraining recovers it"
        note = "The frozen arm declines with non-overlapping intervals."
    else:
        title = "F6 · Walk-forward: retrained versus frozen"
        note = ""

    ax.set_xlabel("evaluation block (2 days each)")
    ax.set_ylabel("test AUPRC")
    ax.set_xticks(sorted(results["block"].unique()))
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    if note:
        fig.text(0.5, -0.09, note, ha="center", fontsize=7.5, color="#444")
    return _save(fig, out_dir, "F6_walkforward.png")
