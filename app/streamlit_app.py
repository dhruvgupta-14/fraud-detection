"""AML alert triage viewer (architecture.md 10.2).

**This is a results viewer over batch predictions, not a monitoring system.** It reads a
precomputed bundle produced by `scripts/06_demo_bundle.py` from artifacts the analysis
pipeline already wrote. It does not score transactions live, does not connect to anything,
and holds no state. The report describes it in exactly those terms.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from aml.config import load_config  # noqa: E402
from aml.graph.interner import NODE_INDEX_FILE  # noqa: E402
from aml.graph.snapshots import Snapshot, open_store  # noqa: E402
from aml.io import ArtifactStore  # noqa: E402
from aml.viz.demo import (  # noqa: E402
    ALERTS_FILE,
    BUNDLE_STAGE,
    MAX_NEIGHBOURS,
    SHAP_FILE,
    SUMMARY_FILE,
    ego_edges,
)

DEMO_ARM = "ablation_graph"
BUNDLE_SECTIONS = ("dataset", "time", "graph", "features", "models")

RISK = "#c0392b"
SAFE = "#2b6cb0"
MUTED = "#a0aec0"
GROUP_COLOURS = {
    "tabular": "#a0aec0",
    "streaming": "#4299e1",
    "structural": "#2b6cb0",
    "motif": "#c05621",
}

st.set_page_config(page_title="AML Alert Triage", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _store():
    cfg = load_config(DEMO_ARM)
    return cfg, ArtifactStore(cfg)


@st.cache_data(show_spinner="Loading alert bundle…")
def load_bundle():
    cfg, store = _store()
    bundle = store.stage(BUNDLE_STAGE, BUNDLE_SECTIONS)
    if not bundle.exists(ALERTS_FILE):
        return None, None, None
    return (
        bundle.read_frame(ALERTS_FILE),
        bundle.read_frame(SHAP_FILE),
        bundle.read_json(SUMMARY_FILE),
    )


@st.cache_resource(show_spinner="Loading graph snapshot…")
def load_snapshot():
    """The last snapshot in the modelling window -- the graph the test rows were scored on."""
    cfg, store = _store()
    return Snapshot.load(open_store(store), cfg.time.max_day)


@st.cache_data(show_spinner=False)
def load_node_names() -> pd.Series:
    _, store = _store()
    index = store.read_processed(NODE_INDEX_FILE)
    return index.set_index("node_id")["acct"]


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------


def project_metrics(summary: dict) -> None:
    """The headline ablation, stated from measured output rather than typed in."""
    arms = summary["arms"]
    st.markdown("#### Model performance — the ablation behind this viewer")

    cols = st.columns(4)
    labels = {
        "E1": ("Tabular only", "conventional row-wise model"),
        "E3": ("+ account counters", "per-account history, no graph"),
        "E2": ("+ graph structure", "the deployed arm"),
    }
    for col, arm in zip(cols, ("E1", "E3", "E2")):
        if arm not in arms:
            continue
        title, caption = labels[arm]
        values = arms[arm]
        delta = None
        if arm == "E2" and "E3" in arms:
            delta = f"+{values['auprc'] - arms['E3']['auprc']:.4f} vs E3"
        col.metric(f"{arm} · {title}", f"{values['auprc']:.4f} AUPRC", delta)
        col.caption(caption)

    reduction = summary.get("alert_reduction_e3_to_e2")
    if reduction:
        cols[3].metric("Alert reduction", f"{reduction:.0%}", "same 90% recall")
        cols[3].caption(
            f"{arms['E3']['alerts_per_day']:,.0f} → {arms['E2']['alerts_per_day']:,.0f} alerts/day"
        )

    st.caption(
        f"Test window prevalence {arms['E2']['prevalence']:.4%} — a random ranker scores "
        f"AUPRC = prevalence. Graph features cut the analyst queue by "
        f"{reduction:.0%} at identical recall."
        if reduction
        else ""
    )


def risk_panel(row: pd.Series, threshold: float) -> None:
    left, right = st.columns([1, 2])

    with left:
        flagged = bool(row["alert"])
        st.markdown(
            f"<div style='padding:1rem;border-radius:8px;background:"
            f"{'#fdeaea' if flagged else '#eef4fb'};border-left:6px solid "
            f"{RISK if flagged else SAFE}'>"
            f"<div style='font-size:0.8rem;color:#555'>ALERT STATUS</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:"
            f"{RISK if flagged else SAFE}'>{'⚠ ALERT' if flagged else '✓ CLEAR'}</div>"
            f"<div style='font-size:0.78rem;color:#555;margin-top:.35rem'>"
            f"threshold {threshold:.4f} · chosen on validation at 90% recall</div></div>",
            unsafe_allow_html=True,
        )
        st.metric("Risk score", f"{row['score']:.4f}")
        st.progress(min(float(row["score"]), 1.0))
        st.caption(f"Rank {int(row['rank'])} of {int(row['n_bundled'])} bundled transactions")

    with right:
        st.markdown("**Transaction**")
        details = {
            "Transaction id": int(row["tx_id"]),
            "Timestamp": str(row["timestamp"]),
            "Sender": f"{row['src_acct']} ({row['src_bank']})",
            "Receiver": f"{row['dst_acct']} ({row['dst_bank']})",
            "Amount paid": f"{row['amount_paid']:,.2f} {row['currency_paid']}",
            "Payment format": row["payment_format"],
            "Cross-bank": "yes" if row["is_cross_bank"] else "no",
        }
        st.dataframe(
            pd.DataFrame({"field": list(details), "value": [str(v) for v in details.values()]}),
            hide_index=True,
            width="stretch",
        )

        # Ground truth is shown because this is a demo over a labelled test set, not a
        # production queue. Hiding it would make the viewer look more capable than it is.
        truth = "ILLICIT" if row["label"] == 1 else "licit"
        typology = row["typology"] if isinstance(row["typology"], str) else None
        note = f"Ground truth: **{truth}**"
        if typology:
            note += f" · laundering typology **{typology}** (attempt {int(row['attempt_id'])})"
        elif row["label"] == 1:
            note += " · not annotated with a typology (62% of illicit rows are)"
        st.info(note, icon="🎯")


def reasons_panel(shap_rows: pd.DataFrame, top_n: int = 8) -> None:
    st.markdown("#### Why this score")
    if shap_rows.empty:
        st.caption("No attribution available for this transaction.")
        return

    top = shap_rows.reindex(shap_rows["shap"].abs().sort_values(ascending=False).index).head(top_n)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(top) + 0.8))
    ax.barh(
        top["feature"],
        top["shap"],
        color=[GROUP_COLOURS.get(g, MUTED) for g in top["group"]],
    )
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("SHAP contribution to the risk score")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.caption(
        "Bars to the right push the score up. Colours are feature groups: "
        "grey tabular · light blue account counters · dark blue graph structure · orange motifs."
    )

    display = top.iloc[::-1][["feature", "group", "value", "shap"]].copy()
    display["direction"] = np.where(display["shap"] > 0, "raises risk", "lowers risk")
    st.dataframe(
        display.rename(columns={"value": "feature value", "shap": "contribution"}),
        hide_index=True,
        width="stretch",
    )


def graph_panel(snapshot, row: pd.Series, names: pd.Series) -> None:
    st.markdown("#### Transaction neighbourhood")
    side = st.radio(
        "Centre the network on",
        ["Sender", "Receiver"],
        horizontal=True,
        label_visibility="collapsed",
    )
    node = int(row["src_node"] if side == "Sender" else row["dst_node"])

    edges = ego_edges(snapshot, node, MAX_NEIGHBOURS)
    if edges.empty:
        st.warning(
            "This account had no transactions in the graph snapshot the model read "
            "(a dormant account, or its first ever activity)."
        )
        return

    counterparty = int(row["dst_node"] if side == "Sender" else row["src_node"])
    graph = nx.DiGraph()
    for _, edge in edges.iterrows():
        graph.add_edge(int(edge["source"]), int(edge["target"]), weight=edge["weight"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pos = nx.spring_layout(graph, seed=42, k=0.9)

    colours, sizes = [], []
    for n in graph.nodes():
        if n == node:
            colours.append(RISK if row["alert"] else SAFE)
            sizes.append(1100)
        elif n == counterparty:
            colours.append("#dd6b20")
            sizes.append(700)
        else:
            colours.append(MUTED)
            sizes.append(360)

    widths = [0.6 + 1.8 * min(graph[u][v]["weight"], 8) / 8 for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, edge_color="#8899aa",
                           arrows=True, arrowsize=11, alpha=0.75)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colours, node_size=sizes,
                           edgecolors="white", linewidths=1.4)
    nx.draw_networkx_labels(
        graph, pos, ax=ax, font_size=7,
        labels={n: str(names.get(n, n))[:8] for n in graph.nodes()},
    )
    ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    hidden = int(edges["hidden"].max())
    legend = (
        f"Centre = the {side.lower()} account · orange = the counterparty on this "
        f"transaction · grey = other counterparties. Edge width is transaction count."
    )
    if hidden:
        legend += (
            f" **{hidden} further counterparties are not drawn** — display is capped at "
            f"{MAX_NEIGHBOURS} per direction for readability."
        )
    st.caption(legend)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


def main() -> None:
    st.title("🔍 AML Alert Triage")
    st.caption(
        "A results viewer over batch predictions from the IBM HI-Small graph-feature model. "
        "It reads saved outputs — it does not score transactions live."
    )

    alerts, shap_long, summary = load_bundle()
    if alerts is None:
        st.error(
            "No demo bundle found. Build it first:\n\n"
            "```\npython scripts/06_demo_bundle.py\n```"
        )
        st.stop()

    project_metrics(summary)
    st.divider()

    with st.sidebar:
        st.header("Select a transaction")
        only_alerts = st.checkbox("Alerts only", value=True)
        only_illicit = st.checkbox("Known illicit only", value=False)

        view = alerts
        if only_alerts:
            view = view[view["alert"]]
        if only_illicit:
            view = view[view["label"] == 1]
        if view.empty:
            st.warning("No transactions match these filters.")
            st.stop()

        st.caption(f"{len(view):,} of {len(alerts):,} transactions match")
        options = view["tx_id"].tolist()
        chosen = st.selectbox(
            "Transaction",
            options,
            format_func=lambda t: (
                f"#{int(t)} · {view.loc[view.tx_id == t, 'score'].iloc[0]:.3f}"
                f"{' · illicit' if view.loc[view.tx_id == t, 'label'].iloc[0] == 1 else ''}"
            ),
        )
        st.divider()
        st.caption(
            f"Bundle holds the top {int(summary['n_above_threshold']):,} alerts plus a "
            f"random sample of lower-scored rows, so the viewer can show confident "
            f"negatives too."
        )

    row = alerts.loc[alerts["tx_id"] == chosen].iloc[0].copy()
    row["n_bundled"] = len(alerts)

    risk_panel(row, summary["threshold"])
    st.divider()

    left, right = st.columns([1, 1])
    with left:
        reasons_panel(shap_long.loc[shap_long["tx_id"] == chosen])
    with right:
        graph_panel(load_snapshot(), row, load_node_names())


if __name__ == "__main__":
    main()
