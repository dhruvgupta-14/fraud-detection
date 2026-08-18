"""Guards on the demo bundle helpers (architecture.md 10.2).

The viewer is presentation, so these tests are deliberately thin -- they cover the two
things that would make the demo *lie* rather than merely look wrong:

* an alert table whose ``alert`` flag disagrees with the threshold, so the screen shows a
  status the model did not produce;
* an ego network silently truncated to a readable size, so a hub account looks small.

Rendering is not tested. A chart that looks bad is obvious in the demo; a chart that shows
12 of 14,230 counterparties without saying so is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml.config import load_config
from aml.graph.snapshots import build_snapshot
from aml.viz.demo import MAX_NEIGHBOURS, build_alert_table, ego_edges

N_NODES = 8


def make_predictions(scores, labels) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": np.arange(len(scores), dtype=np.int64),
            "split": "test",
            "score": np.asarray(scores, dtype=float),
            "label": np.asarray(labels, dtype=np.int8),
        }
    )


def make_transactions(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": np.arange(n, dtype=np.int64),
            "timestamp": pd.to_datetime("2022-09-08") + pd.to_timedelta(np.arange(n), unit="m"),
            "day_idx": np.full(n, 7, dtype=np.int16),
            "src_node": (np.arange(n, dtype=np.int32) % 4),
            "dst_node": ((np.arange(n, dtype=np.int32) + 1) % 4),
            "src_bank": "BankA",
            "dst_bank": "BankB",
            "amount_paid": np.full(n, 500.0),
            "amount_received": np.full(n, 500.0),
            "currency_paid": "US Dollar",
            "payment_format": "Wire",
            "is_self_loop": np.zeros(n, dtype=bool),
            "is_cross_bank": np.ones(n, dtype=bool),
        }
    )


def make_node_index(n: int = N_NODES) -> pd.DataFrame:
    return pd.DataFrame(
        {"node_id": np.arange(n, dtype=np.int32),
         "bank": [f"Bank{i}" for i in range(n)],
         "acct": [f"ACCT{i:03d}" for i in range(n)]}
    )


def make_typology() -> pd.DataFrame:
    return pd.DataFrame({"tx_id": [0], "attempt_id": [7], "typology": ["CYCLE"]}).astype(
        {"tx_id": "int64", "attempt_id": "int32"}
    )


# --------------------------------------------------------------------------------------
# Alert table
# --------------------------------------------------------------------------------------


def test_alert_flag_matches_the_threshold():
    """The status shown on screen must be the one the model's threshold produces."""
    predictions = make_predictions([0.9, 0.5, 0.1, 0.05], [1, 0, 0, 0])
    table = build_alert_table(
        predictions, make_transactions(4), make_typology(), make_node_index(),
        threshold=0.4, top_n=4, n_sample=0,
    )
    by_id = table.set_index("tx_id")
    assert bool(by_id.loc[0, "alert"]) and bool(by_id.loc[1, "alert"])
    assert not bool(by_id.loc[2, "alert"]) and not bool(by_id.loc[3, "alert"])


def test_table_is_ranked_by_score_descending():
    predictions = make_predictions([0.1, 0.9, 0.5], [0, 1, 0])
    table = build_alert_table(
        predictions, make_transactions(3), make_typology(), make_node_index(),
        threshold=0.4, top_n=3, n_sample=0,
    )
    assert table["score"].is_monotonic_decreasing
    assert table["rank"].tolist() == [1, 2, 3]


def test_bundle_includes_lower_scored_rows_when_sampled():
    """A table of only top alerts makes every row look like a hit; the demo needs
    confident negatives to be honest about what the model does."""
    predictions = make_predictions(np.linspace(1.0, 0.0, 20), [1] + [0] * 19)
    table = build_alert_table(
        predictions, make_transactions(20), make_typology(), make_node_index(),
        threshold=0.9, top_n=3, n_sample=5,
    )
    assert len(table) == 8
    assert (~table["alert"]).any()


def test_account_names_are_resolved_for_display():
    predictions = make_predictions([0.9], [1])
    table = build_alert_table(
        predictions, make_transactions(1), make_typology(), make_node_index(),
        threshold=0.5, top_n=1, n_sample=0,
    )
    assert table.loc[0, "src_acct"] == "ACCT000"
    assert table.loc[0, "dst_acct"] == "ACCT001"


def test_typology_is_attached_where_annotated():
    predictions = make_predictions([0.9, 0.8], [1, 1])
    table = build_alert_table(
        predictions, make_transactions(2), make_typology(), make_node_index(),
        threshold=0.5, top_n=2, n_sample=0,
    )
    by_id = table.set_index("tx_id")
    assert by_id.loc[0, "typology"] == "CYCLE"
    assert pd.isna(by_id.loc[1, "typology"])  # unannotated stays visibly unannotated


# --------------------------------------------------------------------------------------
# Ego network
# --------------------------------------------------------------------------------------


def ego_snapshot():
    """Node 0 sends to 1..5 and receives from 6; node 7 is dormant."""
    rows = [(0, i) for i in range(1, 6)] + [(6, 0)]
    df = pd.DataFrame(rows, columns=["src_node", "dst_node"])
    n = len(df)
    frame = pd.DataFrame(
        {
            "day_idx": np.full(n, 0, dtype=np.int16),
            "src_node": df["src_node"].to_numpy(np.int32),
            "dst_node": df["dst_node"].to_numpy(np.int32),
            "amount_paid": np.full(n, 100.0),
            "currency_paid": "US Dollar",
            "timestamp": pd.to_datetime("2022-09-01") + pd.to_timedelta(np.arange(n), unit="s"),
            "is_self_loop": np.zeros(n, dtype=bool),
        }
    )
    cfg = load_config(
        overrides={"time": {"max_day": 2, "train_days": [0, 0], "val_days": [1, 1], "test_days": [2, 2]}}
    )
    return build_snapshot(frame, 0, N_NODES, cfg)


def test_ego_edges_cover_both_directions():
    edges = ego_edges(ego_snapshot(), node=0)
    assert set(edges["direction"]) == {"sends to", "receives from"}
    assert len(edges) == 6  # 5 out + 1 in


def test_ego_edges_orient_correctly():
    edges = ego_edges(ego_snapshot(), node=0)
    outgoing = edges[edges["direction"] == "sends to"]
    incoming = edges[edges["direction"] == "receives from"]
    assert (outgoing["source"] == 0).all()
    assert (incoming["target"] == 0).all()


def test_ego_edges_report_how_many_were_hidden():
    """A hub has 14,230 counterparties. Drawing 12 without saying so would make the demo
    misrepresent the graph, which is worse than an ugly chart."""
    edges = ego_edges(ego_snapshot(), node=0, max_neighbours=2)
    assert len(edges) == 3  # 2 out + 1 in
    assert int(edges["hidden"].max()) == 3  # 5 outgoing, 2 shown


def test_nothing_is_hidden_when_under_the_cap():
    edges = ego_edges(ego_snapshot(), node=0, max_neighbours=MAX_NEIGHBOURS)
    assert int(edges["hidden"].max()) == 0


def test_dormant_account_yields_an_empty_neighbourhood():
    """Node 7 has no edges; the app shows an explanation rather than an empty chart."""
    assert ego_edges(ego_snapshot(), node=7).empty


def test_ego_edges_keep_the_strongest_relationships_under_truncation():
    """Truncation must drop the weakest edges, not an arbitrary slice."""
    snapshot = ego_snapshot()
    edges = ego_edges(snapshot, node=0, max_neighbours=1)
    outgoing = edges[edges["direction"] == "sends to"]
    assert len(outgoing) == 1
    assert outgoing["weight"].iloc[0] == max(
        ego_edges(snapshot, node=0)["weight"][:5]
    )


# --------------------------------------------------------------------------------------
# App smoke test
# --------------------------------------------------------------------------------------


def _bundle_exists() -> bool:
    from aml.io import ArtifactStore
    from aml.viz.demo import ALERTS_FILE, BUNDLE_STAGE

    cfg = load_config("ablation_graph")
    sections = ("dataset", "time", "graph", "features", "models")
    return ArtifactStore(cfg).stage(BUNDLE_STAGE, sections).exists(ALERTS_FILE)


@pytest.mark.skipif(
    not _bundle_exists(),
    reason="demo bundle absent; run scripts/06_demo_bundle.py",
)
def test_streamlit_app_runs_without_exceptions():
    """Executes the whole app headlessly via Streamlit's own harness.

    "The demo works" is otherwise a claim nobody checks until it is on a projector. This
    runs the real script against the real bundle and fails on any uncaught exception.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"),
                            default_timeout=300)
    app.run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [str(e.value) for e in app.error]

    # The four numbers the demo exists to communicate must actually reach the screen.
    rendered = {m.label: m.value for m in app.metric}
    assert any("E1" in label for label in rendered)
    assert any("E2" in label for label in rendered)
    assert "Alert reduction" in rendered
    assert app.selectbox, "no transaction selector rendered"
