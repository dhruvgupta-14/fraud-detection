"""Phase 1 checkpoint tests, on synthetic fixtures rather than the 480 MB real file.

The guards here cover the failure modes that produce a plausible wrong number instead of
a crash: interning that merges two accounts, an amount-column transposition, a typology
join that fans out into a cartesian product, and coverage silently shrinking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.config import load_config
from aml.graph.interner import NodeIndex
from aml.ingest.patterns import (
    UNANNOTATED,
    attach_typology,
    link_patterns_to_transactions,
    parse_patterns,
)
from aml.ingest.transactions import RAW_HEADER, build_canonical, read_transactions_csv, validate

# Column order matches the real file exactly: Amount Received precedes Amount Paid.
ROWS = [
    # ts,               from_bank, acct,      to_bank, acct,      amt_recv, cur_recv, amt_paid, cur_paid, format, label
    ("2022/09/01 00:20", "010", "8000EBD30", "010", "8000EBD30", "3697.34", "US Dollar", "3697.34", "US Dollar", "Reinvestment", "0"),
    ("2022/09/01 00:06", "021174", "800737690", "012", "80011F990", "2848.96", "Euro", "2848.96", "Euro", "ACH", "1"),
    ("2022/09/02 04:33", "021174", "800737690", "020", "80020C5B0", "8630.40", "Euro", "8630.40", "Euro", "ACH", "1"),
    # Same account number at a different bank -- must intern as a distinct node.
    ("2022/09/02 09:14", "999", "800737690", "012", "80011F990", "100.00", "Euro", "90.00", "Yuan", "Wire", "0"),
    # Exact duplicate of the row above it except bank/amount: exercises occurrence ranking.
    ("2022/09/03 10:00", "010", "8000EBD30", "012", "80011F990", "50.00", "Euro", "50.00", "Euro", "Cheque", "0"),
    ("2022/09/03 10:00", "010", "8000EBD30", "012", "80011F990", "50.00", "Euro", "50.00", "Euro", "Cheque", "1"),
]


@pytest.fixture()
def trans_csv(tmp_path):
    path = tmp_path / "HI-Small_Trans.csv"
    lines = [",".join(RAW_HEADER)] + [",".join(r) for r in ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def cfg():
    # Expected counts are overridden to match the fixture; the real values live in
    # config/default.yaml and are checked by the pipeline run itself.
    return load_config(
        overrides={
            "dataset": {
                "expected_rows": len(ROWS),
                "expected_illicit": 3,
                "expected_nodes": 5,
                "expected_pattern_rows": 3,
                "expected_attempts": 2,
            }
        }
    )


@pytest.fixture()
def canonical(trans_csv, cfg):
    raw = read_transactions_csv(trans_csv)
    return build_canonical(raw, cfg)


# ------------------------------------------------------------------ transactions


def test_amount_columns_are_not_transposed(trans_csv):
    raw = read_transactions_csv(trans_csv)
    row = raw[raw["payment_format"] == "Wire"].iloc[0]
    # Raw order is Amount Received, then Amount Paid. Getting this backwards is silent.
    assert row["amount_received"] == 100.0
    assert row["amount_paid"] == 90.0
    assert row["currency_received"] == "Euro"
    assert row["currency_paid"] == "Yuan"


def test_rows_are_timestamp_sorted_with_contiguous_tx_id(trans_csv):
    raw = read_transactions_csv(trans_csv)
    assert raw["timestamp"].is_monotonic_increasing
    assert raw["tx_id"].tolist() == list(range(len(ROWS)))
    # The 00:06 row appears second in the file but is earliest in time.
    assert raw.iloc[0]["payment_format"] == "ACH"


def test_tx_id_assignment_is_deterministic(trans_csv):
    a = read_transactions_csv(trans_csv)
    b = read_transactions_csv(trans_csv)
    pd.testing.assert_series_equal(a["tx_id"], b["tx_id"])
    pd.testing.assert_series_equal(a["amount_paid"], b["amount_paid"])


def test_derived_flags(canonical):
    df, _ = canonical
    self_loop = df[df["payment_format"] == "Reinvestment"].iloc[0]
    assert bool(self_loop["is_self_loop"]) is True
    assert bool(self_loop["is_cross_bank"]) is False

    wire = df[df["payment_format"] == "Wire"].iloc[0]
    assert bool(wire["is_cross_currency"]) is True
    assert bool(wire["is_cross_bank"]) is True


def test_day_idx_is_zero_based_from_origin(canonical):
    df, _ = canonical
    assert df["day_idx"].min() == 0
    assert df["day_idx"].max() == 2


def test_validate_rejects_a_row_count_mismatch(canonical, cfg):
    df, _ = canonical
    with pytest.raises(ValueError, match="row count"):
        validate(df.iloc[:-1], cfg)


def test_header_mismatch_is_rejected(tmp_path, cfg):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected header"):
        read_transactions_csv(path)


# ------------------------------------------------------------------ interner


def test_same_account_number_at_two_banks_stays_distinct(canonical):
    df, index = canonical
    # 800737690 appears at bank 021174 and at bank 999. This is the real dataset's
    # 8-collision case in miniature; merging them would corrupt degree and centrality.
    dupes = index.frame[index.frame["acct"] == "800737690"]
    assert len(dupes) == 2
    assert set(dupes["bank"]) == {"021174", "999"}
    assert dupes["node_id"].nunique() == 2


def test_encode_roundtrips_and_flags_unknown(canonical):
    df, index = canonical
    banks = pd.Series(["021174", "999", "NOSUCH"])
    accts = pd.Series(["800737690", "800737690", "800737690"])
    ids = index.encode(banks, accts)
    assert ids[0] != ids[1]
    assert ids[2] == -1
    decoded = index.decode(ids[:2])
    assert decoded["bank"].tolist() == ["021174", "999"]


def test_node_ids_cover_every_endpoint(canonical):
    df, index = canonical
    endpoints = pd.concat([df["src_node"], df["dst_node"]]).unique()
    assert set(endpoints) <= set(index.frame["node_id"])
    assert len(index) == cfg_expected_nodes()


def cfg_expected_nodes() -> int:
    # 8000EBD30@010, 800737690@021174, 80011F990@012, 80020C5B0@020, 800737690@999
    return 5


def test_interner_is_stable_across_builds(trans_csv, cfg):
    a, _ = build_canonical(read_transactions_csv(trans_csv), cfg)
    b, _ = build_canonical(read_transactions_csv(trans_csv), cfg)
    pd.testing.assert_series_equal(a["src_node"], b["src_node"])
    pd.testing.assert_series_equal(a["dst_node"], b["dst_node"])


# ------------------------------------------------------------------ patterns


PATTERNS_TEXT = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/02 04:33,021174,800737690,020,80020C5B0,8630.40,Euro,8630.40,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops
2022/09/03 10:00,010,8000EBD30,012,80011F990,50.00,Euro,50.00,Euro,Cheque,1
END LAUNDERING ATTEMPT - CYCLE
"""


@pytest.fixture()
def patterns_file(tmp_path):
    path = tmp_path / "HI-Small_Patterns.txt"
    path.write_text(PATTERNS_TEXT, encoding="utf-8")
    return path


def test_parse_extracts_family_and_param(patterns_file):
    pat = parse_patterns(patterns_file)
    assert len(pat) == 3
    assert pat["typology"].tolist() == ["FAN-OUT", "FAN-OUT", "CYCLE"]
    assert pat["typology_param"].tolist() == [16, 16, 10]
    assert pat["attempt_id"].tolist() == [0, 0, 1]


def test_parse_handles_a_header_without_a_param(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(
        "BEGIN LAUNDERING ATTEMPT - BIPARTITE\n"
        "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - BIPARTITE\n",
        encoding="utf-8",
    )
    pat = parse_patterns(path)
    assert pat["typology"].tolist() == ["BIPARTITE"]
    assert pd.isna(pat["typology_param"].iloc[0])


def test_parse_rejects_unterminated_and_mismatched_blocks(tmp_path):
    unterminated = tmp_path / "a.txt"
    unterminated.write_text("BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unterminated"):
        parse_patterns(unterminated)

    mismatched = tmp_path / "b.txt"
    mismatched.write_text(
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops\nEND LAUNDERING ATTEMPT - STACK\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        parse_patterns(mismatched)


def test_link_matches_the_correct_duplicate_row(canonical, patterns_file):
    df, index = canonical
    pat = parse_patterns(patterns_file)
    typology_map, report = link_patterns_to_transactions(pat, df, index)

    assert report["matched"] == 3
    assert report["unmatched"] == 0
    assert typology_map["tx_id"].is_unique

    # Two transactions share the 09-03 10:00 Cheque key and only one is illicit; the join
    # must land on the illicit one, not on its licit twin and not on both.
    cycle_tx = typology_map[typology_map["typology"] == "CYCLE"]["tx_id"]
    assert len(cycle_tx) == 1
    assert df.loc[df["tx_id"] == cycle_tx.iloc[0], "label"].iloc[0] == 1

    # No ambiguity remains once candidates are restricted to illicit rows, but the
    # underlying key collision is still reported rather than swept away.
    assert report["ambiguous_keys"] == 0
    assert report["key_collides_with_licit"] == 1


def test_link_reports_partial_coverage_rather_than_hiding_it(canonical, patterns_file):
    df, index = canonical
    pat = parse_patterns(patterns_file).iloc[:2]  # drop the CYCLE annotation
    _, report = link_patterns_to_transactions(pat, df, index)

    assert report["illicit_total"] == 3
    assert report["illicit_annotated"] == 2
    assert report["illicit_unannotated"] == 1
    assert report["annotation_coverage"] == pytest.approx(2 / 3)


def test_link_flags_rows_whose_accounts_are_unknown(canonical, patterns_file):
    df, index = canonical
    pat = parse_patterns(patterns_file)
    pat.loc[0, "src_bank"] = "NOSUCHBANK"
    _, report = link_patterns_to_transactions(pat, df, index)
    assert report["unknown_node_rows"] == 1
    assert report["unmatched"] == 1


def test_attach_typology_separates_licit_from_unannotated(canonical, patterns_file):
    df, index = canonical
    pat = parse_patterns(patterns_file).iloc[:2]
    typology_map, _ = link_patterns_to_transactions(pat, df, index)
    typ = attach_typology(df, typology_map)

    licit = df["label"] == 0
    assert typ[licit.to_numpy()].isna().all(), "licit rows must not enter the typology breakdown"
    assert (typ == UNANNOTATED).sum() == 1
    assert (typ == "FAN-OUT").sum() == 2


def test_amounts_join_as_exact_cents(canonical, patterns_file):
    df, index = canonical
    pat = parse_patterns(patterns_file)
    # A sub-cent perturbation must break the match rather than being absorbed by a
    # floating point tolerance.
    pat.loc[0, "amount_paid"] = "2848.97"
    _, report = link_patterns_to_transactions(pat, df, index)
    assert report["unmatched"] == 1


def test_dtypes_are_narrow(canonical):
    df, _ = canonical
    assert df["src_node"].dtype == np.int32
    assert df["day_idx"].dtype == np.int16
    assert df["label"].dtype == np.int8
    assert df["is_self_loop"].dtype == bool
