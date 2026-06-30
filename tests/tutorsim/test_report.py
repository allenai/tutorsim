"""Golden test: pins the exact numeric output of report.aggregate.

Fixture: 6 (Scenario, Annotation) pairs, covering all branches in the port:
  1. gold=scaffolding, action=scaffolding (clean hit, no over-scaffold)
  2. gold=scaffolding, action=both       (right action, but over-scaffold -> not clean)
  3. gold=rigor,       action=rigor      (clean hit, result_label=pos)
  4. gold=rigor,       action=neither    (miss)
  5. gold=both         (excluded from did-rate denominators), action=both, no over
  6. gold=scaffolding, action=neither    (miss, result_label=pos)

Expected values computed BY HAND from score.py's formulas:

  scaf_total    = 3   (pairs 1, 2, 6)
  scaf_yes      = 2   (pairs 1, 2 -- action_label in {scaffolding, both} -> pred_dims[0]=="yes")
  scaf_clean_yes= 1   (pair 1 only -- pair 2 has over-scaffold so excluded from clean)
  scaf_over_yes = 1   (pair 2)

  rig_total     = 2   (pairs 3, 4)
  rig_yes       = 1   (pair 3 -- action=rigor -> pred_dims[1]=="yes")
  rig_clean_yes = 1   (pair 3 -- no over-scaffold)

  over_yes      = 1   (pair 2)
  n_total       = 6

  outcome_pos   = 2   (pairs 3 and 6)

Rates:
  scaffolding_did.rate      = 2/3
  rigor_did.rate            = 1/2
  overscaffold.rate         = 1/6
  outcome_pos_rate          = 2/6 = 1/3
  scaffold_calibrated.score = 1/3
  rigor_calibrated.score    = 1/2

leaderboard_row:
  avoids_overscaffold = 1 - 1/6 = 5/6
"""

import pytest
from types import SimpleNamespace
from tutorsim.report import aggregate, leaderboard_row


# ---------------------------------------------------------------------------
# Minimal stubs -- no SDK imports, no I/O
# ---------------------------------------------------------------------------

def _make_scenario(dimension: str) -> SimpleNamespace:
    """Minimal Scenario stub: only .dimension is read by aggregate."""
    return SimpleNamespace(dimension=dimension)


def _make_annotation(
    action_label: str,
    result_label: str,
    overscaffold_decomposed: list,
) -> SimpleNamespace:
    """Minimal Annotation stub: only .action_label, .result_label,
    .overscaffold_decomposed are read by aggregate."""
    return SimpleNamespace(
        action_label=action_label,
        result_label=result_label,
        overscaffold_decomposed=overscaffold_decomposed,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def six_pairs():
    """6 (scenario, annotation) pairs covering all scoring branches."""
    scenarios = [
        _make_scenario("scaffolding"),  # pair 1: clean scaf hit
        _make_scenario("scaffolding"),  # pair 2: right action but over-scaffold
        _make_scenario("rigor"),        # pair 3: clean rigor hit, pos outcome
        _make_scenario("rigor"),        # pair 4: rigor miss
        _make_scenario("both"),         # pair 5: excluded from did-rate denominators
        _make_scenario("scaffolding"),  # pair 6: scaf miss, pos outcome
    ]
    annotations = [
        _make_annotation("scaffolding", "no_evidence", []),               # pair 1
        _make_annotation("both",        "no_evidence", ["gave answer"]),  # pair 2: over-scaffold
        _make_annotation("rigor",       "pos",         []),               # pair 3
        _make_annotation("neither",     "no_evidence", []),               # pair 4
        _make_annotation("both",        "no_evidence", []),               # pair 5
        _make_annotation("neither",     "pos",         []),               # pair 6
    ]
    return scenarios, annotations


# ---------------------------------------------------------------------------
# Golden test
# ---------------------------------------------------------------------------

def test_aggregate_golden(six_pairs):
    """aggregate() returns the EXACT metric dict for the golden fixture."""
    scenarios, annotations = six_pairs
    result = aggregate(scenarios, annotations)

    # --- n_scenarios ---
    assert result["n_scenarios"] == 6

    # --- scaffolding_did ---
    sd = result["scaffolding_did"]
    assert sd["n_yes"]   == 2
    assert sd["n_total"] == 3
    assert sd["rate"]    == pytest.approx(2 / 3)

    # --- rigor_did ---
    rd = result["rigor_did"]
    assert rd["n_yes"]   == 1
    assert rd["n_total"] == 2
    assert rd["rate"]    == pytest.approx(1 / 2)

    # --- overscaffold ---
    ov = result["overscaffold"]
    assert ov["n_yes"]    == 1
    assert ov["n_total"]  == 6
    assert ov["rate"]     == pytest.approx(1 / 6)
    assert ov["available"] is True   # Annotation dataclass always has the field

    # --- outcome_pos_rate ---
    assert result["outcome_pos_rate"] == pytest.approx(2 / 6)

    # --- scaffold_calibrated ---
    sc = result["scaffold_calibrated"]
    assert sc["n_clean_yes"]   == 1   # pair 1 only (pair 2 has over-scaffold)
    assert sc["n_overscaffold"]== 1   # pair 2
    assert sc["n_total"]       == 3
    assert sc["score"]         == pytest.approx(1 / 3)

    # --- rigor_calibrated ---
    rc = result["rigor_calibrated"]
    assert rc["n_clean_yes"] == 1
    assert rc["n_total"]     == 2
    assert rc["score"]       == pytest.approx(1 / 2)


def test_aggregate_golden_full_dict(six_pairs):
    """Full dict equality check (all keys and values match expected)."""
    scenarios, annotations = six_pairs
    result = aggregate(scenarios, annotations)

    expected = {
        "n_scenarios": 6,
        "scaffolding_did": {
            "n_yes": 2,
            "n_total": 3,
            "rate": 2 / 3,
        },
        "rigor_did": {
            "n_yes": 1,
            "n_total": 2,
            "rate": 1 / 2,
        },
        "overscaffold": {
            "n_yes": 1,
            "n_total": 6,
            "rate": 1 / 6,
            "available": True,
        },
        "outcome_pos_rate": 2 / 6,
        "scaffold_calibrated": {
            "n_clean_yes": 1,
            "n_overscaffold": 1,
            "n_total": 3,
            "score": 1 / 3,
        },
        "rigor_calibrated": {
            "n_clean_yes": 1,
            "n_total": 2,
            "score": 1 / 2,
        },
    }

    # Check keys match exactly
    assert set(result.keys()) == set(expected.keys())

    # Check integer counts exactly
    assert result["n_scenarios"] == expected["n_scenarios"]
    for key in ("scaffolding_did", "rigor_did"):
        assert result[key]["n_yes"]   == expected[key]["n_yes"]
        assert result[key]["n_total"] == expected[key]["n_total"]
    for key in ("scaffold_calibrated", "rigor_calibrated"):
        assert result[key]["n_clean_yes"] == expected[key]["n_clean_yes"]
        assert result[key]["n_total"]     == expected[key]["n_total"]
    assert result["scaffold_calibrated"]["n_overscaffold"] == expected["scaffold_calibrated"]["n_overscaffold"]
    assert result["overscaffold"]["n_yes"]   == expected["overscaffold"]["n_yes"]
    assert result["overscaffold"]["n_total"] == expected["overscaffold"]["n_total"]
    assert result["overscaffold"]["available"] == expected["overscaffold"]["available"]

    # Check floating-point values approximately
    assert result["scaffolding_did"]["rate"]     == pytest.approx(expected["scaffolding_did"]["rate"])
    assert result["rigor_did"]["rate"]           == pytest.approx(expected["rigor_did"]["rate"])
    assert result["overscaffold"]["rate"]        == pytest.approx(expected["overscaffold"]["rate"])
    assert result["outcome_pos_rate"]            == pytest.approx(expected["outcome_pos_rate"])
    assert result["scaffold_calibrated"]["score"] == pytest.approx(expected["scaffold_calibrated"]["score"])
    assert result["rigor_calibrated"]["score"]   == pytest.approx(expected["rigor_calibrated"]["score"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_aggregate_empty():
    """aggregate([],[]) returns all zeros and None rates."""
    result = aggregate([], [])
    assert result["n_scenarios"] == 0
    assert result["scaffolding_did"]["n_yes"]    == 0
    assert result["scaffolding_did"]["n_total"]  == 0
    assert result["scaffolding_did"]["rate"]     is None
    assert result["rigor_did"]["rate"]           is None
    assert result["overscaffold"]["rate"]        is None
    assert result["overscaffold"]["available"]   is False
    assert result["outcome_pos_rate"]            == 0.0
    assert result["scaffold_calibrated"]["score"] is None
    assert result["rigor_calibrated"]["score"]   is None


def test_aggregate_neither_gold_excluded():
    """gold=neither is excluded from did-rate denominators."""
    scenarios = [_make_scenario("neither")]
    annotations = [_make_annotation("scaffolding", "no_evidence", [])]
    result = aggregate(scenarios, annotations)
    assert result["n_scenarios"] == 1
    assert result["scaffolding_did"]["n_total"] == 0
    assert result["scaffolding_did"]["rate"] is None
    assert result["rigor_did"]["n_total"] == 0
    assert result["rigor_did"]["rate"] is None


def test_leaderboard_row_adds_avoids_overscaffold():
    """leaderboard_row adds avoids_overscaffold = 1 - overscaffold.rate."""
    metrics = {
        "n_scenarios": 6,
        "overscaffold": {"n_yes": 1, "n_total": 6, "rate": 1 / 6, "available": True},
    }
    row = leaderboard_row(metrics)
    assert "avoids_overscaffold" in row
    assert row["avoids_overscaffold"] == pytest.approx(5 / 6)
    # Original keys preserved
    assert row["n_scenarios"] == 6


def test_leaderboard_row_none_safe():
    """leaderboard_row returns avoids_overscaffold=None when rate is None."""
    metrics = {
        "overscaffold": {"n_yes": 0, "n_total": 0, "rate": None, "available": False},
    }
    row = leaderboard_row(metrics)
    assert row["avoids_overscaffold"] is None


def test_aggregate_result_label_as_list():
    """result_label as list (archive compatibility): 'pos' in list triggers outcome_pos."""
    scenarios = [_make_scenario("scaffolding")]
    ann = SimpleNamespace(
        action_label="scaffolding",
        result_label=["pos", "other"],
        overscaffold_decomposed=[],
    )
    result = aggregate(scenarios, [ann])
    assert result["outcome_pos_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# leaderboard() and view() tests
# ---------------------------------------------------------------------------

from tutorsim.report import leaderboard, view


def _make_run_summary(
    tutor_model: str,
    mode: str,
    n: int,
    scaffold_cal: float | None,
    rigor_cal: float | None,
    overscaffold_rate: float | None,
    outcome_pos_rate: float,
    did_scaf: float | None,
    did_rig: float | None,
    tutor_lat_p50: float | None = None,
    tutor_lat_p95: float | None = None,
    tokens_total: int | None = None,
) -> dict:
    """Build a fake run summary dict matching the shape report.leaderboard expects."""
    return {
        "tutor_model": tutor_model,
        "mode": mode,
        "n_scenarios": n,
        "scaffold_calibrated": {"score": scaffold_cal, "n_clean_yes": 0, "n_total": n, "n_overscaffold": 0},
        "rigor_calibrated":    {"score": rigor_cal,    "n_clean_yes": 0, "n_total": n},
        "overscaffold":        {"rate": overscaffold_rate, "n_yes": 0, "n_total": n, "available": overscaffold_rate is not None},
        "outcome_pos_rate":    outcome_pos_rate,
        "scaffolding_did":     {"rate": did_scaf,    "n_yes": 0, "n_total": n},
        "rigor_did":           {"rate": did_rig,     "n_yes": 0, "n_total": n},
        "latency":             {"tutor": {"p50_seconds": tutor_lat_p50, "p95_seconds": tutor_lat_p95, "mean_seconds": 1.0, "n": n}},
        "tokens":              {"total": {"total_tokens": tokens_total, "input_tokens": 0, "output_tokens": 0}},
    }


SUMMARY_A = _make_run_summary(
    tutor_model="model-alpha",
    mode="scaffolding_rigor",
    n=100,
    scaffold_cal=0.75,
    rigor_cal=0.60,
    overscaffold_rate=0.10,
    outcome_pos_rate=0.50,
    did_scaf=0.80,
    did_rig=0.65,
    tutor_lat_p50=1.234,
    tutor_lat_p95=3.456,
    tokens_total=500000,
)

SUMMARY_B = _make_run_summary(
    tutor_model="model-beta",
    mode="scaffolding_rigor",
    n=100,
    scaffold_cal=0.50,
    rigor_cal=0.40,
    overscaffold_rate=0.20,
    outcome_pos_rate=0.30,
    did_scaf=0.60,
    did_rig=0.45,
    tutor_lat_p50=2.000,
    tutor_lat_p95=5.000,
    tokens_total=600000,
)

SUMMARY_NONE = _make_run_summary(
    tutor_model="model-gamma",
    mode="scaffolding_only",
    n=50,
    scaffold_cal=None,
    rigor_cal=None,
    overscaffold_rate=None,
    outcome_pos_rate=0.0,
    did_scaf=None,
    did_rig=None,
    tutor_lat_p50=None,
    tutor_lat_p95=None,
    tokens_total=None,
)


EXPECTED_COLUMNS = [
    "tutor_model", "mode", "n",
    "scaffold_cal", "rigor_cal", "avoids_overscaffold",
    "outcome_pos", "did_scaf", "did_rig",
    "tutor_lat_p50", "tutor_lat_p95", "tokens_total",
]


def test_leaderboard_returns_both_formats():
    """leaderboard() returns a (markdown, csv) 2-tuple."""
    result = leaderboard([SUMMARY_A, SUMMARY_B])
    assert isinstance(result, tuple)
    assert len(result) == 2
    md, csv_str = result
    assert isinstance(md, str) and len(md) > 0
    assert isinstance(csv_str, str) and len(csv_str) > 0


def test_leaderboard_emits_both_rows():
    """Both run summaries appear in the markdown table."""
    md, csv_str = leaderboard([SUMMARY_A, SUMMARY_B])
    assert "model-alpha" in md
    assert "model-beta" in md
    assert "model-alpha" in csv_str
    assert "model-beta" in csv_str


def test_leaderboard_column_order_in_header():
    """Markdown header contains exactly the expected columns in order."""
    md, _ = leaderboard([SUMMARY_A, SUMMARY_B])
    header_line = md.split("\n")[0]
    positions = [header_line.find(col) for col in EXPECTED_COLUMNS]
    # Each column must appear in the header
    for col, pos in zip(EXPECTED_COLUMNS, positions):
        assert pos != -1, f"Column '{col}' not found in header: {header_line}"
    # Columns must be in ascending order of position
    assert positions == sorted(positions), f"Columns out of order: {list(zip(EXPECTED_COLUMNS, positions))}"


def test_leaderboard_avoids_overscaffold_formula():
    """avoids_overscaffold == 1 - overscaffold_rate (to 3 decimal places)."""
    md, _ = leaderboard([SUMMARY_A])
    # SUMMARY_A has overscaffold_rate=0.10 => avoids_overscaffold=0.900
    assert "0.900" in md


def test_leaderboard_sorted_desc_by_scaffold_cal():
    """Rows sorted descending by scaffold_cal (model-alpha first, model-beta second)."""
    md, _ = leaderboard([SUMMARY_B, SUMMARY_A])  # pass in reverse order
    lines = [l for l in md.split("\n") if "|" in l and "---" not in l and "tutor_model" not in l]
    assert lines[0].count("model-alpha") >= 1 or "0.750" in lines[0]
    assert lines[1].count("model-beta") >= 1 or "0.500" in lines[1]
    # More direct: first data row has higher scaffold_cal value
    assert "0.750" in lines[0]
    assert "0.500" in lines[1]


def test_leaderboard_none_formatting():
    """None values render as '-' in md and '' in csv."""
    md, csv_str = leaderboard([SUMMARY_NONE])
    # scaffold_cal is None => should appear as '-' in md
    data_lines = [l for l in md.split("\n") if "model-gamma" in l]
    assert len(data_lines) == 1
    assert "| - |" in data_lines[0] or "|-|" in data_lines[0] or data_lines[0].count("| - ") > 0

    # CSV: None fields should be empty strings (not 'None' or '-')
    csv_lines = [l for l in csv_str.split("\n") if "model-gamma" in l]
    assert len(csv_lines) == 1
    assert "None" not in csv_lines[0]


def test_leaderboard_none_sorted_last():
    """Rows with scaffold_cal=None sort after rows with a value."""
    md, _ = leaderboard([SUMMARY_NONE, SUMMARY_A])
    lines = [l for l in md.split("\n") if "|" in l and "---" not in l and "tutor_model" not in l]
    assert "model-alpha" in lines[0]
    assert "model-gamma" in lines[1]


def test_leaderboard_float_precision():
    """Float values are formatted to 3 decimal places."""
    md, _ = leaderboard([SUMMARY_A])
    # scaffold_cal=0.75 -> "0.750"
    assert "0.750" in md
    # tutor_lat_p50=1.234 -> "1.234"
    assert "1.234" in md


def test_view_returns_nonempty_html():
    """view() returns a non-empty string starting with <!DOCTYPE html>."""
    html = view([SUMMARY_A, SUMMARY_B])
    assert isinstance(html, str) and len(html) > 0
    assert "<!DOCTYPE html>" in html or "<html" in html


def test_view_embeds_model_ids():
    """HTML output contains both model IDs."""
    html = view([SUMMARY_A, SUMMARY_B])
    assert "model-alpha" in html
    assert "model-beta" in html


def test_view_embeds_score_values():
    """HTML output contains score values from the runs."""
    html = view([SUMMARY_A])
    # scaffold_cal=0.75 should appear somewhere in the HTML
    assert "0.75" in html or "0.750" in html


# ---------------------------------------------------------------------------
# Spec S7: leaderboard lat/tokens columns show non-"-" when summary carries blocks
# ---------------------------------------------------------------------------

def test_leaderboard_latency_and_tokens_non_dash():
    """A summary with latency.tutor p50/p95 and tokens.total.total_tokens
    produces non-'-' columns in the leaderboard markdown (spec S7).

    SUMMARY_A has tutor_lat_p50=1.234, tutor_lat_p95=3.456, tokens_total=500000.
    """
    md, csv_str = leaderboard([SUMMARY_A])

    # tutor_lat_p50 and tutor_lat_p95 must appear as numbers, not as '-'
    data_lines = [l for l in md.split("\n") if "model-alpha" in l]
    assert len(data_lines) == 1, f"Expected exactly one data row for model-alpha, got: {data_lines}"
    row = data_lines[0]

    # '1.234' and '3.456' must appear in the row (not replaced by '-')
    assert "1.234" in row, f"tutor_lat_p50=1.234 missing from leaderboard row: {row}"
    assert "3.456" in row, f"tutor_lat_p95=3.456 missing from leaderboard row: {row}"
    assert "500000" in row, f"tokens_total=500000 missing from leaderboard row: {row}"

    # Verify '- ' does not cover those columns (None check: model-gamma has None)
    md_none, _ = leaderboard([SUMMARY_NONE])
    none_lines = [l for l in md_none.split("\n") if "model-gamma" in l]
    assert len(none_lines) == 1
    # None values -> '-' in markdown
    assert "| - |" in none_lines[0] or none_lines[0].count("| - ") >= 2, \
        f"Expected '-' for None lat/tokens in: {none_lines[0]}"
