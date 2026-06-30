import pytest
from tutorsim.scenarios import (
    DatasetNotFoundError,
    Scenario,
    load_scenarios,
    validate_dataset,
)


def test_scenario_roundtrip():
    s = Scenario(
        id="balanced_520:conv__hum_5_7",
        context=[
            {"turn_number": 5, "role": "tutor", "text": "Hi"},
            {"turn_number": 7, "role": "student", "text": "ok"},
        ],
        dimension="scaffolding",
        student={"mode": "oracle", "reference": "Turn 8. STUDENT: ...", "context": "Grade 5, fractions"},
        rubric={"gold": "scaffolding", "hint": "The student gave a guess."},
        provenance={"conv_id": "conv", "cut_turn": 7, "turn_start": 5, "turn_end": 7},
    )
    d = s.to_dict()
    assert Scenario.from_dict(d) == s
    assert d["dimension"] == "scaffolding"
    assert d["rubric"]["gold"] == "scaffolding"
    assert d["context"][0]["role"] == "tutor"
    assert d["context"][0]["turn_number"] == 5


def test_load_scenarios_reads_in_file_order():
    """Verify load_scenarios reads scenarios.jsonl in file order."""
    scs = load_scenarios("mini_set", root="tests/tutorsim/fixtures")
    assert len(scs) == 2
    assert scs[0].id == "mini:conv1__hum_1_2"
    assert scs[1].dimension == "rigor"
    assert [s.dimension for s in scs] == ["scaffolding", "rigor"]


def test_load_scenarios_missing_raises():
    """Verify load_scenarios raises FileNotFoundError for missing set."""
    with pytest.raises(DatasetNotFoundError):
        load_scenarios("does_not_exist", root="tests/tutorsim/fixtures")


def test_validate_dataset_reads_manifest_and_hashes_fixture():
    report = validate_dataset("mini_set", root="tests/tutorsim/fixtures")
    assert report["record_count"] == 2
    assert report["manifest"]["version"] == "fixture"
    assert len(report["content_hash"]) == 64
