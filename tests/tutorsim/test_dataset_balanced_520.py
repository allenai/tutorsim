"""Validation tests for the frozen balanced_520 dataset.

These tests assert hard invariants on scenarios/balanced_520/scenarios.jsonl
and exercise the load_scenarios API round-trip. All counts are exact; any
deviation is a failure, not a warning.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from tutorsim.scenarios import load_scenarios

# Repo root relative to this file: tests/tutorsim/ -> ../../
_REPO_ROOT = Path(__file__).parent.parent.parent
_SCENARIOS_FILE = _REPO_ROOT / "scenarios" / "balanced_520" / "scenarios.jsonl"
_IDS_FILE = _REPO_ROOT / "data" / "balanced_520_scenario_ids.json"

# The frozen balanced_520 dataset is NOT committed to git (distributed via
# HuggingFace). These validation tests run only when the dataset has been built
# or downloaded locally; otherwise they skip rather than fail.
pytestmark = pytest.mark.skipif(
    not (_SCENARIOS_FILE.exists() and _IDS_FILE.exists()),
    reason="balanced_520 dataset not present locally (distributed via HuggingFace, not committed)",
)


@pytest.fixture(scope="module")
def raw_scenarios() -> list[dict]:
    """Load scenarios as raw dicts for cheap attribute checks."""
    assert _SCENARIOS_FILE.exists(), f"Missing: {_SCENARIOS_FILE}"
    records = []
    with open(_SCENARIOS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def canonical_ids() -> list[str]:
    """Load the canonical ordered id list."""
    assert _IDS_FILE.exists(), f"Missing: {_IDS_FILE}"
    return json.load(open(_IDS_FILE, encoding="utf-8"))


class TestExactCounts:
    def test_total_count(self, raw_scenarios):
        assert len(raw_scenarios) == 520, f"Expected 520, got {len(raw_scenarios)}"

    def test_dimension_split(self, raw_scenarios):
        counts = Counter(s["dimension"] for s in raw_scenarios)
        assert counts["scaffolding"] == 260, (
            f"Expected 260 scaffolding, got {counts['scaffolding']}"
        )
        assert counts["rigor"] == 260, (
            f"Expected 260 rigor, got {counts['rigor']}"
        )

    def test_all_have_non_empty_context(self, raw_scenarios):
        empty = [s["id"] for s in raw_scenarios if not s.get("context")]
        assert not empty, f"Scenarios with empty context: {empty}"

    def test_all_have_rubric_gold(self, raw_scenarios):
        missing = [
            s["id"]
            for s in raw_scenarios
            if not s.get("rubric", {}).get("gold")
        ]
        assert not missing, f"Scenarios missing rubric.gold: {missing}"

    def test_ids_are_unique(self, raw_scenarios):
        all_ids = [s["id"] for s in raw_scenarios]
        assert len(set(all_ids)) == len(all_ids), "Duplicate scenario ids found"

    def test_ids_are_namespaced(self, raw_scenarios):
        bad = [s["id"] for s in raw_scenarios if not s["id"].startswith("balanced_520:")]
        assert not bad, f"IDs not namespaced with 'balanced_520:': {bad[:5]}"

    def test_reference_counts(self, raw_scenarios):
        """Report reference presence. Empty is allowed only when real convo ended at cut."""
        non_empty = sum(
            1 for s in raw_scenarios if s.get("student", {}).get("reference", "")
        )
        empty = len(raw_scenarios) - non_empty
        # Both empty and non-empty are acceptable; report the split as an assertion
        # so test output shows the numbers even when passing.
        assert non_empty + empty == 520
        # Log counts via assertion message for visibility
        print(f"reference non-empty: {non_empty}, empty: {empty}")


class TestOrderMatchesIdList:
    def test_order_matches_id_list(self, raw_scenarios, canonical_ids):
        """Scenario order must match data/balanced_520_scenario_ids.json."""
        # Build the expected prefixed ids in the order they appear in the id-list
        actual_ids = [s["id"] for s in raw_scenarios]
        actual_set = set(actual_ids)

        expected_ordered = [
            f"balanced_520:{raw_id}"
            for raw_id in canonical_ids
            if f"balanced_520:{raw_id}" in actual_set
        ]

        assert actual_ids == expected_ordered, (
            "Scenario order does not match canonical id-list order"
        )


class TestLoadScenariosRoundTrip:
    def test_load_scenarios_count(self):
        """load_scenarios returns exactly 520 Scenario objects."""
        from tutorsim.scenarios import Scenario

        # Use repo root as the root argument
        scenarios = load_scenarios("balanced_520", root=str(_REPO_ROOT / "scenarios"))
        assert len(scenarios) == 520, f"Expected 520, got {len(scenarios)}"
        assert all(isinstance(s, Scenario) for s in scenarios)

    def test_load_scenarios_first_last_ids(self, canonical_ids):
        """First and last scenario ids match canonical id-list order."""
        scenarios = load_scenarios("balanced_520", root=str(_REPO_ROOT / "scenarios"))
        expected_first = f"balanced_520:{canonical_ids[0]}"
        expected_last = f"balanced_520:{canonical_ids[-1]}"
        assert scenarios[0].id == expected_first, (
            f"First id mismatch: {scenarios[0].id} != {expected_first}"
        )
        assert scenarios[-1].id == expected_last, (
            f"Last id mismatch: {scenarios[-1].id} != {expected_last}"
        )
