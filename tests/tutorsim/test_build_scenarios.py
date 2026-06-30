"""Hermetic tests for build_scenarios using fixture data.

Fixture layout:
  tests/tutorsim/fixtures/build_src/
    ground_truth/  conv-aaa.json  conv-bbb.json  conv-ccc.json  conv-ddd.json
    transcripts/   conv-aaa.json  conv-bbb.json  conv-ccc.json
    transcripts.jsonl  (normalized JSONL records; contains conv-ddd only)

conv-aaa: cluster (3,5), votes=[4,4,5] -> modal=4, STUDENT turn -> cut stays 4
conv-bbb: cluster (1,3), votes=[2,3]   -> modal=2 (tie->min), STUDENT turn -> cut stays 2
conv-ccc: cluster (2,4), votes=[3]     -> modal=3, TUTOR turn -> cut decrements to 2
conv-ddd: cluster (1,3), votes=[2]     -> JSONL-only conv; STUDENT turn -> cut stays 2
"""

import json
from pathlib import Path
import pytest

FIXTURE_DIR = Path("tests/tutorsim/fixtures/build_src")
GT_DIR = FIXTURE_DIR / "ground_truth"
TX_DIR = FIXTURE_DIR / "transcripts"
TX_JSONL = str(FIXTURE_DIR / "transcripts.jsonl")


def _ids_for(*conv_ts_te_pairs):
    """Build scenario id strings from (conv_id, ts, te) tuples."""
    return [f"{conv}__hum_{ts}_{te}" for conv, ts, te in conv_ts_te_pairs]


# ---------------------------------------------------------------------------
# Import guard — fails loudly if build_scenarios isn't implemented yet
# ---------------------------------------------------------------------------

def _import_build():
    from tutorsim.scenarios import build_scenarios
    return build_scenarios


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildScenarios:
    """All tests are fully hermetic -- fixture dirs only, no real data."""

    def _build(self, ids, set_name="test"):
        build_scenarios = _import_build()
        return build_scenarios(
            set_name=set_name,
            ids=ids,
            ground_truth_dir=str(GT_DIR),
            transcripts_dir=str(TX_DIR),
        )

    def test_field_mapping_dimension_and_gold(self):
        """dimension and rubric.gold both come from situation_label_agg."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.dimension == "scaffolding"
        assert s.rubric["gold"] == "scaffolding"

    def test_field_mapping_hint_from_situation(self):
        """rubric.hint comes from the situation field."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        assert s.rubric["hint"] == "Student gave a vague guess about fractions"

    def test_field_mapping_id(self):
        """Scenario id = set_name:conv_id__hum_ts_te."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids, set_name="myset")
        assert scenarios[0].id == "myset:conv-aaa__hum_3_5"

    def test_context_is_prefix_turns(self):
        """context contains turns up to and including cut_turn as [{turn_number, role, text}]."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        # conv-aaa cut=4 (STUDENT at 4, stays). Prefix: turns 1-4
        assert len(s.context) == 4
        assert s.context[0] == {"turn_number": 1, "role": "tutor", "text": "What is half of 6?"}
        assert s.context[1] == {"turn_number": 2, "role": "student", "text": "I think it might be 3?"}
        assert s.context[2] == {"turn_number": 3, "role": "tutor", "text": "Can you explain how you got that?"}
        assert s.context[3] == {"turn_number": 4, "role": "student", "text": "I split it into two groups"}
        # Roles must be lowercased
        for turn in s.context:
            assert turn["role"] in ("tutor", "student")
        # turn_numbers must be real (from source transcript)
        assert [t["turn_number"] for t in s.context] == [1, 2, 3, 4]

    def test_student_reference_is_post_cut(self):
        """student.reference contains only turns AFTER cut_turn."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        # cut=4, post-cut turns are 5,6,7
        ref = s.student["reference"]
        assert "Turn 5." in ref
        assert "Turn 6." in ref
        assert "Turn 7." in ref
        # Pre-cut turns must NOT appear
        assert "Turn 4." not in ref
        assert "Turn 3." not in ref

    def test_student_context_field(self):
        """student.context comes from conversation context."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        assert s.student["context"] == "Grade 5, Mathematics, fractions"

    def test_student_mode_is_oracle(self):
        """student.mode is always 'oracle'."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        assert scenarios[0].student["mode"] == "oracle"

    def test_provenance_fields(self):
        """provenance includes all expected fields."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        p = scenarios[0].provenance
        assert p["conv_id"] == "conv-aaa"
        assert p["cut_turn"] == 4           # final adjusted cut
        assert p["turn_start"] == 3
        assert p["turn_end"] == 5
        assert p["moment_id"] == "m-001"
        assert "annotator_id" in p
        assert p["chosen_cut_turn"] == 4    # pre-adjustment modal vote
        assert "cut_votes" in p
        assert "cluster_size" in p

    def test_cut_logic_modal_vote_with_tie_picks_smallest(self):
        """conv-bbb: votes=[2,3] -> tie -> smallest=2. STUDENT at 2 -> stays 2."""
        ids = _ids_for(("conv-bbb", 1, 3))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.provenance["chosen_cut_turn"] == 2
        assert s.provenance["cut_turn"] == 2

    def test_cut_logic_tutor_role_decrements(self):
        """conv-ccc: vote=3, turn 3 is TUTOR -> cut decrements to 2."""
        ids = _ids_for(("conv-ccc", 2, 4))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.provenance["chosen_cut_turn"] == 3   # modal before adjustment
        assert s.provenance["cut_turn"] == 2           # after TUTOR decrement
        # context should be prefix up to cut=2 (turns 1,2)
        assert len(s.context) == 2

    def test_ids_order_is_preserved(self):
        """Output order matches the requested ids order, not file-system order."""
        # Request bbb before aaa intentionally
        ids = _ids_for(("conv-bbb", 1, 3), ("conv-aaa", 3, 5), ("conv-ccc", 2, 4))
        scenarios = self._build(ids)
        assert len(scenarios) == 3
        assert scenarios[0].provenance["conv_id"] == "conv-bbb"
        assert scenarios[1].provenance["conv_id"] == "conv-aaa"
        assert scenarios[2].provenance["conv_id"] == "conv-ccc"

    def test_unknown_ids_skipped(self):
        """IDs that don't match any ground-truth cluster are silently skipped."""
        ids = _ids_for(("conv-aaa", 3, 5), ("conv-zzz", 9, 11))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        assert scenarios[0].provenance["conv_id"] == "conv-aaa"

    def test_cut_votes_structure(self):
        """conv-aaa: votes=[4,4,5] -> modal=4, cut_votes={4:2, 5:1}."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        p = scenarios[0].provenance
        assert p["cluster_size"] == 3
        votes = p["cut_votes"]
        assert votes.get(4) == 2
        assert votes.get(5) == 1


class TestBuildScenariosJsonl:
    """Tests that JSONL transcripts are loaded and hydrated correctly."""

    def _build(self, ids, step_up_jsonl=None, set_name="test"):
        from tutorsim.scenarios import build_scenarios
        return build_scenarios(
            set_name=set_name,
            ids=ids,
            ground_truth_dir=str(GT_DIR),
            transcripts_dir=str(TX_DIR),
            step_up_jsonl=step_up_jsonl,
        )

    def test_jsonl_only_conv_hydrates_with_jsonl_param(self):
        """conv-ddd exists only in JSONL; with step_up_jsonl it should hydrate fully."""
        ids = _ids_for(("conv-ddd", 1, 3))
        scenarios = self._build(ids, step_up_jsonl=TX_JSONL)
        assert len(scenarios) == 1
        s = scenarios[0]
        # context = turns up to cut=2 (STUDENT at 2, stays)
        assert len(s.context) == 2
        assert s.context[0]["role"] == "tutor"
        assert s.context[1]["role"] == "student"
        # turn_number must be present and correct
        assert "turn_number" in s.context[0]
        assert "turn_number" in s.context[1]
        # student.reference has post-cut turns (3, 4)
        ref = s.student["reference"]
        assert "Turn 3." in ref
        assert "Turn 4." in ref
        assert "Turn 2." not in ref
        # student.context comes from demographics (Grade 7, Algebra)
        ctx = s.student["context"]
        assert "7" in ctx
        assert "Algebra" in ctx
        # provenance
        assert s.provenance["conv_id"] == "conv-ddd"
        assert s.provenance["cut_turn"] == 2

    def test_jsonl_only_conv_not_hydrated_without_jsonl_param(self):
        """conv-ddd exists only in JSONL; without step_up_jsonl it must be skipped."""
        ids = _ids_for(("conv-ddd", 1, 3))
        scenarios = self._build(ids, step_up_jsonl=None)
        assert len(scenarios) == 0

    def test_jsonl_conv_and_json_conv_both_hydrate(self):
        """When JSONL and per-file JSON convs are both requested, both hydrate."""
        ids = _ids_for(("conv-aaa", 3, 5), ("conv-ddd", 1, 3))
        scenarios = self._build(ids, step_up_jsonl=TX_JSONL)
        assert len(scenarios) == 2
        conv_ids = [s.provenance["conv_id"] for s in scenarios]
        assert "conv-aaa" in conv_ids
        assert "conv-ddd" in conv_ids
