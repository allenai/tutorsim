"""Tests for extract_human_scenarios in benchmark/core/scenarios.py.

The new 'human' mode reads scaffolding-flavored key moments from
data/ground_truth_hybrid/ (filtered to situation_label_agg in
{scaffolding, rigor} and cut_turn present) and turns each into a Scenario.
"""
import json
from unittest.mock import patch

from benchmark.core.scenarios import extract_human_scenarios, load_scenarios, Scenario
from benchmark.core.scenarios import (
    _pick_modal_cut, _role_adjust_cut, _pick_representative_member,
)
from benchmark.core.annotator_bridge import build_synthetic_detections
from benchmark.core.exchange import Exchange


def _make_transcripts():
    """Three transcripts keyed by composite conv_ids (UUID at the end)."""
    return {
        "2024-t1_2024-s1_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": {
            "conversation_id": "2024-t1_2024-s1_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "turns": [
                {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
                 "text": f"t{n}"} for n in range(1, 21)
            ],
        },
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb": {
            "conversation_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "turns": [
                {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
                 "text": f"t{n}"} for n in range(1, 21)
            ],
        },
    }


def _gt_files():
    """Hybrid GT shape that exercises clustering + modal cut + tie + role adjust.

    Transcript A turn roles: odd=TUTOR, even=STUDENT (1..20).
    Clusters and their expected scenario behavior:
      (5,8)   singleton, cut=6 STUDENT -> kept, cut_turn=6
      (10,12) singleton, cut=11 TUTOR  -> kept, cut_turn=10 (adjusted)
      (3,5)   two members vote 4 and 5 (tie) -> smallest=4 STUDENT -> cut_turn=4
      (16,18) three members vote 17, 17, 18 -> modal=17 TUTOR -> cut_turn=16 (adjusted)
      (13,15) mixed agg -> cluster dropped
      (17,19) rapport -> cluster dropped (situation_label_agg absent)
      (2,4)   votes 1 (< ts) and 5 (> te) -> all votes filtered -> cluster dropped
    Transcript B:
      (8,9)   rigor singleton, cut=8 STUDENT -> kept, cut_turn=8
    Transcript C: not in transcripts dict -> all moments dropped.
    """
    def m(ts, te, ann, cut, agg="scaffolding", ann_type="scaffolding",
          situation="S", moment_id=None):
        d = {
            "turn_start": ts, "turn_end": te,
            "annotation_type": ann_type, "annotator_id": ann,
            "situation": situation, "action": "A", "result": "R",
            "strategy_label": "effective",
        }
        if agg is not None:
            d["situation_label_agg"] = agg
        if cut is not None:
            d["cut_turn"] = cut
        if moment_id is not None:
            d["moment_id"] = moment_id
        return d

    return [
        {
            "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "num_turns": 20,
            "key_moments": [
                m(5, 8, "ann1", 6, situation="S_5_8", moment_id="m1"),
                m(10, 12, "ann1", 11, situation="S_10_12", moment_id="m2"),
                m(3, 5, "ann1", 4, situation="S_3_5_v4"),
                m(3, 5, "ann2", 5, situation="S_3_5_v5"),
                m(16, 18, "ann1", 17, situation="S_16_18_a"),
                m(16, 18, "ann2", 17, situation="S_16_18_b"),
                m(16, 18, "ann1", 18, situation="S_16_18_c"),
                m(13, 15, "ann1", 14, agg="mixed"),
                m(17, 19, "ann1", 18, ann_type="rapport", agg=None),
                m(2, 4, "ann1", 1),                      # cut < ts -> vote dropped
                m(2, 4, "ann2", 5),                      # cut > te -> vote dropped
            ],
        },
        {
            "conversation_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "num_turns": 20,
            "key_moments": [
                m(8, 9, "ann2", 8, agg="rigor", situation="S_8_9", moment_id="m5"),
            ],
        },
        {
            "conversation_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "num_turns": 20,
            "key_moments": [
                m(3, 5, "ann3", 4),
            ],
        },
    ]


def _run_extract():
    with patch("benchmark.core.scenarios.load_all_ground_truth_files",
               return_value=_gt_files()):
        return extract_human_scenarios(_make_transcripts())


def test_extracts_one_scenario_per_cluster():
    scenarios = _run_extract()
    # Transcript A keeps 4 clusters (5-8, 10-12, 3-5, 16-18); transcript B keeps 1
    # (8-9). Transcripts C / dropped clusters contribute 0.
    assert len(scenarios) == 5


def test_scenario_id_uses_turn_range():
    scenarios = _run_extract()
    ids = {s.scenario_id.rsplit("__", 1)[1] for s in scenarios}
    assert ids == {"hum_5_8", "hum_10_12", "hum_3_5", "hum_16_18", "hum_8_9"}


def test_singleton_student_cut_is_unchanged():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_5_8"))
    assert s.cut_turn == 6
    assert s.detection["chosen_cut_turn"] == 6
    assert s.detection["situation"] == "S_5_8"
    assert s.detection["moment_id"] == "m1"
    assert s.detection["annotator_id"] == "ann1"
    assert s.detection["situation_label_agg"] == "scaffolding"
    assert s.detection["cluster_size"] == 1
    assert s.detection["cut_votes"] == {6: 1}
    # Prefix includes turn 6:
    assert "Turn 6." in s.transcript_prefix
    assert "Turn 7." not in s.transcript_prefix


def test_singleton_tutor_cut_is_adjusted():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_10_12"))
    # cut_turn=11 is TUTOR (odd) -> adjusted to 10.
    assert s.cut_turn == 10
    assert s.detection["chosen_cut_turn"] == 11
    assert "Turn 10." in s.transcript_prefix
    assert "Turn 11." not in s.transcript_prefix


def test_tie_resolves_to_smallest_cut():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_3_5"))
    # votes were 4 and 5 -> tie -> smallest=4 (STUDENT, no adjustment)
    assert s.cut_turn == 4
    assert s.detection["chosen_cut_turn"] == 4
    assert s.detection["cut_votes"] == {4: 1, 5: 1}
    assert s.detection["cluster_size"] == 2


def test_same_annotator_dupes_inflate_votes_and_modal_wins():
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_16_18"))
    # votes: ann1 -> 17, ann2 -> 17, ann1 -> 18.  modal=17 (TUTOR) -> adjusted to 16.
    assert s.cut_turn == 16
    assert s.detection["chosen_cut_turn"] == 17
    assert s.detection["cut_votes"] == {17: 2, 18: 1}
    assert s.detection["cluster_size"] == 3
    # Representative should be one of the modal voters (ann1 or ann2, smallest annotator_id):
    assert s.detection["annotator_id"] == "ann1"


def test_drops_cluster_when_all_votes_out_of_range():
    scenarios = _run_extract()
    assert not any(s.scenario_id.endswith("__hum_2_4") for s in scenarios)


def test_drops_mixed_and_rapport_clusters():
    scenarios = _run_extract()
    assert not any(s.scenario_id.endswith("__hum_13_15") for s in scenarios)
    assert not any(s.scenario_id.endswith("__hum_17_19") for s in scenarios)


def test_skips_when_transcript_missing():
    scenarios = _run_extract()
    assert not any(s.conv_id.startswith("cccccccc") for s in scenarios)


def test_scenario_id_stable_across_runs():
    a = _run_extract()
    b = _run_extract()
    assert sorted(s.scenario_id for s in a) == sorted(s.scenario_id for s in b)


def test_detection_shape_is_compatible_with_annotator_bridge():
    """Round-trip: a human scenario must work with build_synthetic_detections."""
    scenarios = _run_extract()
    s = next(s for s in scenarios if s.scenario_id.endswith("__hum_5_8"))
    fake_exchange = Exchange(
        scenario_id=s.scenario_id,
        tutor_model="x",
        generated_turns=[
            {"turn_number": s.cut_turn + 1, "role": "TUTOR", "text": "ok"},
            {"turn_number": s.cut_turn + 2, "role": "STUDENT", "text": "yes"},
        ],
        tutor_usage={}, student_usage={}, completed=True,
    )
    detections = build_synthetic_detections(s, fake_exchange)
    assert isinstance(detections, dict)
    assert s.conv_id in detections
    conv_detections = detections[s.conv_id]["detections"]
    assert len(conv_detections) == 1
    det = conv_detections[0]
    assert det["annotation_type"] == "scaffolding"
    assert det["turn_start"] == s.detection["turn_start"]
    assert det["turn_end"] == s.cut_turn + 2


def test_load_scenarios_human_mode_skips_detection_and_extracts():
    transcripts = _make_transcripts()
    with patch("benchmark.core.scenarios.load_all_ground_truth_files",
               return_value=_gt_files()), \
         patch("benchmark.core.scenarios.load_transcripts",
               return_value=transcripts):
        scenarios = load_scenarios(
            {"mode": "human"},
            detections_by_conv=None,   # MUST NOT raise -- human mode skips detection
        )
    assert len(scenarios) == 5
    assert all(s.mode == "human" for s in scenarios)


# ---------------------------------------------------------------------------
# Helper tests (Task 1)
# ---------------------------------------------------------------------------

def test_pick_modal_cut_single_winner():
    assert _pick_modal_cut([4, 4, 5]) == 4


def test_pick_modal_cut_tie_returns_smallest():
    assert _pick_modal_cut([4, 5]) == 4
    assert _pick_modal_cut([7, 3, 7, 3]) == 3


def test_pick_modal_cut_singleton():
    assert _pick_modal_cut([9]) == 9


def test_pick_modal_cut_empty_returns_none():
    assert _pick_modal_cut([]) is None


def _conv(turns_pattern):
    """Build a conversation dict with turn roles per turns_pattern[i] for turn_number i+1."""
    return {
        "turns": [
            {"turn_number": n + 1, "role": role, "text": f"t{n+1}"}
            for n, role in enumerate(turns_pattern)
        ],
    }


def test_role_adjust_cut_student_no_change():
    # Odd-numbered turns are TUTOR, even are STUDENT (matches the main fixture below).
    conv = _conv(["TUTOR", "STUDENT"] * 10)  # turns 1..20
    assert _role_adjust_cut(6, conv) == 6  # turn 6 is STUDENT


def test_role_adjust_cut_tutor_decrements():
    conv = _conv(["TUTOR", "STUDENT"] * 10)
    assert _role_adjust_cut(11, conv) == 10  # turn 11 is TUTOR -> cut-1


def test_role_adjust_cut_tutor_at_first_turn_returns_none():
    conv = _conv(["TUTOR", "STUDENT"] * 10)
    # cut=1 is TUTOR; adjustment would give 0, which is below the 1-turn minimum.
    assert _role_adjust_cut(1, conv) is None


def test_role_adjust_cut_missing_turn_returns_none():
    conv = _conv(["TUTOR", "STUDENT"] * 10)
    assert _role_adjust_cut(99, conv) is None


def test_pick_representative_member_prefers_modal_voter():
    members = [
        {"annotator_id": "z", "cut_turn": 9, "situation": "z-other"},
        {"annotator_id": "b", "cut_turn": 6, "situation": "b-modal"},
        {"annotator_id": "a", "cut_turn": 6, "situation": "a-modal"},
    ]
    rep = _pick_representative_member(members, chosen_cut=6)
    # Among members voting modal=6, smallest annotator_id wins.
    assert rep["annotator_id"] == "a"
    assert rep["situation"] == "a-modal"


def test_pick_representative_member_falls_back_when_no_modal_voter():
    # Defensive: if no member voted the chosen cut (shouldn't happen in practice),
    # return the smallest-annotator_id member overall.
    members = [
        {"annotator_id": "z", "cut_turn": 9, "situation": "z"},
        {"annotator_id": "a", "cut_turn": 8, "situation": "a"},
    ]
    rep = _pick_representative_member(members, chosen_cut=6)
    assert rep["annotator_id"] == "a"


# Note: next_problem_queue was removed -- [NEXT_PROBLEM] is now just a second
# end-token. Scenarios don't carry a queue of subsequent moments anymore.
