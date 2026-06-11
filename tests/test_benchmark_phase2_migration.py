"""Tests for the benchmark Phase 2 migration to annotate -> decompose -> structure."""
from unittest.mock import MagicMock

from benchmark.core.annotator_bridge import (
    decompose_bulk, structure_bulk,
)


def test_decompose_bulk_enriches_each_annotation_with_facets(monkeypatch):
    """decompose_bulk wraps run_decompose; should add action_decomposed and
    result_decomposed to each annotation in the per-scenario results dict."""
    def fake_run_decompose(version, model, mode, phase_cfg, **kwargs):
        data = kwargs["annotations_data"]
        for sid, scen in data["results"].items():
            for ann in scen["annotations"]:
                ann["action_decomposed"] = ["facet a1", "facet a2"]
                ann["result_decomposed"] = ["facet r1"]
        return data

    monkeypatch.setattr(
        "benchmark.core.annotator_bridge.run_decompose",
        fake_run_decompose,
    )

    per_scenario_results = {
        "s1": {"s1": {"annotations": [
            {"annotation_type": "scaffolding", "turn_start": 5, "turn_end": 10,
             "situation": "Student stuck on solving for x.",
             "action": "Tutor broke the problem into two steps.",
             "result": "Student followed the steps."}
        ]}}
    }
    enriched = decompose_bulk(
        per_scenario_results=per_scenario_results,
        annotator_profile="anthropic",
        mode="sync",
    )
    ann = enriched["s1"]["s1"]["annotations"][0]
    assert ann["action_decomposed"] == ["facet a1", "facet a2"]
    assert ann["result_decomposed"] == ["facet r1"]


def test_structure_bulk_adds_action_and_result_labels(monkeypatch):
    """structure_bulk wraps run_structure_label; should add action_label and
    result_label to each annotation."""
    def fake_run_structure(version, model, mode, phase_cfg, **kwargs):
        data = kwargs["annotations_data"]
        for sid, scen in data["results"].items():
            for ann in scen["annotations"]:
                ann["action_label"] = ["scaffolding", "neither"]
                ann["result_label"] = ["pos"]
        return data

    monkeypatch.setattr(
        "benchmark.core.annotator_bridge.run_structure_label",
        fake_run_structure,
    )

    per_scenario_results = {
        "s1": {"s1": {"annotations": [
            {"annotation_type": "scaffolding",
             "turn_start": 5, "turn_end": 10,
             "action_decomposed": ["a1", "a2"],
             "result_decomposed": ["r1"]}
        ]}}
    }
    enriched = structure_bulk(
        per_scenario_results=per_scenario_results,
        annotator_profile="anthropic",
        mode="sync",
    )
    ann = enriched["s1"]["s1"]["annotations"][0]
    assert ann["action_label"] == ["scaffolding", "neither"]
    assert ann["result_label"] == ["pos"]


def test_decompose_bulk_empty_input_returns_empty(monkeypatch):
    enriched = decompose_bulk(
        per_scenario_results={},
        annotator_profile="anthropic",
        mode="sync",
    )
    assert enriched == {}


def test_structure_bulk_empty_input_returns_empty(monkeypatch):
    enriched = structure_bulk(
        per_scenario_results={},
        annotator_profile="anthropic",
        mode="sync",
    )
    assert enriched == {}


from benchmark.core.score import score_scenarios


def _scenario_score_input(sid, agg, action_label, result_label):
    """Build a (scenario, annotation) pair using single-string labels
    (matches structure.py's actual output shape)."""
    scenario = {
        "scenario_id": sid,
        "conv_id": sid + "_conv",
        "mode": "human",
        "detection": {"situation_label_agg": agg},
    }
    annotation = {
        "annotations": [
            {"action_decomposed": ["a"], "action_label": action_label,
             "result_decomposed": ["r"], "result_label": result_label},
        ],
    }
    return scenario, annotation


def test_score_scenarios_scaffolding_tp_fn():
    """gt=scaffolding & pred=scaffolding -> scaffolding TP; gt=scaffolding & pred=neither -> scaffolding FN."""
    pairs = [
        _scenario_score_input("a", "scaffolding", "scaffolding", "pos"),
        _scenario_score_input("b", "scaffolding", "neither", "neg"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding"]["tp"] == 1
    assert result["scaffolding"]["fn"] == 1
    assert result["scaffolding"]["fp"] == 0
    assert abs(result["scaffolding"]["f1"] - 2/3) < 1e-6


def test_score_scenarios_rigor_fp():
    """gt=scaffolding & pred=rigor -> rigor FP and scaffolding FN."""
    pairs = [
        _scenario_score_input("a", "scaffolding", "rigor", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["rigor"]["fp"] == 1
    assert result["scaffolding"]["fn"] == 1


def test_score_scenarios_both_decomposes_to_both_dimensions():
    """'both' decomposes to (scaf=yes, rig=yes) on both sides via Lucy's
    _ACTION_LABEL_TO_DIMENSIONS, so partial agreement is scored per dimension."""
    pairs = [
        # gt='scaffolding' -> (yes,no); pred='both' -> (yes,yes): scaf TP, rigor FP
        _scenario_score_input("a", "scaffolding", "both", "pos"),
        # gt='rigor' -> (no,yes); pred='both' -> (yes,yes): scaf FP, rigor TP
        _scenario_score_input("b", "rigor", "both", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding"]["tp"] == 1
    assert result["rigor"]["tp"] == 1
    assert result["scaffolding"]["fp"] == 1
    assert result["rigor"]["fp"] == 1


def test_score_scenarios_gold_both_scores_both_dimensions():
    """A gold 'both' tag should count toward both dimensions' recall (positive on each)."""
    pairs = [
        # gt='both' -> (yes,yes); pred='scaffolding' -> (yes,no): scaf TP, rigor FN
        _scenario_score_input("a", "both", "scaffolding", "pos"),
        # gt='both' -> (yes,yes); pred='both' -> (yes,yes): scaf TP, rigor TP
        _scenario_score_input("b", "both", "both", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding"]["tp"] == 2
    assert result["rigor"]["tp"] == 1
    assert result["rigor"]["fn"] == 1


def test_score_scenarios_outcome_rate():
    pairs = [
        _scenario_score_input("a", "scaffolding", "scaffolding", "pos"),
        _scenario_score_input("b", "scaffolding", "scaffolding", "neg"),
        _scenario_score_input("c", "scaffolding", "scaffolding", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["outcome_pos_rate"] == 2/3


def test_score_scenarios_sentinel_labels_excluded_from_action_f1():
    """gt or pred 'mixed'/'unclear'/'unknown' -> no per-dimension verdict -> excluded
    from F1 (still counts toward outcome rate)."""
    pairs = [
        _scenario_score_input("a", "scaffolding", "scaffolding", "pos"),
        _scenario_score_input("b", "mixed", "scaffolding", "pos"),       # gt sentinel
        _scenario_score_input("c", "scaffolding", "unclear", "neg"),     # pred sentinel
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding"]["tp"] == 1
    assert result["n_scored_for_f1"] == 1
    assert result["outcome_pos_rate"] == 2/3


import pytest


def test_phase2_e2e_produces_one_annotation_per_scenario_and_score(monkeypatch, tmp_path):
    """Mocked end-to-end: Phase 2 produces flat annotations/{profile}/{scenario_id}.json
    (no styles dir) and scores/{profile}.json with action_f1 + outcome_rate."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear(); st._backend = None

    from benchmark.core import annotator_bridge as ab

    def fake_prepare(scenarios, exchanges, **kw):
        entries = []
        all_detections = {}
        for s in scenarios:
            entries.append({"key": f"{s.scenario_id}__0"})
            all_detections[s.scenario_id] = {
                s.scenario_id: {"detections": [{"annotation_type": "scaffolding",
                                                "turn_start": 5, "turn_end": 10}]}
            }
        return entries, all_detections, {}

    def fake_execute(entries, all_detections, annotator_profile, mode, existing_batch_id, on_batch_created):
        out = {}
        for sid in all_detections:
            out[sid] = {sid: {"annotations": [
                {"annotation_type": "scaffolding", "turn_start": 5, "turn_end": 10,
                 "situation": "S", "action": "A1. A2.", "result": "R1."}
            ]}}
        return out

    def fake_decompose_bulk(per_scenario_results, annotator_profile, mode):
        for sid, results in per_scenario_results.items():
            for inner_sid, scen in results.items():
                for ann in scen["annotations"]:
                    ann["action_decomposed"] = ["A1.", "A2."]
                    ann["result_decomposed"] = ["R1."]
        return per_scenario_results

    def fake_structure_bulk(per_scenario_results, annotator_profile, mode):
        for sid, results in per_scenario_results.items():
            for inner_sid, scen in results.items():
                for ann in scen["annotations"]:
                    # structure.py emits single-string labels per annotation
                    ann["action_label"] = "scaffolding"
                    ann["result_label"] = "pos"
        return per_scenario_results

    monkeypatch.setattr(ab, "prepare_bulk_entries", fake_prepare)
    monkeypatch.setattr(ab, "execute_and_parse_bulk", fake_execute)
    monkeypatch.setattr(ab, "decompose_bulk", fake_decompose_bulk)
    monkeypatch.setattr(ab, "structure_bulk", fake_structure_bulk)

    from benchmark.core.scenarios import Scenario
    from benchmark.core.exchange import Exchange

    scenarios = [
        Scenario(scenario_id="s1", conv_id="c1", cut_turn=4,
                 transcript_prefix="...", student_context="ctx",
                 last_student_message="hi", mode="human",
                 detection={"turn_start": 5, "turn_end": 10,
                            "annotation_type": "scaffolding",
                            "situation_label_agg": "scaffolding"}),
        Scenario(scenario_id="s2", conv_id="c2", cut_turn=4,
                 transcript_prefix="...", student_context="ctx",
                 last_student_message="hi", mode="human",
                 detection={"turn_start": 5, "turn_end": 10,
                            "annotation_type": "scaffolding",
                            "situation_label_agg": "rigor"}),
    ]
    exchanges = {
        "s1": Exchange(scenario_id="s1", tutor_model="m",
                       generated_turns=[{"turn_number": 5, "role": "TUTOR", "text": "x"}],
                       completed=True),
        "s2": Exchange(scenario_id="s2", tutor_model="m",
                       generated_turns=[{"turn_number": 5, "role": "TUTOR", "text": "x"}],
                       completed=True),
    }

    from benchmark.run import run_phase2_and_score
    summary = run_phase2_and_score(
        version="t1",
        profile="anthropic",
        annotator_profile="anthropic",
        annotator_mode="sync",
        prompt_version="v13",
        context_window=20,
        scenarios=scenarios,
        exchanges=exchanges,
    )

    assert "scaffolding" in summary
    assert "rigor" in summary
    assert "outcome_pos_rate" in summary
    assert summary["scaffolding"]["tp"] == 1
    assert summary["rigor"]["fn"] == 1
    assert summary["outcome_pos_rate"] == 1.0
