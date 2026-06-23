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


def test_score_scenarios_scaffolding_did_rate():
    """Lucy's scaffolding_did_rate = (# scaffolding-gold scenarios where LM
    scaffolded) / (# scaffolding-gold scenarios). gt=scaffolding & pred=scaffolding
    counts as yes; gt=scaffolding & pred=neither counts as no."""
    pairs = [
        _scenario_score_input("a", "scaffolding", "scaffolding", "pos"),
        _scenario_score_input("b", "scaffolding", "neither", "neg"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding_did"]["n_yes"] == 1
    assert result["scaffolding_did"]["n_total"] == 2
    assert result["scaffolding_did"]["rate"] == 0.5


def test_score_scenarios_rigor_pred_on_scaffolding_gold_no_penalty():
    """Under Lucy's scoring, pred=rigor on scaffolding-gold no longer
    penalizes the rigor axis (no FP). It just counts as 'didn't scaffold'."""
    pairs = [
        _scenario_score_input("a", "scaffolding", "rigor", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding_did"]["n_yes"] == 0
    assert result["scaffolding_did"]["n_total"] == 1
    # rigor_did denominator is scenarios where GOLD says rigor -- this one isn't.
    assert result["rigor_did"]["n_total"] == 0


def test_score_scenarios_both_credits_both_dimensions():
    """'both' decomposes to (scaf=yes, rig=yes), so it counts as a yes on
    whichever dimension the gold cared about."""
    pairs = [
        # gt='scaffolding' (denominator counts), pred='both' -> scaffolded: yes
        _scenario_score_input("a", "scaffolding", "both", "pos"),
        # gt='rigor' (denominator counts), pred='both' -> pushed rigor: yes
        _scenario_score_input("b", "rigor", "both", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["scaffolding_did"]["rate"] == 1.0
    assert result["rigor_did"]["rate"] == 1.0


def test_score_scenarios_gold_both_excluded_from_did_rates():
    """Gold 'both' / 'mixed' / 'neither' / 'unknown' lack a clean direction so
    they're excluded from BOTH did-rate denominators (still count toward
    outcome / over-scaffold rates)."""
    pairs = [
        _scenario_score_input("a", "both", "scaffolding", "pos"),
        _scenario_score_input("b", "scaffolding", "scaffolding", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    # Only 1 scenario has gold='scaffolding'; the 'both' scenario is excluded.
    assert result["scaffolding_did"]["n_total"] == 1
    assert result["scaffolding_did"]["n_yes"] == 1
    assert result["rigor_did"]["n_total"] == 0
    # Outcome rate is over ALL scenarios, including the excluded 'both' one.
    assert result["outcome_pos_rate"] == 1.0


def test_score_scenarios_outcome_rate():
    pairs = [
        _scenario_score_input("a", "scaffolding", "scaffolding", "pos"),
        _scenario_score_input("b", "scaffolding", "scaffolding", "neg"),
        _scenario_score_input("c", "scaffolding", "scaffolding", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["outcome_pos_rate"] == 2/3


def test_score_scenarios_sentinel_labels_excluded_from_did_rate():
    """gt or pred 'mixed'/'unclear'/'unknown' don't carry a clean direction
    on the relevant axis -- excluded from the corresponding did-rate."""
    pairs = [
        _scenario_score_input("a", "scaffolding", "scaffolding", "pos"),
        _scenario_score_input("b", "mixed", "scaffolding", "pos"),       # gt sentinel: excluded
        _scenario_score_input("c", "scaffolding", "unclear", "neg"),     # pred sentinel: counts as 'didn't scaffold'
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    # scaffolding_did: gold='scaffolding' on (a, c). LM scaffolded only on (a).
    assert result["scaffolding_did"]["n_yes"] == 1
    assert result["scaffolding_did"]["n_total"] == 2
    assert result["outcome_pos_rate"] == 2/3


def test_score_scenarios_overscaffold_rate_when_field_absent():
    """If no annotation has overscaffold_decomposed (PR #18 not run),
    `overscaffold.available` is False and rate is 0."""
    pairs = [
        _scenario_score_input("a", "scaffolding", "scaffolding", "pos"),
    ]
    result = score_scenarios([p[0] for p in pairs], [p[1] for p in pairs])
    assert result["overscaffold"]["available"] is False
    assert result["overscaffold"]["n_yes"] == 0


def test_score_scenarios_overscaffold_rate_when_field_present():
    """When `overscaffold_decomposed` is populated, the rate counts scenarios
    with any non-empty facet list."""
    sa, aa = _scenario_score_input("a", "scaffolding", "scaffolding", "pos")
    sb, ab = _scenario_score_input("b", "scaffolding", "scaffolding", "pos")
    aa["annotations"][0]["overscaffold_decomposed"] = ["The tutor over-explained."]
    ab["annotations"][0]["overscaffold_decomposed"] = []
    result = score_scenarios([sa, sb], [aa, ab])
    assert result["overscaffold"]["available"] is True
    assert result["overscaffold"]["n_yes"] == 1
    assert result["overscaffold"]["n_total"] == 2
    assert result["overscaffold"]["rate"] == 0.5


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

    assert "scaffolding_did" in summary
    assert "rigor_did" in summary
    assert "outcome_pos_rate" in summary
    # Test fixture: 1 scaffolding-gold scenario where the LM scaffolded, 1
    # rigor-gold scenario where the LM did NOT push rigor.
    assert summary["scaffolding_did"]["n_yes"] == 1
    assert summary["scaffolding_did"]["n_total"] == 1
    assert summary["rigor_did"]["n_yes"] == 0
    assert summary["rigor_did"]["n_total"] == 1
    assert summary["outcome_pos_rate"] == 1.0
