"""Tests for eval metric functions."""
import pytest

pytest.importorskip("krippendorff")  # annotator.eval.eval imports krippendorff

from annotator.eval import eval as eval_mod
from annotator.eval.eval import (
    cohens_kappa, compute_consensus_label, map_to_binary, compute_student_outcome_f1,
    eval_output_filename, load_eval_json,
    EFFECTIVENESS_LABELS, BINARY_LABELS,
)


def _gold_moment(turn_start, turn_end, student_outcome_agg):
    return {
        "annotation_type": "scaffolding",
        "turn_start": turn_start,
        "turn_end": turn_end,
        "student_outcome_agg": student_outcome_agg,
    }


def _llm_annotation(turn_start, turn_end, result_label):
    return {
        "annotation_type": "scaffolding",
        "turn_start": turn_start,
        "turn_end": turn_end,
        "result_label": result_label,
    }


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = ["effective", "partial", "ineffective"]
        b = ["effective", "partial", "ineffective"]
        assert cohens_kappa(a, b, EFFECTIVENESS_LABELS) == 1.0

    def test_empty_lists(self):
        assert cohens_kappa([], [], EFFECTIVENESS_LABELS) == 0.0

    def test_complete_disagreement(self):
        # With linear weights and a uniform distribution, kappa = 0 (not negative)
        # because pe equals po when all predictions are in the same off-diagonal cell.
        # A simpler disagreement case: one wrong direction on a binary scale.
        a = ["right", "right"]
        b = ["wrong", "wrong"]
        kappa = cohens_kappa(a, b, BINARY_LABELS)
        assert kappa <= 0

    def test_binary_perfect(self):
        a = ["right", "wrong", "right"]
        b = ["right", "wrong", "right"]
        assert cohens_kappa(a, b, BINARY_LABELS) == 1.0


class TestComputeConsensusLabel:
    def test_majority_vote(self):
        assert compute_consensus_label(["effective", "effective", "partial"]) == "effective"

    def test_tie_uses_median(self):
        # Tie between effective and ineffective: sorted ordinals [0,0,2,2],
        # median index = len//2 = 2 -> ordinal 2 -> "ineffective"
        labels = ["effective", "effective", "ineffective", "ineffective"]
        assert compute_consensus_label(labels) == "ineffective"

    def test_empty(self):
        assert compute_consensus_label([]) == "unclear"

    def test_single_label(self):
        assert compute_consensus_label(["partial"]) == "partial"


class TestMapToBinary:
    def test_effective(self):
        assert map_to_binary("effective") == "right"

    def test_partial(self):
        assert map_to_binary("partial") == "wrong"

    def test_ineffective(self):
        assert map_to_binary("ineffective") == "wrong"

    def test_unknown(self):
        assert map_to_binary("unclear") is None


class TestComputeStudentOutcomeF1:
    def test_excludes_gold_no_evidence_and_unclear_spans(self):
        # Gold spans without a substantive pos/neg verdict carry no signal to
        # score the LLM's pos/neg call against -- they must not count as units,
        # even when the LLM disagrees with them.
        ground_truth = {"conversations": {"conv1": {"key_moments": [
            _gold_moment(1, 10, "pos"),
            _gold_moment(20, 30, "no_evidence"),
            _gold_moment(40, 50, "unclear"),
        ]}}}
        structure_labels_by_conv = {"conv1": [
            _llm_annotation(1, 10, "pos"),
            _llm_annotation(20, 30, "neg"),
            _llm_annotation(40, 50, "pos"),
        ]}
        result = compute_student_outcome_f1(ground_truth, structure_labels_by_conv, ["conv1"])
        assert result["n_units"] == 1
        assert set(result["f1"].keys()) == {"pos"}

    def test_reports_only_pos_class_with_neg_as_negative(self):
        # 2 gold "pos" (1 correctly predicted, 1 missed as "neg") and 1 gold
        # "neg" predicted as "pos" (a false positive for the "pos" class).
        ground_truth = {"conversations": {"conv1": {"key_moments": [
            _gold_moment(1, 10, "pos"),
            _gold_moment(20, 30, "pos"),
            _gold_moment(40, 50, "neg"),
        ]}}}
        structure_labels_by_conv = {"conv1": [
            _llm_annotation(1, 10, "pos"),
            _llm_annotation(20, 30, "neg"),
            _llm_annotation(40, 50, "pos"),
        ]}
        result = compute_student_outcome_f1(ground_truth, structure_labels_by_conv, ["conv1"])
        assert result["n_units"] == 3
        assert set(result["f1"].keys()) == {"pos"}
        pos = result["f1"]["pos"]
        assert pos["precision"] == pytest.approx(0.5)   # tp=1, fp=1
        assert pos["recall"] == pytest.approx(0.5)      # tp=1, fn=1
        assert pos["f1"] == pytest.approx(0.5)
        assert result["macro_f1"] == pytest.approx(0.5)

    def test_no_substantive_gold_spans_returns_empty(self):
        ground_truth = {"conversations": {"conv1": {"key_moments": [
            _gold_moment(1, 10, "no_evidence"),
        ]}}}
        structure_labels_by_conv = {"conv1": [_llm_annotation(1, 10, "pos")]}
        result = compute_student_outcome_f1(ground_truth, structure_labels_by_conv, ["conv1"])
        assert result == {"f1": {}, "macro_f1": None, "n_units": 0, "confusion": {}}


class TestEvalOutputFilename:
    def test_includes_profile_suffix(self):
        # The whole bug: profile must appear in the output filename so two
        # profiles don't clobber each other's scorecard.
        assert eval_output_filename("annotations", profile="anthropic") == \
            "eval_annotations_anthropic.json"

    def test_suffix_order_profile_style_split(self):
        assert eval_output_filename(
            "annotations", profile="anthropic", annotator_style="lenient", split="test"
        ) == "eval_annotations_anthropic_lenient_test.json"

    def test_train_split_omitted(self):
        # train is the default and stays unsuffixed for back-compat
        assert eval_output_filename("detections", profile="gemini", split="train") == \
            "eval_detections_gemini.json"

    def test_no_optional_args_matches_legacy_mode_name(self):
        assert eval_output_filename("full") == "eval_full.json"


class TestLoadEvalJson:
    def test_prefers_profile_suffixed_file(self, monkeypatch):
        seen = []

        def fake_load(version, filename):
            seen.append(filename)
            return {"mode": "annotations"} if filename == "eval_annotations_anthropic.json" else None

        monkeypatch.setattr(eval_mod, "load_annotator_result", fake_load)
        data = load_eval_json("v1", "annotations", profile="anthropic")
        assert data == {"mode": "annotations"}
        assert seen[0] == "eval_annotations_anthropic.json"

    def test_falls_back_to_unsuffixed_then_legacy(self, monkeypatch):
        def fake_load(version, filename):
            if filename == "eval.json":
                return {"mode": "annotations"}
            return None

        monkeypatch.setattr(eval_mod, "load_annotator_result", fake_load)
        data = load_eval_json("v1", "annotations", profile="anthropic")
        assert data == {"mode": "annotations"}
