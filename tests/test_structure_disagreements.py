"""Tests for the action-direction disagreement collector.

These pin the collector to eval.compute_action_direction_f1's scoring: gold and
LLM 4-way action labels are decomposed into independent (scaffolding, rigor)
yes/no dimensions, sentinel labels ("unclear"/"unknown") are excluded, and a
span is a disagreement only when the two sides differ on at least one dimension.
"""
import pytest

pytest.importorskip("krippendorff")  # transitively imported via annotator.eval.eval

from annotator.iteration.structure_disagreements import (
    collect_action_direction_disagreements,
    collect_action_direction_agreements,
    collect_student_outcome_disagreements,
    collect_student_outcome_agreements,
)


def _gold_moment(turn_start, turn_end, action_direction_agg=None, student_outcome_agg=None):
    moment = {
        "annotation_type": "scaffolding",
        "turn_start": turn_start,
        "turn_end": turn_end,
    }
    if action_direction_agg is not None:
        moment["action_direction_agg"] = action_direction_agg
    if student_outcome_agg is not None:
        moment["student_outcome_agg"] = student_outcome_agg
    return moment


def _llm_annotation(turn_start, turn_end, action_label=None, result_label=None):
    ann = {
        "annotation_type": "scaffolding",
        "turn_start": turn_start,
        "turn_end": turn_end,
    }
    if action_label is not None:
        ann["action_label"] = action_label
    if result_label is not None:
        ann["result_label"] = result_label
    return ann


def _collect(gold_moments, llm_anns, collector=collect_action_direction_disagreements):
    ground_truth = {"conversations": {"conv1": {"key_moments": gold_moments}}}
    structure_labels_by_conv = {"conv1": llm_anns}
    return collector(
        ground_truth, structure_labels_by_conv, ["conv1"], unified_facets_by_span={})


def _by_span(disagreements):
    return {(d["turn_start"], d["turn_end"]): d for d in disagreements}


class TestCollectActionDirectionDisagreements:
    def test_excludes_sentinel_labels(self):
        # gold "unclear"/"unknown" and llm "unclear" carry no per-dimension
        # verdict, so eval drops them from F1 -- they must not appear as
        # disagreements even though the raw 4-way labels differ.
        disagreements = _collect(
            [
                _gold_moment(1, 10, "unclear"),
                _gold_moment(20, 30, "unknown"),
                _gold_moment(40, 50, "both"),
            ],
            [
                _llm_annotation(1, 10, "both"),
                _llm_annotation(20, 30, "scaffolding"),
                _llm_annotation(40, 50, "unclear"),
            ],
        )
        assert disagreements == []

    def test_excludes_full_agreement(self):
        disagreements = _collect(
            [_gold_moment(1, 10, "both")],
            [_llm_annotation(1, 10, "both")],
        )
        assert disagreements == []

    def test_partial_agreement_reports_only_disagreeing_dimension(self):
        # gold "both" = (scaffolding=yes, rigor=yes); llm "scaffolding" =
        # (yes, no). They agree on scaffolding, disagree only on rigor.
        disagreements = _collect(
            [_gold_moment(1, 10, "both")],
            [_llm_annotation(1, 10, "scaffolding")],
        )
        entry = _by_span(disagreements)[(1, 10)]
        assert entry["disagree_dims"] == ["rigor"]
        assert entry["gold_dims"] == {"scaffolding": "yes", "rigor": "yes"}
        assert entry["llm_dims"] == {"scaffolding": "yes", "rigor": "no"}

    def test_total_disagreement_reports_both_dimensions(self):
        # gold "both" = (yes, yes); llm "neither" = (no, no).
        disagreements = _collect(
            [_gold_moment(1, 10, "both")],
            [_llm_annotation(1, 10, "neither")],
        )
        entry = _by_span(disagreements)[(1, 10)]
        assert entry["disagree_dims"] == ["scaffolding", "rigor"]

    def test_skips_span_without_llm_match(self):
        disagreements = _collect(
            [_gold_moment(1, 10, "both")],
            [_llm_annotation(99, 100, "neither")],
        )
        assert disagreements == []


class TestCollectActionDirectionAgreements:
    """Agreements mirror the same per-dimension F1 framing as disagreements: a
    span is an agreement on each dimension where the decomposed gold/LLM verdicts
    match. Sentinel labels are excluded just as in the disagreement view."""

    def _agree(self, gold_moments, llm_anns):
        return _collect(gold_moments, llm_anns,
                        collector=collect_action_direction_agreements)

    def test_full_agreement_reports_both_dimensions(self):
        # gold "both" == llm "both": agree on scaffolding and rigor.
        agreements = self._agree(
            [_gold_moment(1, 10, "both")],
            [_llm_annotation(1, 10, "both")],
        )
        entry = _by_span(agreements)[(1, 10)]
        assert entry["agree_dims"] == ["scaffolding", "rigor"]
        assert entry["disagree_dims"] == []

    def test_partial_agreement_reports_only_agreeing_dimension(self):
        # gold "both" = (yes, yes); llm "scaffolding" = (yes, no). Agree on
        # scaffolding, disagree on rigor -- it surfaces as an agreement (on
        # scaffolding) AND would surface as a disagreement (on rigor).
        agreements = self._agree(
            [_gold_moment(1, 10, "both")],
            [_llm_annotation(1, 10, "scaffolding")],
        )
        entry = _by_span(agreements)[(1, 10)]
        assert entry["agree_dims"] == ["scaffolding"]
        assert entry["disagree_dims"] == ["rigor"]

    def test_excludes_total_disagreement(self):
        # gold "both" = (yes, yes); llm "neither" = (no, no): no agreeing dim.
        agreements = self._agree(
            [_gold_moment(1, 10, "both")],
            [_llm_annotation(1, 10, "neither")],
        )
        assert agreements == []

    def test_excludes_sentinel_labels(self):
        agreements = self._agree(
            [_gold_moment(1, 10, "unclear"), _gold_moment(20, 30, "both")],
            [_llm_annotation(1, 10, "both"), _llm_annotation(20, 30, "unclear")],
        )
        assert agreements == []


class TestCollectStudentOutcomeDisagreements:
    def _disagree(self, gold_moments, llm_anns):
        return _collect(gold_moments, llm_anns,
                        collector=collect_student_outcome_disagreements)

    def test_reports_pos_neg_mismatch(self):
        disagreements = self._disagree(
            [_gold_moment(1, 10, student_outcome_agg="pos")],
            [_llm_annotation(1, 10, result_label="neg")],
        )
        entry = _by_span(disagreements)[(1, 10)]
        assert entry["gold_label"] == "pos"
        assert entry["llm_label"] == "neg"

    def test_excludes_match(self):
        disagreements = self._disagree(
            [_gold_moment(1, 10, student_outcome_agg="pos")],
            [_llm_annotation(1, 10, result_label="pos")],
        )
        assert disagreements == []

    def test_excludes_non_substantive_gold(self):
        # Only pos/neg gold verdicts are scored by F1 -- no_evidence/unclear
        # gold spans must not appear even when the LLM labels them differently.
        disagreements = self._disagree(
            [_gold_moment(1, 10, student_outcome_agg="no_evidence")],
            [_llm_annotation(1, 10, result_label="pos")],
        )
        assert disagreements == []


class TestCollectStudentOutcomeAgreements:
    def _agree(self, gold_moments, llm_anns):
        return _collect(gold_moments, llm_anns,
                        collector=collect_student_outcome_agreements)

    def test_reports_pos_neg_match(self):
        agreements = self._agree(
            [_gold_moment(1, 10, student_outcome_agg="pos")],
            [_llm_annotation(1, 10, result_label="pos")],
        )
        entry = _by_span(agreements)[(1, 10)]
        assert entry["gold_label"] == "pos"
        assert entry["llm_label"] == "pos"

    def test_excludes_mismatch(self):
        agreements = self._agree(
            [_gold_moment(1, 10, student_outcome_agg="pos")],
            [_llm_annotation(1, 10, result_label="neg")],
        )
        assert agreements == []

    def test_excludes_non_substantive_gold(self):
        agreements = self._agree(
            [_gold_moment(1, 10, student_outcome_agg="no_evidence")],
            [_llm_annotation(1, 10, result_label="no_evidence")],
        )
        assert agreements == []
