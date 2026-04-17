"""Tests for eval metric functions."""
import pytest
from annotator.eval.eval import (
    cohens_kappa, compute_consensus_label, map_to_binary,
    EFFECTIVENESS_LABELS, BINARY_LABELS,
)


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
