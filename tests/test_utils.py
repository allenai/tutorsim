"""Tests for annotator.core.utils pure functions."""
import pytest
from annotator.core.utils import compute_iou, merge_overlapping_ranges


class TestComputeIou:
    def test_identical_ranges(self):
        assert compute_iou((1, 5), (1, 5)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((1, 3), (5, 7)) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((1, 5), (3, 7))
        assert abs(iou - 3 / 7) < 1e-9

    def test_one_contains_other(self):
        iou = compute_iou((1, 10), (3, 5))
        assert abs(iou - 3 / 10) < 1e-9

    def test_adjacent_ranges(self):
        assert compute_iou((1, 3), (4, 6)) == 0.0

    def test_single_turn_overlap(self):
        iou = compute_iou((1, 3), (3, 5))
        assert abs(iou - 1 / 5) < 1e-9


class TestMergeOverlappingRanges:
    def test_empty_input(self):
        assert merge_overlapping_ranges([]) == []

    def test_no_overlap(self):
        moments = [
            {"turn_start": 1, "turn_end": 3, "annotation_type": "scaffolding"},
            {"turn_start": 10, "turn_end": 12, "annotation_type": "scaffolding"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 2

    def test_overlapping_same_type(self):
        moments = [
            {"turn_start": 1, "turn_end": 5, "annotation_type": "scaffolding"},
            {"turn_start": 3, "turn_end": 8, "annotation_type": "scaffolding"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 1
        assert clusters[0]["turn_start"] == 1
        assert clusters[0]["turn_end"] == 8

    def test_adjacent_same_type_merged(self):
        moments = [
            {"turn_start": 1, "turn_end": 3, "annotation_type": "scaffolding"},
            {"turn_start": 4, "turn_end": 6, "annotation_type": "scaffolding"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 1

    def test_different_types_not_merged(self):
        moments = [
            {"turn_start": 1, "turn_end": 5, "annotation_type": "scaffolding"},
            {"turn_start": 3, "turn_end": 8, "annotation_type": "rapport"},
        ]
        clusters = merge_overlapping_ranges(moments)
        assert len(clusters) == 2
