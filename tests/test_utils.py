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


class TestFormatTranscriptWithScreenshots:
    def _conv(self):
        return {
            "conversation_id": "c1",
            "turns": [
                {"turn_number": 1, "role": "TUTOR", "text": "hi", "type": "DIALOGUE"},
                {"turn_number": 2, "role": "TUTOR", "text": "look here", "type": "DIALOGUE"},
                {"turn_number": 3, "role": "STUDENT", "text": "ok", "type": "DIALOGUE"},
            ],
        }

    def test_marker_rendered_after_anchor_turn(self):
        from annotator.core.utils import format_transcript
        ss = [{"anchor_turn": 2, "filename": "5.0.jpg", "timestamp_seconds": 5.0}]
        out = format_transcript(self._conv(), screenshots=ss)
        lines = out.split("\n")
        assert lines[0] == "Turn 1. TUTOR: hi"
        assert lines[1] == "Turn 2. TUTOR: look here"
        assert lines[2] == "  [SCREEN @ turn 2: image 1]"
        assert lines[3] == "Turn 3. STUDENT: ok"

    def test_multiple_images_numbered_positionally(self):
        from annotator.core.utils import format_transcript
        ss = [
            {"anchor_turn": 1, "filename": "0.5.jpg", "timestamp_seconds": 0.5},
            {"anchor_turn": 2, "filename": "5.0.jpg", "timestamp_seconds": 5.0},
        ]
        out = format_transcript(self._conv(), screenshots=ss)
        assert "[SCREEN @ turn 1: image 1]" in out
        assert "[SCREEN @ turn 2: image 2]" in out

    def test_no_screenshots_unchanged_output(self):
        from annotator.core.utils import format_transcript
        conv = self._conv()
        assert format_transcript(conv) == format_transcript(conv, screenshots=None)
        assert format_transcript(conv) == format_transcript(conv, screenshots=[])


class TestFormatExcerptWithScreenshots:
    def _conv(self):
        turns = [
            {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
             "text": f"t{n}", "type": "DIALOGUE"}
            for n in range(1, 11)
        ]
        return {"conversation_id": "c1", "turns": turns}

    def test_screenshot_outside_excerpt_omitted(self):
        from annotator.core.utils import format_excerpt
        ss_all = [
            {"anchor_turn": 1, "filename": "a.jpg", "timestamp_seconds": 1.0},
            {"anchor_turn": 5, "filename": "b.jpg", "timestamp_seconds": 5.0},
            {"anchor_turn": 9, "filename": "c.jpg", "timestamp_seconds": 9.0},
        ]
        # The caller pre-filters to what's in window; format_excerpt renders markers
        # for the ones passed in.
        filtered = [s for s in ss_all if 4 <= s["anchor_turn"] <= 6]
        out = format_excerpt(self._conv(), turn_start=5, turn_end=5,
                             context_before=1, context_after=1,
                             screenshots=filtered)
        assert "[SCREEN @ turn 5: image 1]" in out
        assert "a.jpg" not in out
        assert "c.jpg" not in out
