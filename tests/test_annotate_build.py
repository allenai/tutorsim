"""Tests for annotator.core.annotate entry building with screenshots."""
import pytest


class TestBuildAnalysisEntriesWithScreenshots:
    def _detections_for(self, conv_id):
        return {
            conv_id: {
                "detections": [
                    {"turn_start": 1, "turn_end": 1, "annotation_type": "scaffolding",
                     "brief_description": "moment at turn 1"},
                    {"turn_start": 3, "turn_end": 3, "annotation_type": "scaffolding",
                     "brief_description": "moment at turn 3"},
                ],
                "usage": {},
            }
        }

    def test_per_moment_image_filtering(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.annotate import build_analysis_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.annotate as a
        monkeypatch.setattr(
            a, "load_prompt",
            lambda v, t: "P {brief_description} X {excerpt} X {turn_start} X {turn_end}",
        )

        # Narrow context window so only one moment's excerpt includes the screenshot's
        # anchor turn. Fixture has turns at start_seconds 0, 3, 10 and a screenshot at
        # 4.000s which anchors to turn 2.
        # Moment at turn 1 (window 0..2): turn 2 is in scope -> image included.
        # Moment at turn 3 (window 2..4): turn 2 is also in scope -> image included.
        # To distinguish, use context_window=0:
        # Moment at turn 1 (window 1..1): turn 2 NOT in scope -> no image.
        # Moment at turn 3 (window 3..3): turn 2 NOT in scope -> no image.
        # That doesn't test the inclusion case. Use context_window=1:
        # Moment at turn 1 (window 0..2 which clamps to 1..2): turn 2 in scope -> image.
        # Moment at turn 3 (window 2..4 which clamps to 2..3): turn 2 in scope -> image.
        # Both include. Hmm.
        # Try context_window = 0 for moment 1, and a larger detection range for moment 2.
        # Actually cleanest: use two moments, narrow context, one near and one far.
        # Fixture is only 3 turns. Use context_window=0 and manually expect both out.
        # Instead, shift the moments: use turn_start=1, turn_end=1 and turn_start=3, turn_end=3
        # with context_window=0 to get strict ranges [1,1] and [3,3]. Neither includes turn 2.
        # Then the image (anchor turn 2) is in NEITHER. Both entries should have no images.
        entries_zero = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=0, version="v4",
            with_screenshots=True,
        )
        # With context_window=0, neither moment window includes turn 2 (the image's anchor)
        for e in entries_zero:
            assert "images" not in e["request"]

        # With context_window=1, the first moment's window [1..2] DOES include turn 2,
        # and the second moment's window [2..3] also includes turn 2. Both get the image.
        entries_one = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=1, version="v4",
            with_screenshots=True,
        )
        for e in entries_one:
            assert e["request"]["images"] == [
                "deidentified/screenshots/099bf759-abcd/4.000.jpg"
            ]

    def test_no_images_when_flag_off(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.annotate import build_analysis_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.annotate as a
        monkeypatch.setattr(
            a, "load_prompt",
            lambda v, t: "P {brief_description} X {excerpt} X {turn_start} X {turn_end}",
        )

        entries = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=20, version="v4",
            with_screenshots=False,
        )
        for e in entries:
            assert "images" not in e["request"]
