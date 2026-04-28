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

        # Fixture: turns at start_seconds 0/3/10, screenshot at 4.000s -> anchors to turn 2.
        # Moments are at turn 1 and turn 3.
        # context_window=0 -> windows [1,1] and [3,3]: neither includes turn 2 -> no images.
        # context_window=1 -> windows [1,2] and [2,3]: both include turn 2 -> image attached.
        entries_zero = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=0, version="v4",
            with_screenshots=True,
        )
        for e in entries_zero:
            assert "images" not in e["request"]

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


def test_build_analysis_entries_uses_provided_screenshots(temp_data, monkeypatch):
    """When screenshots_by_conv is passed, the function uses it directly
    and does NOT call load_anchored_screenshots."""
    from unittest.mock import patch
    import annotator.core.annotate as a
    from annotator.core.annotate import build_analysis_entries

    monkeypatch.setattr(
        a, "load_prompt",
        lambda v, t: "P {brief_description} X {excerpt} X {turn_start} X {turn_end}",
    )

    detections_by_conv = {
        "scen_abc": {"detections": [
            {"turn_start": 5, "turn_end": 7,
             "annotation_type": "scaffolding", "brief_description": "x"}
        ]},
    }
    conversations_map = {
        "scen_abc": {
            "conversation_id": "scen_abc",
            "turns": [
                {"turn_number": i, "role": "TUTOR", "text": f"t{i}",
                 "type": "DIALOGUE", "timestamp": "", "start_seconds": float(i)}
                for i in range(1, 11)
            ],
        },
    }
    fake_screenshots = [
        {"filename": "s1.jpg", "anchor_turn": 6, "storage_path": "deidentified/screenshots/REAL_CONV/s1.jpg",
         "timestamp_seconds": 6.0},
    ]
    screenshots_by_conv = {"scen_abc": fake_screenshots}

    with patch("annotator.core.screenshots.load_anchored_screenshots") as mock_load:
        entries = build_analysis_entries(
            detections_by_conv, conversations_map,
            context_window=2, version="v4",
            with_screenshots=True,
            screenshots_by_conv=screenshots_by_conv,
        )
    mock_load.assert_not_called()
    assert len(entries) == 1
    request = entries[0]["request"]
    assert request.get("images") == ["deidentified/screenshots/REAL_CONV/s1.jpg"]
