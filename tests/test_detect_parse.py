"""Tests for detection result parsing."""
import pytest
import json
from annotator.core.detect import parse_detection_results


class TestParseDetectionResults:
    def test_valid_json(self):
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({
                    "detections": [{
                        "turn_start": 1,
                        "turn_end": 5,
                        "annotation_type": "scaffolding",
                        "situation": "test",
                        "suggested_cut_turn": 1,
                    }]
                }),
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        }
        result = parse_detection_results(raw)
        assert "conv1" in result
        assert len(result["conv1"]["detections"]) == 1
        assert result["conv1"]["detections"][0]["turn_start"] == 1

    def test_invalid_json(self):
        raw = {
            "conv1__scaffolding": {
                "text": "not json at all",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        }
        result = parse_detection_results(raw)
        assert "conv1" in result
        assert len(result["conv1"]["detections"]) == 0

    def test_error_entry(self):
        raw = {
            "conv1__scaffolding": {
                "error": "API error",
                "text": "",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        }
        result = parse_detection_results(raw)
        assert "conv1" in result
        assert len(result["conv1"]["detections"]) == 0

    def test_missing_suggested_cut_turn_defaults(self):
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({
                    "detections": [{
                        "turn_start": 5,
                        "turn_end": 10,
                        "annotation_type": "scaffolding",
                        "situation": "test",
                    }]
                }),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        }
        result = parse_detection_results(raw)
        det = result["conv1"]["detections"][0]
        assert det["suggested_cut_turn"] == 4

    def test_usage_accumulates(self):
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
            "conv1__rapport": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 200, "output_tokens": 75, "total_tokens": 275},
            },
        }
        result = parse_detection_results(raw)
        assert result["conv1"]["usage"]["input_tokens"] == 300
        assert result["conv1"]["usage"]["output_tokens"] == 125

    def test_image_counts_per_conv(self):
        """images_seen is the per-conv max (unique images);
        images_attached sums across targets (counts API attachments)."""
        raw = {
            "conv1__scaffolding": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
            "conv1__rapport": {
                "text": json.dumps({"detections": []}),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        }
        images_per_key = {"conv1__scaffolding": 3, "conv1__rapport": 3}
        result = parse_detection_results(raw, images_per_key=images_per_key)
        assert result["conv1"]["images_seen"] == 3       # unique per conv
        assert result["conv1"]["images_attached"] == 6   # 3 images * 2 targets


class TestBuildDetectionEntriesWithScreenshots:
    def test_includes_images_when_flag_set(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.detect import build_detection_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        # Stub prompt loader to return a predictable template
        import annotator.core.detect as d
        monkeypatch.setattr(d, "load_prompt", lambda v, t: "PROMPT: {transcript}")

        entries = build_detection_entries(
            [conv], targets=["scaffolding"], version="v5",
            with_screenshots=True,
        )
        assert len(entries) == 1
        # 4.000.jpg usable; 11.500.jpg filtered (eedi_ip=True)
        assert entries[0]["request"]["images"] == [
            "deidentified/screenshots/099bf759-abcd/4.000.jpg"
        ]
        # Text marker appears in the prompt. 4s falls inside turn 2 (starts at 3s).
        prompt_text = entries[0]["request"]["contents"][0]["parts"][0]["text"]
        assert "[SCREEN @ turn 2: image 1]" in prompt_text

    def test_no_images_when_flag_off(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.detect import build_detection_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.detect as d
        monkeypatch.setattr(d, "load_prompt", lambda v, t: "PROMPT: {transcript}")

        entries = build_detection_entries(
            [conv], targets=["scaffolding"], version="v5",
            with_screenshots=False,
        )
        assert "images" not in entries[0]["request"]


def test_build_detection_entries_uses_provided_screenshots(temp_data):
    from unittest.mock import patch
    from annotator.core.detect import build_detection_entries

    conversations = [{
        "conversation_id": "scen_abc",
        "turns": [{"turn_number": i, "role": "TUTOR", "text": f"t{i}",
                   "type": "DIALOGUE", "timestamp": "", "start_seconds": float(i)}
                  for i in range(1, 5)],
        "context": "ctx",
    }]
    fake = [{"filename": "s1.jpg", "anchor_turn": 2,
             "storage_path": "deidentified/screenshots/REAL/s1.jpg",
             "timestamp_seconds": 2.0}]
    screenshots_by_conv = {"scen_abc": fake}

    with patch("annotator.core.screenshots.load_anchored_screenshots") as mock_load:
        entries = build_detection_entries(
            conversations, targets=["scaffolding"], version="v4",
            with_screenshots=True, screenshots_by_conv=screenshots_by_conv,
        )

    mock_load.assert_not_called()
    assert len(entries) == 1
    assert entries[0]["request"].get("images") == ["deidentified/screenshots/REAL/s1.jpg"]
