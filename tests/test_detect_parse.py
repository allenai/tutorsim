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
                        "brief_description": "test",
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
                        "brief_description": "test",
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
