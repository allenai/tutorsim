"""Tests for annotator.core.client pure functions."""
import pytest
from annotator.core.client import (
    infer_provider, _strip_json_fences, _extract_entry, build_batch_entry,
)


class TestInferProvider:
    def test_gemini(self):
        assert infer_provider("gemini-3.1-pro-preview") == "gemini"

    def test_openai_gpt(self):
        assert infer_provider("gpt-5.4") == "openai"

    def test_openai_o_series(self):
        assert infer_provider("o3-mini") == "openai"
        assert infer_provider("o4-mini") == "openai"

    def test_anthropic(self):
        assert infer_provider("claude-opus-4-6") == "anthropic"
        assert infer_provider("claude-sonnet-4-6") == "anthropic"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot infer provider"):
            infer_provider("llama-3")

    def test_case_insensitive(self):
        assert infer_provider("GEMINI-3.1-pro") == "gemini"
        assert infer_provider("Claude-Opus-4-6") == "anthropic"


class TestStripJsonFences:
    def test_no_fences(self):
        assert _strip_json_fences('{"key": "value"}') == '{"key": "value"}'

    def test_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_json_fences(text) == '{"key": "value"}'

    def test_bare_fences(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_json_fences(text) == '{"key": "value"}'

    def test_whitespace_preserved_inside(self):
        text = '```json\n{\n  "key": "value"\n}\n```'
        assert '"key": "value"' in _strip_json_fences(text)


class TestBuildBatchEntry:
    def test_basic_entry(self):
        entry = build_batch_entry("test_key", "test prompt")
        assert entry["key"] == "test_key"
        assert entry["request"]["contents"][0]["parts"][0]["text"] == "test prompt"

    def test_json_mode_default(self):
        entry = build_batch_entry("k", "p")
        gen_cfg = entry["request"]["generation_config"]
        assert gen_cfg["response_mime_type"] == "application/json"

    def test_json_mode_false(self):
        entry = build_batch_entry("k", "p", json_mode=False)
        gen_cfg = entry["request"]["generation_config"]
        assert "response_mime_type" not in gen_cfg


class TestExtractEntry:
    def test_round_trip(self):
        entry = build_batch_entry("my_key", "my prompt", json_mode=True, max_tokens=1000)
        key, prompt, json_mode, max_tokens = _extract_entry(entry)
        assert key == "my_key"
        assert prompt == "my prompt"
        assert json_mode is True
        assert max_tokens == 1000
