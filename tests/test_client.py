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


class TestMimeFromPath:
    def test_jpg(self):
        from annotator.core.client import _mime_from_path
        assert _mime_from_path("foo/bar.jpg") == "image/jpeg"
        assert _mime_from_path("X.JPEG") == "image/jpeg"

    def test_png(self):
        from annotator.core.client import _mime_from_path
        assert _mime_from_path("a/b/c.png") == "image/png"

    def test_webp(self):
        from annotator.core.client import _mime_from_path
        assert _mime_from_path("x.webp") == "image/webp"

    def test_unknown_raises(self):
        from annotator.core.client import _mime_from_path
        with pytest.raises(ValueError, match="unknown image extension"):
            _mime_from_path("foo.bmp")


class TestValidateVisionSupport:
    def test_accepts_claude_opus_4(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("claude-opus-4-6")

    def test_accepts_gemini_3(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("gemini-3.1-pro-preview")

    def test_accepts_gpt5(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("gpt-5.4")

    def test_accepts_gpt_4o(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("gpt-4o-mini")

    def test_rejects_old_text_only(self):
        from annotator.core.client import validate_vision_support
        with pytest.raises(ValueError, match="not in the vision-capable list"):
            validate_vision_support("llama-3")

    def test_case_insensitive(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("CLAUDE-OPUS-4-6")


class TestImageBlocks:
    def _mock_backend(self, monkeypatch, is_local):
        """Stub out the storage backend for deterministic tests."""
        import annotator.core.client as c

        def fake_read_bytes(path):
            return b"bytes-for-" + path.encode()

        def fake_get_presigned_url(path, expires_seconds=None):
            return f"https://example.com/{path}?sig=x"

        class FakeBE:
            pass

        fake_be = FakeBE()
        fake_be.read_bytes = fake_read_bytes
        fake_be.get_presigned_url = fake_get_presigned_url

        class FakeLocal: pass
        class FakeS3: pass
        import annotator.core.storage as s
        monkeypatch.setattr(s, "_get_backend", lambda: fake_be)
        monkeypatch.setattr(
            s, "LocalBackend",
            FakeLocal if not is_local else type(fake_be)
        )

    def test_anthropic_local_inlines_base64(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=True)
        from annotator.core.client import _build_image_blocks_anthropic
        blocks = _build_image_blocks_anthropic(["x/1.jpg"], use_url=False, enable_cache=False)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["type"] == "base64"
        assert blocks[0]["source"]["media_type"] == "image/jpeg"
        assert "data" in blocks[0]["source"]

    def test_anthropic_s3_uses_url(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=False)
        from annotator.core.client import _build_image_blocks_anthropic
        blocks = _build_image_blocks_anthropic(["x/1.jpg"], use_url=True, enable_cache=False)
        assert blocks[0]["source"]["type"] == "url"
        assert blocks[0]["source"]["url"].startswith("https://")

    def test_anthropic_cache_control(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=True)
        from annotator.core.client import _build_image_blocks_anthropic
        blocks = _build_image_blocks_anthropic(["x/1.jpg"], use_url=False, enable_cache=True)
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_openai_local_uses_data_url(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=True)
        from annotator.core.client import _build_image_blocks_openai
        blocks = _build_image_blocks_openai(["x/1.jpg"], use_url=False)
        assert blocks[0]["type"] == "image_url"
        assert blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_openai_s3_uses_https(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=False)
        from annotator.core.client import _build_image_blocks_openai
        blocks = _build_image_blocks_openai(["x/1.jpg"], use_url=True)
        assert blocks[0]["image_url"]["url"].startswith("https://")

    def test_gemini_always_inlines(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=False)  # even on S3
        from annotator.core.client import _build_image_blocks_gemini
        blocks = _build_image_blocks_gemini(["x/1.jpg"])
        assert "inline_data" in blocks[0]
        assert blocks[0]["inline_data"]["mime_type"] == "image/jpeg"
        assert "data" in blocks[0]["inline_data"]


class TestGenerateWithImages:
    def test_anthropic_sends_image_blocks(self, monkeypatch):
        """generate() with images wraps them into Anthropic content blocks."""
        from annotator.core.client import ModelClient

        captured = {}

        class FakeResponse:
            class Usage:
                input_tokens = 1
                output_tokens = 1
            usage = Usage()
            content = [type("T", (), {"type": "text", "text": "ok"})()]

        class FakeAnthropic:
            class messages:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return FakeResponse()

        client = ModelClient.__new__(ModelClient)
        client.model = "claude-opus-4-6"
        client.provider = "anthropic"
        client._client = FakeAnthropic()

        # Stub image block builder to avoid storage calls
        import annotator.core.client as c
        monkeypatch.setattr(
            c, "_build_image_blocks_anthropic",
            lambda paths, use_url, enable_cache: [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "xx"}}
            ],
        )
        monkeypatch.setattr(c, "_should_use_presigned_url", lambda: False)

        resp = client.generate("hello", images=["foo.jpg"], json_mode=False)
        assert resp.text == "ok"
        content = captured["messages"][0]["content"]
        # Multimodal content is a list of blocks: text block + image block(s)
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "hello"
        assert content[1]["type"] == "image"
