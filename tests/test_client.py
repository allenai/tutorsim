"""Tests for annotator.core.client pure functions."""
import pytest
from annotator.core.client import (
    infer_provider, _strip_json_fences, _extract_entry, build_batch_entry,
    _anthropic_thinking_param,
)


class TestAnthropicThinkingParam:
    def test_opus_4_8_uses_adaptive(self):
        # Opus 4.8 rejects the legacy enabled+budget shape; must be adaptive.
        param = _anthropic_thinking_param("claude-opus-4-8", 16384)
        assert param == {"type": "adaptive"}

    def test_opus_4_6_keeps_enabled_budget(self):
        param = _anthropic_thinking_param("claude-opus-4-6", 16384)
        assert param == {"type": "enabled", "budget_tokens": 16384}

    def test_budget_defaults_when_unset(self):
        param = _anthropic_thinking_param("claude-opus-4-6", 0)
        assert param == {"type": "enabled", "budget_tokens": 16384}


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
        key, prompt, json_mode, max_tokens, images = _extract_entry(entry)
        assert key == "my_key"
        assert prompt == "my prompt"
        assert json_mode is True
        assert max_tokens == 1000
        assert images == []


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


class TestAnthropicMultipleTextBlocks:
    """Anthropic can return more than one text block (e.g. after a thinking
    block, or interleaved output). The extractor must concatenate all of them;
    keeping only the first silently truncates the response."""

    def _block(self, btype, text):
        return type("Block", (), {"type": btype, "text": text})()

    def _fake_client(self, captured, content):
        from annotator.core.client import ModelClient

        class FakeResponse:
            class Usage:
                input_tokens = 1
                output_tokens = 1
            usage = Usage()

        FakeResponse.content = content

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
        return client

    def test_concatenates_text_blocks_after_thinking(self):
        content = [
            self._block("thinking", "internal reasoning, must be skipped"),
            self._block("text", '{"action": "the tutor '),
            self._block("text", 'broke the problem into steps"}'),
        ]
        client = self._fake_client({}, content)
        resp = client.generate("hello", json_mode=False)
        assert resp.text == '{"action": "the tutor broke the problem into steps"}'

    def test_single_text_block_unchanged(self):
        content = [self._block("text", "ok")]
        client = self._fake_client({}, content)
        resp = client.generate("hello", json_mode=False)
        assert resp.text == "ok"

    def test_helper_joins_text_blocks_and_skips_non_text(self):
        # Direct unit test of the shared extractor used by both the sync and
        # batch code paths (the batch path is otherwise hard to drive).
        from annotator.core.client import _extract_anthropic_text
        content = [
            self._block("thinking", "skip me"),
            self._block("text", "a"),
            self._block("text", "b"),
        ]
        assert _extract_anthropic_text(content) == "ab"
        assert _extract_anthropic_text([]) == ""


class TestBuildBatchEntryWithImages:
    def test_images_stored_in_request(self):
        from annotator.core.client import build_batch_entry
        entry = build_batch_entry("k", "p", images=["x/1.jpg", "x/2.jpg"])
        assert entry["request"]["images"] == ["x/1.jpg", "x/2.jpg"]

    def test_no_images_field_when_empty(self):
        from annotator.core.client import build_batch_entry
        entry = build_batch_entry("k", "p")
        assert "images" not in entry["request"]


class TestExtractEntryWithImages:
    def test_extract_returns_images(self):
        from annotator.core.client import build_batch_entry, _extract_entry
        entry = build_batch_entry("k", "p", images=["a.jpg"])
        key, prompt, json_mode, max_tokens, images = _extract_entry(entry)
        assert images == ["a.jpg"]

    def test_extract_empty_when_no_images(self):
        from annotator.core.client import build_batch_entry, _extract_entry
        entry = build_batch_entry("k", "p")
        _, _, _, _, images = _extract_entry(entry)
        assert images == []


class TestInterleaveTextAndImages:
    def _txt(self):
        return lambda s: {"type": "text", "text": s}

    def _imgs(self, n):
        return [{"type": "image", "id": i} for i in range(n)]

    def test_no_markers_no_images_returns_single_text(self):
        from annotator.core.client import _interleave_text_and_images
        out = _interleave_text_and_images("hello world", [], self._txt())
        assert out == [{"type": "text", "text": "hello world"}]

    def test_no_markers_with_orphan_images_appended_at_end(self):
        from annotator.core.client import _interleave_text_and_images
        out = _interleave_text_and_images("hello", self._imgs(2), self._txt())
        assert out == [
            {"type": "text", "text": "hello"},
            {"type": "image", "id": 0},
            {"type": "image", "id": 1},
        ]

    def test_one_marker_one_image(self):
        from annotator.core.client import _interleave_text_and_images
        prompt = "Turn 1. STUDENT: hi\nTurn 2. TUTOR: look\n  [SCREEN @ turn 2: image 1]\nTurn 3. STUDENT: ok"
        out = _interleave_text_and_images(prompt, self._imgs(1), self._txt())
        assert len(out) == 3
        assert out[0]["type"] == "text"
        assert out[0]["text"].endswith("[SCREEN @ turn 2: image 1]")
        assert out[1] == {"type": "image", "id": 0}
        assert out[2]["type"] == "text"
        assert out[2]["text"].startswith("\nTurn 3.")

    def test_multiple_markers_in_order(self):
        from annotator.core.client import _interleave_text_and_images
        prompt = (
            "Turn 1. TUTOR: a\n"
            "  [SCREEN @ turn 1: image 1]\n"
            "Turn 2. STUDENT: b\n"
            "Turn 3. TUTOR: c\n"
            "  [SCREEN @ turn 3: image 2]\n"
            "Turn 4. STUDENT: d"
        )
        out = _interleave_text_and_images(prompt, self._imgs(2), self._txt())
        # text, image0, text, image1, text
        types = [p.get("type") for p in out]
        assert types == ["text", "image", "text", "image", "text"]
        assert out[1]["id"] == 0
        assert out[3]["id"] == 1

    def test_marker_referencing_oob_index_kept_as_text_no_image_inserted(self):
        from annotator.core.client import _interleave_text_and_images
        prompt = "Turn 1.\n  [SCREEN @ turn 1: image 5]\nTurn 2."
        out = _interleave_text_and_images(prompt, self._imgs(1), self._txt())
        # Marker stays as text (prefix + trailing split at marker line). The single
        # image is orphan -> appended at end. No image is silently dropped.
        types = [p.get("type") for p in out]
        assert types == ["text", "text", "image"]
        assert "[SCREEN @ turn 1: image 5]" in out[0]["text"]
        assert out[1]["text"].startswith("\nTurn 2.")
        assert out[2]["id"] == 0

    def test_gemini_text_block_shape(self):
        from annotator.core.client import _interleave_text_and_images
        prompt = "a\n  [SCREEN @ turn 1: image 1]\nb"
        out = _interleave_text_and_images(
            prompt,
            [{"inline_data": {"mime_type": "image/jpeg", "data": "xx"}}],
            lambda s: {"text": s},
        )
        assert out[0] == {"text": "a\n  [SCREEN @ turn 1: image 1]"}
        assert "inline_data" in out[1]
        assert out[2] == {"text": "\nb"}

    def test_marker_with_extra_metadata_in_brackets_still_matches(self):
        # Future-proof: timestamp embedded in marker should not break interleaving.
        from annotator.core.client import _interleave_text_and_images
        prompt = "x\n  [SCREEN @ turn 5, t=603.8s: image 1]\ny"
        out = _interleave_text_and_images(prompt, self._imgs(1), self._txt())
        types = [p.get("type") for p in out]
        assert types == ["text", "image", "text"]


class TestGenerateWithImagesInterleaved:
    def test_anthropic_interleaves_at_marker_position(self, monkeypatch):
        """Image block lands immediately after its marker, not at the end."""
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

        import annotator.core.client as c
        monkeypatch.setattr(
            c, "_build_image_blocks_anthropic",
            lambda paths, use_url, enable_cache: [
                {"type": "image", "tag": f"img-{i}"} for i, _ in enumerate(paths)
            ],
        )
        monkeypatch.setattr(c, "_should_use_presigned_url", lambda: False)

        prompt = (
            "Turn 1. STUDENT: hi\n"
            "Turn 2. TUTOR: look\n"
            "  [SCREEN @ turn 2: image 1]\n"
            "Turn 3. STUDENT: ok"
        )
        client.generate(prompt, images=["a.jpg"], json_mode=False)
        content = captured["messages"][0]["content"]
        # Expect text-up-to-and-including-marker, then image, then trailing text.
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[0]["text"].endswith("[SCREEN @ turn 2: image 1]")
        assert content[1] == {"type": "image", "tag": "img-0"}
        assert content[2]["type"] == "text"
        assert content[2]["text"].startswith("\nTurn 3.")
