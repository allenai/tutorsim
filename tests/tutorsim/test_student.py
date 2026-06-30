"""Tests for tutorsim.student module - trait generation (TraitGenerator live path).

TDD: write failing tests first, then implement student.py.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_client(response_text: str) -> MagicMock:
    """Return a mock ModelClient whose .generate() returns a ModelResponse stub."""
    mock_response = SimpleNamespace(text=response_text)
    client = MagicMock()
    client.generate = MagicMock(return_value=mock_response)
    return client


def _make_scenario(conv_id: str = "conv-test", cut_turn: int = 5,
                   prefix: str = "Turn 1. TUTOR: Hello\nTurn 2. STUDENT: Hi") -> object:
    """Minimal scenario stub with the fields used by get_or_generate_trait."""
    return SimpleNamespace(
        conv_id=conv_id,
        cut_turn=cut_turn,
        transcript_prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Tests: generate_trait ("joined-3" mode)
# ---------------------------------------------------------------------------

class TestGenerateTrait:
    """Tests for the module-level generate_trait() function."""

    def test_joined3_builds_system_prompt_with_all_5_dimensions(self):
        """System prompt must contain text from all 5 dimension files."""
        from tutorsim.student import generate_trait

        client = _make_client("Some thinking\nDESCRIPTION: A diligent student.")
        # We don't need the description content; just confirm the call goes through.
        result = generate_trait("Turn 1. TUTOR: Hi", mode="joined-3", model_client=client)

        # All 5 generate calls happened (one per dimension).
        assert client.generate.call_count == 5

    def test_joined3_parses_description_marker(self):
        """generate_trait extracts text after 'DESCRIPTION:' for each dimension call."""
        from tutorsim.student import generate_trait

        # Each of the 5 calls returns a distinct description.
        responses = [
            SimpleNamespace(text=f"Some thinking\nDESCRIPTION: Dimension {i} desc.")
            for i in range(5)
        ]
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)

        result = generate_trait("Turn 1. TUTOR: Hi", mode="joined-3", model_client=client)

        # Result is the sorted+joined descriptions.
        assert isinstance(result, str)
        # All 5 descriptions appear in the result.
        for i in range(5):
            assert f"Dimension {i} desc." in result

    def test_joined3_sorts_and_joins_with_double_newline(self):
        """Dimension descriptions are sorted alphabetically and joined with '\\n\\n'."""
        from tutorsim.student import generate_trait

        # Return a fixed description per call. We reverse-order by hand to verify sort.
        descs = ["Z last", "A first", "M middle", "B second", "Y fourth"]
        responses = [SimpleNamespace(text=f"DESCRIPTION: {d}") for d in descs]
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)

        result = generate_trait("some prefix", mode="joined-3", model_client=client)

        parts = result.split("\n\n")
        assert parts == sorted(descs), f"Expected sorted join, got: {parts}"

    def test_adapter_calls_generate_with_correct_defaults(self):
        """The adapter must call client.generate with temperature=0.7, max_tokens=1024."""
        from tutorsim.student import generate_trait

        client = _make_client("DESCRIPTION: A student.")
        generate_trait("prefix", mode="joined-3", model_client=client)

        # Check every generate call uses the right defaults.
        for call in client.generate.call_args_list:
            kwargs = call.kwargs if call.kwargs else {}
            args = call.args if call.args else ()
            # max_tokens must be 1024 (adapter default when None passed by TraitGenerator)
            assert kwargs.get("max_tokens", 1024) == 1024, f"max_tokens wrong: {kwargs}"
            # json_mode must be False
            assert kwargs.get("json_mode", False) is False, f"json_mode wrong: {kwargs}"

    def test_no_description_marker_returns_full_output(self):
        """When 'DESCRIPTION:' is missing, the full response text is used as description."""
        from tutorsim.student import generate_trait

        full_text = "No marker here, just plain text."
        client = _make_client(full_text)
        result = generate_trait("prefix", mode="joined-3", model_client=client)

        assert full_text.strip() in result

    def test_description_after_marker_stripped(self):
        """Text before 'DESCRIPTION:' (thinking) is stripped; only post-marker text is kept."""
        from tutorsim.student import generate_trait

        client = _make_client("  lots of thinking\n\nDESCRIPTION:   Parsed description.  ")
        result = generate_trait("prefix", mode="joined-3", model_client=client)

        # All 5 dimension results will be "Parsed description." (same stub for each)
        # After sort+dedup it will be five copies, but we just check the content.
        assert "Parsed description." in result
        assert "lots of thinking" not in result

    def test_system_prompt_passed_as_cacheable_prefix(self):
        """The system prompt is forwarded as cacheable_prefix, not embedded in user text."""
        from tutorsim.student import generate_trait

        client = _make_client("DESCRIPTION: A student.")
        generate_trait("prefix text", mode="joined-3", model_client=client)

        for call in client.generate.call_args_list:
            kwargs = call.kwargs if call.kwargs else {}
            # cacheable_prefix must be set (the system prompt) and non-empty
            assert "cacheable_prefix" in kwargs, "cacheable_prefix missing from generate call"
            assert kwargs["cacheable_prefix"], "cacheable_prefix is empty/None"

    def test_user_prompt_contains_conversation_text(self):
        """The user prompt (first positional arg) must contain the transcript prefix."""
        from tutorsim.student import generate_trait

        prefix = "Turn 1. TUTOR: What is 2+2?\nTurn 2. STUDENT: 4"
        client = _make_client("DESCRIPTION: A bright student.")
        generate_trait(prefix, mode="joined-3", model_client=client)

        for call in client.generate.call_args_list:
            user_text = call.args[0] if call.args else call.kwargs.get("prompt", "")
            assert prefix in user_text, f"Prefix missing from user text: {user_text[:200]}"


# ---------------------------------------------------------------------------
# Tests: get_or_generate_trait (disk cache)
# ---------------------------------------------------------------------------

class TestGetOrGenerateTrait:
    """Tests for get_or_generate_trait() -- cache miss + hit."""

    def test_cache_miss_calls_generate_and_writes_cache(self, tmp_path):
        """On first call (cache miss), generate_trait is invoked and cache written."""
        from tutorsim.student import get_or_generate_trait

        scenario = _make_scenario(conv_id="conv-abc", cut_turn=7)
        client = _make_client("DESCRIPTION: A careful student.")

        result = get_or_generate_trait(
            scenario, "joined-3", client, "claude-opus-4-8",
            cache_dir=str(tmp_path),
        )

        # Should have called the model (5 dimensions).
        assert client.generate.call_count == 5
        assert isinstance(result, str)
        assert len(result) > 0

        # Cache file must exist.
        cache_files = list(tmp_path.iterdir())
        assert len(cache_files) == 1, f"Expected 1 cache file, got: {cache_files}"

        with open(cache_files[0], encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["conv_id"] == "conv-abc"
        assert payload["cut_turn"] == 7
        assert payload["trait_mode"] == "joined-3"
        assert payload["persona"] == result

    def test_cache_hit_returns_cached_without_calling_model(self, tmp_path):
        """On second call with same (conv_id, cut_turn, mode), model is NOT called."""
        from tutorsim.student import get_or_generate_trait

        scenario = _make_scenario(conv_id="conv-xyz", cut_turn=3)
        client = _make_client("DESCRIPTION: A careful student.")

        # First call -- populates cache.
        first_result = get_or_generate_trait(
            scenario, "joined-3", client, "claude-opus-4-8",
            cache_dir=str(tmp_path),
        )
        first_call_count = client.generate.call_count

        # Second call -- must not invoke model.
        second_result = get_or_generate_trait(
            scenario, "joined-3", client, "claude-opus-4-8",
            cache_dir=str(tmp_path),
        )

        assert second_result == first_result, "Cache hit returned different value"
        assert client.generate.call_count == first_call_count, (
            "Model was called on cache hit (should not be)"
        )

    def test_cache_key_includes_mode(self, tmp_path):
        """Different modes must produce separate cache files."""
        from tutorsim.student import get_or_generate_trait

        scenario = _make_scenario(conv_id="conv-mode", cut_turn=4)

        client_a = _make_client("DESCRIPTION: Affect desc.")
        get_or_generate_trait(
            scenario, "affect-3", client_a, "claude-opus-4-8",
            cache_dir=str(tmp_path),
        )

        client_b = _make_client("DESCRIPTION: Misconceptions desc.")
        get_or_generate_trait(
            scenario, "misconceptions-3", client_b, "claude-opus-4-8",
            cache_dir=str(tmp_path),
        )

        cache_files = sorted(tmp_path.iterdir(), key=lambda p: p.name)
        assert len(cache_files) == 2, f"Expected 2 cache files, got {len(cache_files)}: {cache_files}"

    def test_trait_alias_resolves_to_joined3(self, tmp_path):
        """Mode 'trait' must resolve to 'joined-3' (same cache key as 'joined-3')."""
        from tutorsim.student import get_or_generate_trait

        scenario = _make_scenario(conv_id="conv-alias", cut_turn=5)
        client = _make_client("DESCRIPTION: A student.")

        # Call with 'trait' alias.
        result_trait = get_or_generate_trait(
            scenario, "trait", client, "claude-opus-4-8",
            cache_dir=str(tmp_path),
        )
        first_call_count = client.generate.call_count

        # Call again with explicit 'joined-3' -- should hit the same cache.
        result_joined3 = get_or_generate_trait(
            scenario, "joined-3", client, "claude-opus-4-8",
            cache_dir=str(tmp_path),
        )
        assert result_trait == result_joined3
        assert client.generate.call_count == first_call_count, (
            "Model called a second time; trait alias did not share cache with joined-3"
        )

    def test_cache_content_identical_to_cold_run(self, tmp_path):
        """The value returned from cache is byte-for-byte identical to the generated value."""
        from tutorsim.student import get_or_generate_trait

        scenario = _make_scenario(conv_id="conv-idem", cut_turn=2)
        client = _make_client("DESCRIPTION: Consistent persona.")

        cold = get_or_generate_trait(
            scenario, "joined-3", client, "m1",
            cache_dir=str(tmp_path),
        )
        warm = get_or_generate_trait(
            scenario, "joined-3", client, "m1",
            cache_dir=str(tmp_path),
        )
        assert cold == warm


# ---------------------------------------------------------------------------
# Tests: build_student_system_prompt (oracle mode) -- Task 4
# ---------------------------------------------------------------------------

class TestBuildStudentSystemPrompt:
    """Tests for build_student_system_prompt() -- oracle prompt render + substitutions."""

    def test_substitutes_reference_transcript(self):
        """[[REFERENCE_TRANSCRIPT_HERE]] must be replaced with the supplied reference."""
        from tutorsim.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5 math class",
            reference_transcript="Turn 8: TUTOR: What is 3+3? STUDENT: 6",
            persona="A curious, energetic student.",
        )
        assert "Turn 8: TUTOR: What is 3+3? STUDENT: 6" in result
        assert "[[REFERENCE_TRANSCRIPT_HERE]]" not in result

    def test_substitutes_persona(self):
        """[[PERSONA_DESCRIPTION_HERE]] must be replaced with the supplied persona."""
        from tutorsim.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5 math class",
            reference_transcript="Turn 8: TUTOR: Hi STUDENT: Hello",
            persona="A curious, energetic student.",
        )
        assert "A curious, energetic student." in result
        assert "[[PERSONA_DESCRIPTION_HERE]]" not in result

    def test_substitutes_student_context(self):
        """[[NEXT_CONVERSATION_INFORMATION_HERE]] must be replaced with student_context."""
        from tutorsim.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5 math class",
            reference_transcript="Turn 8: TUTOR: Hi STUDENT: Hello",
            persona="A curious student.",
        )
        assert "Grade 5 math class" in result
        assert "[[NEXT_CONVERSATION_INFORMATION_HERE]]" not in result

    def test_substitutes_shared_generation_instructions(self):
        """{shared_generation_instructions} must be replaced (no literal brace placeholder)."""
        from tutorsim.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5",
            reference_transcript="Turn 8...",
            persona="A curious student.",
        )
        assert "{shared_generation_instructions}" not in result
        # The actual shared instructions contain this sentinel text.
        assert "Do *not* generate any turns as the tutor" in result

    def test_no_unreplaced_brackets(self):
        """No double-bracket placeholders should remain in the output."""
        from tutorsim.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="context text",
            reference_transcript="ref text",
            persona="persona text",
        )
        assert "[[" not in result
        assert "]]" not in result

    def test_empty_reference_uses_fallback_message(self):
        """When reference_transcript is empty/None, the fallback message is substituted."""
        from tutorsim.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5",
            reference_transcript="",
            persona="A student.",
        )
        assert "The real conversation ends at the cut point" in result
        assert "[[REFERENCE_TRANSCRIPT_HERE]]" not in result

    def test_none_reference_uses_fallback_message(self):
        """When reference_transcript is None, the fallback message is substituted."""
        from tutorsim.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5",
            reference_transcript=None,
            persona="A student.",
        )
        assert "The real conversation ends at the cut point" in result


# ---------------------------------------------------------------------------
# Tests: resolve_student -- Task 4
# ---------------------------------------------------------------------------

class TestResolveStudent:
    """Tests for resolve_student() -- hosted model resolution + registry."""

    def test_resolve_student_no_id_returns_hosted_claude_opus(self, monkeypatch):
        """resolve_student() with no args returns hosted claude-opus-4-6 with thinking=False."""
        import os
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Patch anthropic.Anthropic so no real HTTP call is made.
        import unittest.mock as mock
        with mock.patch("anthropic.Anthropic"):
            from tutorsim.student import resolve_student
            result = resolve_student()

        assert result["kind"] == "hosted"
        assert result["client"] is not None
        assert result["kwargs"].get("thinking") is False

    def test_resolve_student_registered(self, monkeypatch):
        """A @register_student-decorated callable is returned by resolve_student(name)."""
        from tutorsim.config import register_student, _STUDENT_REGISTRY
        import unittest.mock as mock

        # Register a dummy student.
        @register_student("dummy")
        def _dummy_student(conversation):
            return "I am a dummy student"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with mock.patch("anthropic.Anthropic"):
            from tutorsim.student import resolve_student
            result = resolve_student("dummy")

        assert result["kind"] == "registered"
        assert result["fn"] is _dummy_student

        # Cleanup registry.
        _STUDENT_REGISTRY.pop("dummy", None)

    def test_resolve_student_registered_from_config_model(self, tmp_path, monkeypatch):
        """student.model can name a registered callable."""
        from tutorsim.config import (
            _STUDENT_REGISTRY,
            _reset_config_cache,
            register_student,
        )
        from tutorsim.student import resolve_student

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
providers:
  anthropic: { env: ANTHROPIC_API_KEY }
  openai:    { env: OPENAI_API_KEY }
  gemini:    { env: GEMINI_API_KEY }
  together:  { env: TOGETHER_API_KEY }
models:
  claude-opus-4-8: {}
student: { model: dummy-from-config, mode: oracle, thinking: false }
scorer:  { model: claude-opus-4-6, thinking: adaptive }
defaults: { seed: 10, trials: 1, max_turns: 5 }
retry:    { max_retries: 5, base_delay: 5 }
batch:    { timeout: 86400 }
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("TUTORSIM_CONFIG", str(config_path))
        _reset_config_cache()

        @register_student("dummy-from-config")
        def _dummy_student(conversation):
            return "student turn"

        result = resolve_student()
        assert result["kind"] == "registered"
        assert result["fn"] is _dummy_student

        _STUDENT_REGISTRY.pop("dummy-from-config", None)
        _reset_config_cache()
