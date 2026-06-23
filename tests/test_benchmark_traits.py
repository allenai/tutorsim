"""Tests for benchmark.core.traits: trait generator + per-scenario cache.

Updated for the current API where:
- `get_or_generate_trait(scenario, student_mode, model_client, model_name)`
  (no prompt_version -- TraitGenerator builds prompts in-code).
- `_trait_cache_filename(scenario, trait_mode)` (mode keyed in the filename).
- Persona generation goes through `synth_students.TraitGenerator` via the
  `ModelWrapperAdapter`; mocks therefore target the model client's `generate`.
"""
from unittest.mock import MagicMock

import pytest

from benchmark.core.traits import (
    get_or_generate_trait, _trait_cache_filename,
)
from benchmark.core.scenarios import Scenario


def _stub_response(text="A focused 5th grader who confuses long division steps."):
    resp = MagicMock()
    resp.text = text
    resp.usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    return resp


def _stub_client(text):
    client = MagicMock()
    client.model = "m1"
    client.generate.return_value = _stub_response(text)
    return client


def _scenario(conv_id="conv1", cut_turn=10, prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello"):
    return Scenario(
        scenario_id=f"{conv_id}__hum_x_y",
        conv_id=conv_id,
        cut_turn=cut_turn,
        transcript_prefix=prefix,
        student_context="Grade 5, math",
        last_student_message="hello",
        mode="human",
        detection={"turn_start": 5, "turn_end": 12,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def test_trait_cache_filename_uses_conv_id_cut_turn_and_mode():
    fname = _trait_cache_filename(_scenario(conv_id="abc123", cut_turn=42), "joined-3")
    assert "abc123" in fname
    assert "42" in fname
    assert "joined-3" in fname
    assert fname.endswith(".json")


def test_cache_miss_invokes_client_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    client = _stub_client("persona-A")

    s = _scenario(conv_id="conv-aa", cut_turn=7)
    persona = get_or_generate_trait(s, "affect-1", client, "m1")
    assert persona == "persona-A"
    assert client.generate.called

    cache_dir = tmp_path / "results" / "benchmark" / "_trait_cache"
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    import json
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["persona"] == "persona-A"
    assert saved["conv_id"] == "conv-aa"
    assert saved["cut_turn"] == 7
    assert saved["generator_model"] == "m1"
    assert saved["trait_mode"] == "affect-1"


def test_cache_hit_skips_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    s = _scenario(conv_id="conv-bb", cut_turn=3)

    primer = _stub_client("primed-persona")
    persona1 = get_or_generate_trait(s, "affect-1", primer, "m1")
    assert persona1 == "primed-persona"

    second = MagicMock()
    second.model = "m1"
    second.generate.side_effect = AssertionError("client should not be called on cache hit")
    persona2 = get_or_generate_trait(s, "affect-1", second, "m1")
    assert persona2 == "primed-persona"
    assert not second.generate.called


def test_generator_prompt_contains_only_prefix_no_post_cut_text(tmp_path, monkeypatch):
    """Oracle-leak guard: generator must only see transcript_prefix."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    captured = {}

    def _record_generate(prompt, **kw):
        captured["prompt"] = prompt
        captured["cacheable_prefix"] = kw.get("cacheable_prefix", "")
        return _stub_response("ok")

    client = MagicMock()
    client.model = "m1"
    client.generate = _record_generate

    s = _scenario(conv_id="conv-cc", cut_turn=5,
                  prefix="Turn 1. TUTOR: hi\nTurn 2. STUDENT: hello\nTurn 3. TUTOR: ok")
    SECRET = "POST_CUT_SECRET_TURN_42_TEXT"
    assert SECRET not in s.transcript_prefix

    get_or_generate_trait(s, "affect-1", client, "m1")

    full_input = captured["prompt"] + "\n" + captured["cacheable_prefix"]
    assert "Turn 1." in full_input
    assert "Turn 2." in full_input
    assert "Turn 3." in full_input
    assert SECRET not in full_input


def test_persona_caches_per_cut_turn(tmp_path, monkeypatch):
    """Same conv, different cuts -> different cache entries."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None

    client = MagicMock()
    client.model = "m1"
    client.generate.side_effect = [_stub_response("p-cut3"), _stub_response("p-cut9")]

    s_a = _scenario(conv_id="conv-dd", cut_turn=3)
    s_b = _scenario(conv_id="conv-dd", cut_turn=9)
    p1 = get_or_generate_trait(s_a, "affect-1", client, "m1")
    p2 = get_or_generate_trait(s_b, "affect-1", client, "m1")
    assert p1 == "p-cut3"
    assert p2 == "p-cut9"
    cache_dir = tmp_path / "results" / "benchmark" / "_trait_cache"
    assert len(list(cache_dir.glob("*.json"))) == 2
