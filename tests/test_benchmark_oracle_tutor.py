"""Tests for oracle tutor mode."""
from unittest.mock import MagicMock

import pytest

from benchmark.core.exchange import (
    _build_role_prompt, _build_reference_transcript, run_exchange,
)
from benchmark.core.scenarios import Scenario


def _conv_with_turns(num_turns=10):
    return {
        "conversation_id": "conv1",
        "turns": [
            {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
             "text": f"real-turn-{n}"} for n in range(1, num_turns + 1)
        ],
    }


def _scenario(conv_id="conv1", cut_turn=4):
    return Scenario(
        scenario_id="s1",
        conv_id=conv_id,
        cut_turn=cut_turn,
        transcript_prefix="Turn 1. TUTOR: real-turn-1\nTurn 2. STUDENT: real-turn-2\nTurn 3. TUTOR: real-turn-3\nTurn 4. STUDENT: real-turn-4",
        student_context="Grade 5",
        last_student_message="real-turn-4",
        mode="human",
        detection={"turn_start": 2, "turn_end": 5,
                   "annotation_type": "scaffolding", "situation": "x"},
    )


def _stub_client(replies):
    client = MagicMock()
    client.model = "stub"
    it = iter(replies)
    def _gen(*args, **kwargs):
        resp = MagicMock()
        resp.text = next(it)
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    client.generate = _gen
    return client


def test_build_reference_transcript_returns_post_cut_turns_only():
    conv = _conv_with_turns(num_turns=8)
    ref = _build_reference_transcript(conv, cut_turn=4)
    assert "Turn 1." not in ref
    assert "Turn 4." not in ref
    assert "Turn 5." in ref
    assert "Turn 6." in ref
    assert "Turn 7." in ref
    assert "Turn 8." in ref
    assert "Turn 5. TUTOR: real-turn-5" in ref


def test_build_reference_transcript_empty_when_cut_is_last_turn():
    conv = _conv_with_turns(num_turns=4)
    ref = _build_reference_transcript(conv, cut_turn=4)
    assert ref == ""


def test_build_role_prompt_oracle_mode_substitutes_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: (
            "ORACLE ctx={student_context} ref={reference_transcript}"
            if "tutors/oracle" in fname
            else "DEFAULT-TUTOR {student_context}"
        ),
    )
    head, tail = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="",
        student_context="ctx5",
        prompt_version="v5",
        tutor_mode="oracle",
        reference_transcript="Turn 2. STUDENT: please help",
    )
    assert "ORACLE ctx=ctx5" in head
    assert "ref=Turn 2. STUDENT: please help" in head
    assert "Turn 2. STUDENT: please help" not in tail


def test_build_role_prompt_oracle_without_reference_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "ORACLE {student_context} {reference_transcript}",
    )
    with pytest.raises(ValueError, match="reference_transcript"):
        _build_role_prompt(
            "TUTOR",
            transcript_prefix="Turn 1. TUTOR: hi",
            extra="",
            student_context="ctx",
            prompt_version="v5",
            tutor_mode="oracle",
            reference_transcript=None,
        )


def test_build_role_prompt_tutor_mode_unset_uses_default_prompt(tmp_path, monkeypatch):
    """Back-compat: tutor_mode=None loads the legacy tutor_system.txt."""
    monkeypatch.chdir(tmp_path)
    loaded = []
    def _loader(version, fname):
        loaded.append(fname)
        return "DEFAULT {student_context}"
    monkeypatch.setattr("benchmark.core.tutors._load_template", _loader)

    head, tail = _build_role_prompt(
        "TUTOR",
        transcript_prefix="Turn 1. TUTOR: hi",
        extra="",
        student_context="ctx",
        prompt_version="v5",
    )
    assert any(f == "tutor_system.txt" for f in loaded)
    assert not any("tutors/" in f for f in loaded)


def test_run_exchange_oracle_mode_passes_reference_in_head(tmp_path, monkeypatch):
    """run_exchange with tutor_mode='oracle' must put the post-cut reference
    in the cacheable_prefix passed to the tutor client."""
    monkeypatch.chdir(tmp_path)

    def _loader(version, fname):
        if "tutors/oracle" in fname:
            return "ORACLE {student_context} REF={reference_transcript}"
        return "DEFAULT {student_context}"
    monkeypatch.setattr("benchmark.core.tutors._load_template", _loader)

    seen_prefixes = []
    def _tutor_generate(prompt, **kwargs):
        seen_prefixes.append(kwargs.get("cacheable_prefix"))
        resp = MagicMock()
        resp.text = "Wrap up. [NEXT_PROBLEM]"
        resp.usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return resp
    tutor = MagicMock(); tutor.model = "stub"; tutor.generate = _tutor_generate
    student = _stub_client([])

    scenario = _scenario(conv_id="conv1", cut_turn=4)
    transcripts = {"conv1": _conv_with_turns(num_turns=8)}

    ex = run_exchange(
        scenario=scenario, tutor_client=tutor, student_client=student,
        max_turns=10, tutor_max_tokens=128, student_max_tokens=128,
        prompt_version="v5",
        tutor_mode="oracle",
        transcripts=transcripts,
    )
    assert ex.completed is True
    assert any("Turn 5. TUTOR: real-turn-5" in (p or "") for p in seen_prefixes)
    assert any("Turn 8. STUDENT: real-turn-8" in (p or "") for p in seen_prefixes)


def test_run_exchange_oracle_without_transcripts_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "X {student_context} {reference_transcript}",
    )
    tutor = _stub_client(["Hi"])
    student = _stub_client([])

    with pytest.raises(ValueError, match="oracle"):
        run_exchange(
            scenario=_scenario(), tutor_client=tutor, student_client=student,
            max_turns=10, tutor_max_tokens=128, student_max_tokens=128,
            prompt_version="v5",
            tutor_mode="oracle",
            transcripts=None,
        )


def test_run_exchange_tutor_mode_none_ignores_transcripts(tmp_path, monkeypatch):
    """Legacy back-compat: when tutor_mode is None, no reference is loaded
    even if transcripts is passed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "benchmark.core.tutors._load_template",
        lambda version, fname: "DEFAULT {student_context}",
    )
    tutor = _stub_client(["Done. [NEXT_PROBLEM]"])
    student = _stub_client([])

    ex = run_exchange(
        scenario=_scenario(), tutor_client=tutor, student_client=student,
        max_turns=10, tutor_max_tokens=128, student_max_tokens=128,
        prompt_version="v5",
        transcripts={"conv1": _conv_with_turns(num_turns=8)},
    )
    assert ex.completed is True
    assert ex.ended_via == "PROBLEM_CHANGE"
