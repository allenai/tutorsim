"""Tests for student_mode dispatch in benchmark/core/exchange.py.

The synthetic student is built from `benchmark/core/students.py`, which
dispatches `student_mode` to one of the Python prompt classes in
`benchmark.synth_students.prompts` (verbatim port of Alexis's
synth-students repo). `prompts/benchmark/{version}/students/*.txt` files
are NOT loaded by this path -- the prompt text lives in the synth_students
classes. When student_mode is None, _build_role_prompt defaults to
"simple".
"""
from benchmark.core.exchange import _build_role_prompt


TRANSCRIPT = "Turn 1. TUTOR: hi\nTurn 2. STUDENT: hey"
CTX = "Grade 5, fractions"


def test_student_mode_none_defaults_to_simple():
    """No mode -> falls back to 'simple' (synth_students.SimpleMultiTurnStudentPrompt)."""
    head, tail = _build_role_prompt("STUDENT", transcript_prefix=TRANSCRIPT, extra="",
                             student_context=CTX, prompt_version="v6",
                             student_mode=None)
    out = head + tail
    assert "elementary school student" in out
    assert CTX in out
    assert TRANSCRIPT in out


def test_student_mode_imitate_example_uses_imitate_prompt_class():
    head, tail = _build_role_prompt("STUDENT", transcript_prefix=TRANSCRIPT, extra="",
                             student_context=CTX, prompt_version="v6",
                             student_mode="imitate_example")
    out = head + tail
    assert "imitate a human student" in out
    assert "indistinguishable" in out
    assert CTX in out
    assert TRANSCRIPT in out


def test_student_mode_simple_uses_simple_prompt_class():
    head, tail = _build_role_prompt("STUDENT", transcript_prefix=TRANSCRIPT, extra="",
                             student_context=CTX, prompt_version="v6",
                             student_mode="simple")
    out = head + tail
    assert "elementary school student" in out
    assert "imitate" not in out.lower()
    assert CTX in out


def test_student_mode_expert_uses_expert_prompt_class():
    head, tail = _build_role_prompt("STUDENT", transcript_prefix=TRANSCRIPT, extra="",
                             student_context=CTX, prompt_version="v6",
                             student_mode="expert")
    out = head + tail
    assert "very strong" in out
    assert "no mistakes" in out


def test_student_mode_paraphrase_uses_paraphrase_prompt_class():
    head, tail = _build_role_prompt("STUDENT", transcript_prefix=TRANSCRIPT, extra="",
                             student_context=CTX, prompt_version="v6",
                             student_mode="paraphrase_with_example")
    out = head + tail
    assert "paraphrase" in out.lower()
    assert TRANSCRIPT in out


def test_student_mode_does_not_affect_tutor_role():
    """student_mode is ignored when role == TUTOR."""
    head, tail = _build_role_prompt("TUTOR", transcript_prefix=TRANSCRIPT, extra="",
                             student_context=CTX, prompt_version="v2",
                             student_mode="imitate_example")
    out = head + tail
    assert "online tutor" in out
    assert "imitate" not in out.lower()
