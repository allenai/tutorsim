"""Tests for student_mode dispatch in benchmark/core/exchange.py.

The synthetic student can be configured to use one of several personas
(imitate_example, simple, expert, paraphrase_with_example) by loading
prompts/benchmark/{version}/students/{mode}.txt. When student_mode is
None or omitted, the loader falls back to the legacy student_system.txt
so older prompt versions (e.g. v1) keep working.
"""
from benchmark.core.exchange import _build_role_prompt


TRANSCRIPT = "Turn 1. TUTOR: hi\nTurn 2. STUDENT: hey"
CTX = "Grade 5, fractions"


def test_student_mode_none_falls_back_to_student_system_v1():
    """No mode -> legacy student_system.txt under v1."""
    out = _build_role_prompt("STUDENT", TRANSCRIPT, CTX, prompt_version="v1",
                             student_mode=None)
    assert "role-playing as a K-12 student" in out          # v1 student_system.txt
    assert CTX in out
    assert TRANSCRIPT in out


def test_student_mode_imitate_example_loads_v2_students_folder():
    out = _build_role_prompt("STUDENT", TRANSCRIPT, CTX, prompt_version="v2",
                             student_mode="imitate_example")
    assert "imitate a human K-12 student" in out
    assert "indistinguishable" in out
    assert CTX in out
    assert TRANSCRIPT in out


def test_student_mode_simple_loads_v2_students_folder():
    out = _build_role_prompt("STUDENT", TRANSCRIPT, CTX, prompt_version="v2",
                             student_mode="simple")
    assert "Respond like a K-12 student would" in out
    assert "imitate" not in out.lower()                      # not the imitate prompt
    assert CTX in out


def test_student_mode_expert_loads_v2_students_folder():
    out = _build_role_prompt("STUDENT", TRANSCRIPT, CTX, prompt_version="v2",
                             student_mode="expert")
    assert "very strong" in out
    assert "no mistakes" in out


def test_student_mode_paraphrase_loads_v2_students_folder():
    out = _build_role_prompt("STUDENT", TRANSCRIPT, CTX, prompt_version="v2",
                             student_mode="paraphrase_with_example")
    assert "paraphrase" in out.lower()
    assert TRANSCRIPT in out


def test_student_mode_does_not_affect_tutor_role():
    """student_mode is ignored when role == TUTOR."""
    out = _build_role_prompt("TUTOR", TRANSCRIPT, CTX, prompt_version="v2",
                             student_mode="imitate_example")
    assert "online tutor" in out
    assert "imitate" not in out.lower()
