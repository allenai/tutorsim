"""Build the student system prompt for a given mode.

Single source of truth for synthesizing the STUDENT side of an exchange.
Dispatches `student_mode` to one of the verbatim-ported synth-students prompt
classes in `benchmark.synth_students.prompts` and substitutes the placeholders.

Modes (mirrors synth-students):
- "simple"                   -> SimpleMultiTurnStudentPrompt
- "expert"                   -> ExpertMultiTurnStudentPrompt
- "imitate_example"          -> ImitateExampleMultiTurnStudentPrompt
- "paraphrase_with_example"  -> ParaphraseWithExampleMultiTurnStudentPrompt
- "trait" or "<dim>-<n>" or "joined-<n>"  -> TraitMultiTurnStudentPrompt
- "trait_with_example"       -> TraitWithExampleMultiTurnStudentPrompt
- "oracle"                   -> OracleMomentStudentPrompt
                                  Sees the post-cut real student turns within
                                  the moment range (turn_end-bounded) and is
                                  asked to imitate that specific student's
                                  behavior in this moment. Post-cut aware --
                                  student-side analog of the oracle tutor.

Placeholders consumed (one or more per mode):
- [[NEXT_CONVERSATION_INFORMATION_HERE]]  <- student_context
- [[EXAMPLE_CONVERSATION_HERE]]           <- transcript_prefix
- [[PERSONA_DESCRIPTION_HERE]]            <- persona (trait mode)
- [[STUDENT_DESCRIPTION_HERE]]            <- persona (trait_with_example)
- [[MOMENT_REFERENCE_HERE]]               <- moment_reference (oracle mode)

The function is pure: no file I/O, no model clients, no scenario object.
Trait persona generation lives in `benchmark.core.traits` (cache wrapper
around synth_students.TraitGenerator). Oracle moment reference is built
by the caller (exchange.py) from the conversation + scenario.detection.turn_end.
"""
from __future__ import annotations

from benchmark.synth_students.prompts import (
    SimpleMultiTurnStudentPrompt,
    ExpertMultiTurnStudentPrompt,
    ImitateExampleMultiTurnStudentPrompt,
    OracleMomentStudentPrompt,
    ParaphraseWithExampleMultiTurnStudentPrompt,
    TraitMultiTurnStudentPrompt,
    TraitWithExampleMultiTurnStudentPrompt,
)
from benchmark.synth_students.dimension import ALL_DIMENSION_NAMES


_NEEDS_EXAMPLE = {"imitate_example", "paraphrase_with_example", "trait_with_example"}
_NEEDS_PERSONA_TRAIT = "trait"
_NEEDS_PERSONA_TRAIT_WITH_EXAMPLE = "trait_with_example"
_NEEDS_MOMENT_REFERENCE = "oracle"


def is_trait_mode(student_mode: str) -> bool:
    """True if student_mode requires a generated persona.

    Accepts either the literal 'trait'/'trait_with_example' modes or any
    '<dimension>-<n>' / 'joined-<n>' form (e.g. 'distractedness-3', 'joined-2').
    """
    if student_mode in (_NEEDS_PERSONA_TRAIT, _NEEDS_PERSONA_TRAIT_WITH_EXAMPLE):
        return True
    base = student_mode.split("-", 1)[0] if "-" in student_mode else student_mode
    return base in ALL_DIMENSION_NAMES or base == "joined"


def _resolve_prompt_class(student_mode: str):
    """Return the synth-students Prompt class for `student_mode`."""
    if student_mode == "simple":
        return SimpleMultiTurnStudentPrompt
    if student_mode == "expert":
        return ExpertMultiTurnStudentPrompt
    if student_mode == "imitate_example":
        return ImitateExampleMultiTurnStudentPrompt
    if student_mode == "paraphrase_with_example":
        return ParaphraseWithExampleMultiTurnStudentPrompt
    if student_mode == _NEEDS_PERSONA_TRAIT_WITH_EXAMPLE:
        return TraitWithExampleMultiTurnStudentPrompt
    if student_mode == _NEEDS_MOMENT_REFERENCE:
        return OracleMomentStudentPrompt
    # Trait family: bare "trait" OR "<dim>-<n>" / "joined-<n>"
    if is_trait_mode(student_mode):
        return TraitMultiTurnStudentPrompt
    raise ValueError(f"unknown student_mode: {student_mode!r}")


def build_student_system_prompt(
    student_mode: str,
    *,
    student_context: str,
    transcript_prefix: str,
    persona: str | None = None,
    moment_reference: str | None = None,
    num_turns: int | None = None,
) -> str:
    """Assemble the STUDENT system prompt for `student_mode`.

    Args:
        student_mode: one of the modes documented at module level.
        student_context: text to substitute for [[NEXT_CONVERSATION_INFORMATION_HERE]].
        transcript_prefix: text to substitute for [[EXAMPLE_CONVERSATION_HERE]]
            on modes that need an example.
        persona: trait persona text (required for trait modes).
        moment_reference: real student post-cut turns within the moment range
            (required for oracle mode). Built by the caller from
            conversation + scenario.detection.turn_end.
        num_turns: passed through to the prompt class constructor.

    Returns:
        The fully-substituted system prompt string.
    """
    cls = _resolve_prompt_class(student_mode)
    raw = cls(num_turns=num_turns).get_system_prompt()

    out = raw.replace("[[NEXT_CONVERSATION_INFORMATION_HERE]]", student_context or "")

    if student_mode in _NEEDS_EXAMPLE:
        if not transcript_prefix:
            raise ValueError(
                f"student_mode={student_mode!r} requires a transcript_prefix "
                "to substitute [[EXAMPLE_CONVERSATION_HERE]]"
            )
        out = out.replace("[[EXAMPLE_CONVERSATION_HERE]]", transcript_prefix)

    if is_trait_mode(student_mode):
        if not persona:
            raise ValueError(
                f"student_mode={student_mode!r} requires a persona to "
                "substitute [[PERSONA_DESCRIPTION_HERE]] / [[STUDENT_DESCRIPTION_HERE]]"
            )
        out = out.replace("[[PERSONA_DESCRIPTION_HERE]]", persona)
        out = out.replace("[[STUDENT_DESCRIPTION_HERE]]", persona)

    if student_mode == _NEEDS_MOMENT_REFERENCE:
        # moment_reference can legitimately be empty when the scenario was
        # role-adjusted (cut_turn = turn_end-1) or when the moment span is so
        # tight there are no further turns to show. Substitute a noop string
        # rather than raising -- the LM just has no in-moment reference and
        # falls back to its general "imitate a K-12 student" instruction.
        ref = moment_reference or "(no further turns in this moment)"
        out = out.replace("[[MOMENT_REFERENCE_HERE]]", ref)

    return out
