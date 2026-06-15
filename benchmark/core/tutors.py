"""Build the tutor system prompt for a given mode.

Single source of truth for synthesizing the TUTOR side of an exchange.
Loads the appropriate prompt template from `prompts/benchmark/{version}/`
and substitutes the documented placeholders.

Modes (and their template paths):
- None / "default" -> prompts/benchmark/{version}/tutors/default.txt
                      (fallback for v5 and earlier: tutor_system.txt at the
                       version root, where the default tutor used to live
                       before the symmetric tutors/ layout was introduced in v6)
- "oracle"         -> prompts/benchmark/{version}/tutors/oracle.txt
                      (requires `reference_transcript`)

Placeholders:
- {student_context}        <- student_context (always available)
- {reference_transcript}   <- reference_transcript (oracle only)
"""
from __future__ import annotations

from pathlib import Path


_PROMPTS_BASE = Path(__file__).parent.parent.parent / "prompts" / "benchmark"


def _load_template(prompt_version: str, filename: str) -> str:
    path = _PROMPTS_BASE / prompt_version / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_default_tutor_template(prompt_version: str) -> str:
    """Load the default (non-oracle) tutor template.

    Prefers the symmetric `tutors/default.txt` layout (v6+); falls back to
    the legacy `tutor_system.txt` at the version root for older versions
    that predate the reorg. Keeps v5 (and older) reproducible without
    requiring a file move in archived prompt dirs.
    """
    new_layout = _PROMPTS_BASE / prompt_version / "tutors" / "default.txt"
    if new_layout.exists():
        return _load_template(prompt_version, "tutors/default.txt")
    return _load_template(prompt_version, "tutor_system.txt")


def build_tutor_system_prompt(
    tutor_mode: str | None,
    *,
    prompt_version: str,
    student_context: str,
    reference_transcript: str | None = None,
) -> str:
    """Assemble the TUTOR system prompt.

    Args:
        tutor_mode: None (or "default") for the default tutor; "oracle" for
            the reference-aware tutor that mimics the real human tutor's
            post-cut continuation.
        prompt_version: e.g. "v6". Selects the prompts subdirectory.
        student_context: substituted for `{student_context}`.
        reference_transcript: required when tutor_mode == "oracle".

    Returns:
        The fully-substituted system prompt string.
    """
    if tutor_mode in (None, "", "default"):
        template = _load_default_tutor_template(prompt_version)
    else:
        template = _load_template(prompt_version, f"tutors/{tutor_mode}.txt")

    out = template.replace("{student_context}", student_context or "")

    if tutor_mode == "oracle":
        if reference_transcript is None:
            raise ValueError(
                "tutor_mode='oracle' requires reference_transcript"
            )
        out = out.replace("{reference_transcript}", reference_transcript)

    return out
