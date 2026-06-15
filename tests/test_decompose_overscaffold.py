"""Tests for _build_overscaffold_prompt in annotator/core/decompose.py.

The over-scaffolding decomposition step extracts spans suggesting the tutor
over-scaffolded from an annotation's situation/action/result. These tests pin
down the prompt-building helper: it substitutes all three placeholders, and it
skips (returns None) only when there is genuinely nothing to analyze -- i.e.
both action and result are junk.
"""

from annotator.core.decompose import _build_overscaffold_prompt

TEMPLATE = (
    "Situation: {situation}\nAction: {action}\nResult: {result}"
)


def test_substitutes_all_placeholders():
    prompt = _build_overscaffold_prompt(
        "The student is stuck.",
        "The tutor explains the whole answer.",
        "The student copies it down.",
        TEMPLATE,
    )
    assert prompt == (
        "Situation: The student is stuck.\n"
        "Action: The tutor explains the whole answer.\n"
        "Result: The student copies it down."
    )
    assert "{situation}" not in prompt
    assert "{action}" not in prompt
    assert "{result}" not in prompt


def test_skips_when_both_action_and_result_are_junk():
    # "n/a" and "" are in JUNK_TEXTS -- nothing to analyze.
    assert _build_overscaffold_prompt("anything", "n/a", "", TEMPLATE) is None


def test_builds_when_only_result_is_junk():
    # Over-scaffolding signal can live in the action alone, so don't skip.
    prompt = _build_overscaffold_prompt(
        "The student is stuck.",
        "The tutor explains the whole answer.",
        "n/a",
        TEMPLATE,
    )
    assert prompt is not None
    assert "The tutor explains the whole answer." in prompt


def test_builds_when_only_action_is_junk():
    prompt = _build_overscaffold_prompt(
        "The student is stuck.",
        "",
        "The tutor ended up giving away the answer.",
        TEMPLATE,
    )
    assert prompt is not None
    assert "The tutor ended up giving away the answer." in prompt
