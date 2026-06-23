"""Tests for _parse_action_label / _parse_result_label's tolerance of
verbose/malformed model output.

classify_action.md asks for JSON {"scaffolding": "yes"|"no", "rigor":
"yes"|"no"} -- two independent per-dimension judgments. _parse_action_label
maps the resulting (scaffolding, rigor) tuple to a single action_label via
_YES_NO_TO_ACTION_LABEL, tolerating list-wrapped JSON, markdown code fences,
and verbose responses that bury the JSON in surrounding prose (mirroring
situate._parse_situation_label's tolerance for the same JSON shape).

classify_student_result.md asks for a single bare letter ("A"/"B"); models
sometimes ignore that and respond with markdown emphasis and an explanation,
e.g. "**A**\n\nThe statements indicate...". _parse_result_label recovers the
verdict stated up front rather than rejecting these as "unclear".
"""

from annotator.core.structure import _parse_action_label, _parse_result_label


# --- _parse_action_label -----------------------------------------------------

def test_parses_yes_no_json_as_scaffolding():
    assert _parse_action_label('{"scaffolding": "yes", "rigor": "no"}') == ("scaffolding", False)


def test_parses_no_yes_json_as_rigor():
    assert _parse_action_label('{"scaffolding": "no", "rigor": "yes"}') == ("rigor", False)


def test_parses_yes_yes_json_as_both():
    assert _parse_action_label('{"scaffolding": "yes", "rigor": "yes"}') == ("both", False)


def test_parses_no_no_json_as_neither():
    assert _parse_action_label('{"scaffolding": "no", "rigor": "no"}') == ("neither", False)


def test_unwraps_list_wrapped_json():
    assert _parse_action_label('[{"scaffolding": "yes", "rigor": "no"}]') == ("scaffolding", False)


def test_recovers_fields_from_json_wrapped_in_markdown_fence():
    text = '```json\n{"scaffolding": "no", "rigor": "yes"}\n```'
    assert _parse_action_label(text) == ("rigor", False)


def test_recovers_fields_from_verbose_response_with_surrounding_prose():
    text = ("Here is my analysis of the moment:\n\n"
            '{"scaffolding": "yes", "rigor": "no"}\n\n'
            "The tutor breaks the problem into smaller steps.")
    assert _parse_action_label(text) == ("scaffolding", False)


def test_missing_dimension_falls_back_to_unclear():
    # No "rigor" key -- a half-parsed answer can't be mapped to a verdict.
    label, had_error = _parse_action_label('{"scaffolding": "yes"}')
    assert label == "unclear"
    assert had_error is True


def test_invalid_dimension_value_falls_back_to_unclear():
    label, had_error = _parse_action_label('{"scaffolding": "maybe", "rigor": "no"}')
    assert label == "unclear"
    assert had_error is True


def test_unparseable_response_falls_back_to_unclear():
    label, had_error = _parse_action_label("I'm not sure what to call this.")
    assert label == "unclear"
    assert had_error is True


# --- _parse_result_label ----------------------------------------------------
#
# The student-result classifier prompt's output format changed from a
# {"A": "yes"|"no", "B": ..., "C": ...} JSON dict (three independent
# dimensions) to a single bare "A" or "B" -- a mutually exclusive choice
# between "trending toward understanding" (A) and "misconceptions
# predominantly remain" (B). _parse_result_label now mirrors
# _parse_action_label: map the bare letter to a semantic label, tolerating
# markdown emphasis and verbose explanations that state the verdict up front.

def test_parses_exact_bare_letter_a_as_pos():
    assert _parse_result_label("A") == ("pos", False)


def test_parses_exact_bare_letter_b_as_neg():
    assert _parse_result_label("B") == ("neg", False)


def test_parses_lowercase_and_trailing_period():
    assert _parse_result_label("a.") == ("pos", False)


def test_strips_markdown_emphasis_around_bare_letter():
    assert _parse_result_label("**B**") == ("neg", False)


def test_recovers_letter_stated_up_front_in_verbose_response():
    text = ("**A**\n\nThe statements indicate the student successfully "
            "completed the derivation and explained why each step followed")
    assert _parse_result_label(text) == ("pos", False)


def test_does_not_guess_when_letter_only_appears_inside_reasoning():
    # The verdict ("B") comes after "A" appears in the reasoning -- picking
    # the first letter-like token anywhere would wrongly return "pos".
    text = "Although there are traces of demonstrated understanding (A), the answer is B."
    label, had_error = _parse_result_label(text)
    assert label == "unclear"
    assert had_error is True


def test_unparseable_result_response_falls_back_to_unclear():
    label, had_error = _parse_result_label("I can't determine the outcome here.")
    assert label == "unclear"
    assert had_error is True


def test_whitespace_only_response_falls_back_to_unclear():
    # A whitespace-only response is truthy, so the caller's `not text` guard
    # doesn't catch it -- the parser must not crash on an empty splitlines().
    label, had_error = _parse_result_label("   \n  ")
    assert label == "unclear"
    assert had_error is True
