"""Tests for _parse_decomposed in annotator/core/decompose.py.

The decomposer prompts ask the model for a bare JSON array of facet strings.
That works for Gemini (response_mime_type) and Anthropic (soft system rule),
but OpenAI's response_format={"type": "json_object"} *cannot* emit a top-level
array, so the model wraps the facets in an object. These tests pin down that
the parser recovers facets from those object shapes instead of dropping them.
"""

from annotator.core.decompose import _parse_decomposed


def test_parses_plain_array():
    facets, had_error = _parse_decomposed('["a", "b"]')
    assert had_error is False
    assert facets == ["a", "b"]


def test_parses_array_embedded_in_prose():
    facets, had_error = _parse_decomposed('Here you go: ["a", "b"] done')
    assert had_error is False
    assert facets == ["a", "b"]


def test_empty_array_is_valid_no_facets():
    facets, had_error = _parse_decomposed("[]")
    assert had_error is False
    assert facets == []


def test_parses_wrapper_object_with_list_value():
    # The well-behaved OpenAI json_object shape.
    facets, had_error = _parse_decomposed('{"facets": ["a", "b"]}')
    assert had_error is False
    assert facets == ["a", "b"]


def test_recovers_facets_crammed_as_object_pairs():
    # The exact failure shape observed with the openai profile.
    text = (
        '{ "The student participates." : "The student identifies part of the reasoning.", '
        '"The student reaches the correct answer with support." : '
        '"The student is able to name the source of their error." }'
    )
    facets, had_error = _parse_decomposed(text)
    assert had_error is False
    assert set(facets) == {
        "The student participates.",
        "The student identifies part of the reasoning.",
        "The student reaches the correct answer with support.",
        "The student is able to name the source of their error.",
    }


def test_empty_object_is_valid_no_facets():
    facets, had_error = _parse_decomposed("{}")
    assert had_error is False
    assert facets == []


def test_wrapper_object_with_empty_list_is_empty_not_crammed():
    # Regression: opus-4-8 wraps the array as {"spans": []} / {"facets": []} for an
    # empty result. The cram-fallback must NOT fire on an empty list value, or the
    # key+"[]" get returned as two bogus facets -- which silently flips an empty
    # over-scaffold result into a (wrong) non-empty one.
    for text in ('{"spans": []}', '{"facets": []}', '{"result": []}'):
        facets, had_error = _parse_decomposed(text)
        assert had_error is False, text
        assert facets == [], text


def test_wrapper_object_with_spans_key_extracts_list():
    facets, had_error = _parse_decomposed('{"spans": ["a", "b"]}')
    assert had_error is False
    assert facets == ["a", "b"]


def test_unparseable_text_flags_error():
    facets, had_error = _parse_decomposed("not json at all")
    assert had_error is True
    assert facets == []
