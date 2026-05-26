"""Per-type labeller routing tests.

The labeller config can be either a string (single template for all
annotation_types) or a dict (per-type routing). Verify both shapes load
correctly and `pick_template` falls back sensibly.
"""

import pytest

from annotator.core.label import load_labeller_templates, pick_template


def test_string_config_loads_single_template():
    templates = load_labeller_templates("classify_v2")
    assert list(templates.keys()) == [None]
    assert "classify" in templates[None].lower() or "{annotation_type}" in templates[None]


def test_dict_config_loads_per_type():
    templates = load_labeller_templates(
        {"scaffolding": "classify_scaffolding", "rapport": "classify_rapport"}
    )
    assert set(templates.keys()) == {"scaffolding", "rapport"}
    # The two prompts should not be identical -- they encode different rules
    assert templates["scaffolding"] != templates["rapport"]


def test_dict_config_loads_actual_active_prompts():
    """Active labeller files must exist and have the {annotation_type} placeholder."""
    templates = load_labeller_templates(
        {"scaffolding": "classify_scaffolding", "rapport": "classify_rapport"}
    )
    for ann_type, template in templates.items():
        assert "{annotation_type}" in template
        assert "{situation}" in template
        assert "{action}" in template
        assert "{result_text}" in template


def test_pick_template_routes_per_type():
    templates = {
        "scaffolding": "SCAF_TEMPLATE",
        "rapport": "RAP_TEMPLATE",
    }
    assert pick_template(templates, "scaffolding") == "SCAF_TEMPLATE"
    assert pick_template(templates, "rapport") == "RAP_TEMPLATE"


def test_pick_template_falls_back_to_none_key():
    templates = {None: "DEFAULT_TEMPLATE"}
    assert pick_template(templates, "scaffolding") == "DEFAULT_TEMPLATE"
    assert pick_template(templates, "rapport") == "DEFAULT_TEMPLATE"
    assert pick_template(templates, "anything") == "DEFAULT_TEMPLATE"


def test_pick_template_raises_when_unmapped_and_no_fallback():
    templates = {"scaffolding": "SCAF_ONLY"}
    with pytest.raises(KeyError):
        pick_template(templates, "rapport")
