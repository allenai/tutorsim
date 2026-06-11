"""Gold-moment loading must NOT silently skip moments missing annotation_type.

The old code did `moment.get("annotation_type", "")`, so a moment whose type
key was absent quietly fell out of the target set (`"" not in targets`) and was
dropped with no error. Every real ground_truth_hybrid moment carries
annotation_type, so a missing one is corruption that must surface.
"""

import pytest

import annotator.core.annotate as annotate_module


def _patch(monkeypatch, moment):
    monkeypatch.setattr(
        annotate_module, "load_ground_truth",
        lambda annotator_style=None: {
            "conversations": {"u1": {"num_turns": 3, "key_moments": [moment]}}
        },
    )
    monkeypatch.setattr(annotate_module, "load_split_ids", lambda split: {"u1"})
    monkeypatch.setattr(
        annotate_module, "load_all_transcripts",
        lambda: {"t1_s1_u1": {"transcript_id": "u1", "turns": []}},
    )


def test_gold_moment_missing_annotation_type_raises(monkeypatch):
    _patch(monkeypatch, {"turn_start": 1, "turn_end": 2})  # no annotation_type
    with pytest.raises(ValueError, match="annotation_type"):
        annotate_module.load_gold_moments(["scaffolding"])


def test_gold_scaffolding_missing_situation_label_agg_raises(monkeypatch):
    # annotate uses situation_label_agg to build the per-moment suggestion; a
    # scaffolding moment lacking it must fail loudly, not fall back silently.
    _patch(monkeypatch,
           {"turn_start": 1, "turn_end": 2, "annotation_type": "scaffolding"})
    with pytest.raises(ValueError, match="situation_label_agg"):
        annotate_module.load_gold_moments(["scaffolding"])


def test_valid_gold_moment_loads(monkeypatch):
    _patch(monkeypatch,
           {"turn_start": 1, "turn_end": 2, "annotation_type": "scaffolding",
            "situation_label_agg": "scaffolding"})
    out = annotate_module.load_gold_moments(["scaffolding"])
    assert "t1_s1_u1" in out
    assert out["t1_s1_u1"]["detections"][0]["annotation_type"] == "scaffolding"
