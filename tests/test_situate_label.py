"""Tests for annotator.core.situate output filename naming.

situation_labels_{...}_scaffolding.json must encode --gold the same way
decompose.py / structure.py do -- otherwise a gold run and a non-gold run
(same profile/style/split) write to the same output file and silently
overwrite each other.
"""

import annotator.core.situate as situate_module
from annotator.core.utils import load_split_ids


def _run_and_capture_filename(monkeypatch, **run_kwargs):
    saved = {}

    def fake_save(version, filename, data):
        saved["filename"] = filename

    monkeypatch.setattr(situate_module, "save_annotator_result", fake_save)
    monkeypatch.setattr(situate_module, "ModelClient", lambda model: object())
    monkeypatch.setattr(situate_module, "run_sync_entries", lambda *a, **kw: {})
    monkeypatch.setattr(situate_module, "_load_prompt", lambda: "TEMPLATE")

    train_id = next(iter(load_split_ids("train")))
    conv_id = f"2024-t1_2024-s1_{train_id}"
    # Empty situation is junk -> skipped, so no model call is needed.
    annotations_data = {
        "results": {
            conv_id: {
                "annotations": [
                    {"annotation_type": "scaffolding", "situation": ""},
                ],
            },
        },
    }

    situate_module.run_situation_label(
        version="vtest", model="claude-x", mode="sync",
        phase_cfg={"poll_interval": 1},
        annotations_data=annotations_data,
        **run_kwargs,
    )
    return saved["filename"]


def test_gold_output_filename_distinguished_from_non_gold(monkeypatch):
    gold_filename = _run_and_capture_filename(monkeypatch, gold=True, profile="anthropic")
    non_gold_filename = _run_and_capture_filename(monkeypatch, gold=False, profile="anthropic")

    assert gold_filename == "situation_labels_gold_anthropic_scaffolding.json"
    assert non_gold_filename == "situation_labels_anthropic_scaffolding.json"
    assert gold_filename != non_gold_filename
