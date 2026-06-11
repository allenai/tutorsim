"""Tests for annotator.core.structure output filename naming.

structure_labels_{target}.json must encode --gold the same way decompose.py's
decomposed_gold_{target}.json does -- otherwise a gold run and a non-gold run
(same profile/style/split/target) write to the same output file and silently
overwrite each other.
"""

import annotator.core.structure as structure_module
from annotator.core.utils import load_split_ids


def _run_and_capture_filename(monkeypatch, **run_kwargs):
    saved = {}

    def fake_save(version, filename, data):
        saved["filename"] = filename

    monkeypatch.setattr(structure_module, "save_annotator_result", fake_save)
    monkeypatch.setattr(structure_module, "ModelClient", lambda model: object())
    monkeypatch.setattr(structure_module, "run_sync_entries", lambda *a, **kw: {})
    monkeypatch.setattr(structure_module, "_load_prompt", lambda path: "TEMPLATE")

    train_id = next(iter(load_split_ids("train")))
    conv_id = f"2024-t1_2024-s1_{train_id}"
    annotations_data = {
        "results": {
            conv_id: {
                "annotations": [
                    {"annotation_type": "scaffolding",
                     "action_decomposed": [], "result_decomposed": []},
                ],
            },
        },
    }

    structure_module.run_structure_label(
        version="vtest", model="claude-x", mode="sync",
        phase_cfg={"poll_interval": 1},
        annotations_data=annotations_data,
        **run_kwargs,
    )
    return saved["filename"]


def test_gold_output_filename_distinguished_from_non_gold(monkeypatch):
    gold_filename = _run_and_capture_filename(monkeypatch, gold=True, profile="anthropic")
    non_gold_filename = _run_and_capture_filename(monkeypatch, gold=False, profile="anthropic")

    assert gold_filename == "structure_labels_gold_anthropic_scaffolding.json"
    assert non_gold_filename == "structure_labels_anthropic_scaffolding.json"
    assert gold_filename != non_gold_filename
