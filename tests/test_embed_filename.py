"""Tests for annotator.core.embed (decompose-output mode) filename naming.

embedded_{...}_{target}.json must encode --gold the same way decompose.py /
structure.py do -- otherwise embedding a gold decomposition and a non-gold one
(same profile/style/split/target) write to the same output file and silently
overwrite each other.
"""

import annotator.core.embed as embed_module


def _run_and_capture_filename(monkeypatch, **run_kwargs):
    saved = {}

    def fake_save(version, filename, data):
        saved["filename"] = filename

    # Minimal decomposed data with no facets -> encoder is never actually used.
    monkeypatch.setattr(embed_module, "load_annotator_result",
                        lambda version, fname: {"results": {}})
    monkeypatch.setattr(embed_module, "save_annotator_result", fake_save)
    monkeypatch.setattr(embed_module, "_load_encoder", lambda: object())

    embed_module.run_embed_decomposed(version="vtest", **run_kwargs)
    return saved["filename"]


def test_gold_output_filename_distinguished_from_non_gold(monkeypatch):
    gold_filename = _run_and_capture_filename(monkeypatch, gold=True, profile="anthropic")
    non_gold_filename = _run_and_capture_filename(monkeypatch, gold=False, profile="anthropic")

    assert gold_filename == "embedded_gold_anthropic_scaffolding.json"
    assert non_gold_filename == "embedded_anthropic_scaffolding.json"
    assert gold_filename != non_gold_filename
