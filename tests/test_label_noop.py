"""run_label must NOT report success when it writes nothing.

If the requested --target matches no annotations (e.g. wrong target name), the
old code saved no file yet still returned a truthy data dict, so the CLI exited
0 and a chained runner kept going. The fix: a disk run that saves nothing
returns None so callers can treat it as the failure it is.
"""

from pathlib import Path

import annotator.core.label as label_module


def _common_mocks(monkeypatch, saved):
    monkeypatch.setattr(label_module, "load_split_ids", lambda split: {"uuid"})
    monkeypatch.setattr(label_module, "load_labeller_templates", lambda spec: {None: "T"})
    monkeypatch.setattr(label_module, "ModelClient", lambda model: object())
    monkeypatch.setattr(label_module, "run_sync_entries", lambda *a, **k: {})
    monkeypatch.setattr(label_module, "write_jsonl", lambda *a, **k: None)
    monkeypatch.setattr(label_module, "get_annotator_result_path", lambda v: Path("/tmp"))
    monkeypatch.setattr(label_module, "save_annotator_result",
                        lambda v, f, d: saved.append(f))


def _run(targets):
    return label_module.run_label(
        version="vtest", model="m", mode="sync",
        phase_cfg={"poll_interval": 1}, targets=targets, split="train",
    )


def test_returns_none_when_no_target_matches(monkeypatch):
    rapport_only = {"results": {"t1_s1_uuid": {"annotations": [
        {"annotation_type": "rapport", "result": "x", "situation": "", "action": ""}]}}}
    monkeypatch.setattr(label_module, "load_annotator_result",
                        lambda version, fname: {k: dict(v) for k, v in rapport_only.items()})
    saved = []
    _common_mocks(monkeypatch, saved)
    out = _run(["scaffolding"])
    assert out is None
    assert saved == []


def test_returns_data_and_saves_when_target_matches(monkeypatch):
    scaffolding = {"results": {"t1_s1_uuid": {"annotations": [
        {"annotation_type": "scaffolding", "result": "good", "situation": "", "action": ""}]}}}
    monkeypatch.setattr(label_module, "load_annotator_result",
                        lambda version, fname: {k: dict(v) for k, v in scaffolding.items()})
    saved = []
    _common_mocks(monkeypatch, saved)
    out = _run(["scaffolding"])
    assert out is not None
    assert any("scaffolding" in f for f in saved)
