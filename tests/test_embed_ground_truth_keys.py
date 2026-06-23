"""Ground-truth embedding must NOT silently ignore missing structural keys.

A ground_truth_hybrid file missing `key_moments`, or a moment missing
`action_decomposed` / `result_decomposed`, signals upstream corruption. The
old code did `data.get("key_moments", [])` / `moment.get(field) or []`, which
silently produced zero facets and an embeddings file that looked successful.
These tests pin the fix: such files must raise, not be silently skipped.
"""

import json
import pytest

import annotator.core.embed as embed_module

VALID_MOMENT = {
    "turn_start": 1, "turn_end": 2, "annotation_type": "scaffolding",
    "action_decomposed": ["a1"], "result_decomposed": ["r1"],
}


def _write_gt(tmp_path, monkeypatch, conv):
    """Point embed.DATA_DIR at tmp_path and drop one ground-truth file in it."""
    monkeypatch.setattr(embed_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(embed_module, "_load_encoder", lambda: object())
    monkeypatch.setattr(embed_module, "_encode_all",
                        lambda encoder, facets: [[0.0] for _ in facets])
    gt_dir = tmp_path / "ground_truth_hybrid"
    gt_dir.mkdir(parents=True)
    (gt_dir / "c1.json").write_text(json.dumps(conv), encoding="utf-8")


def test_valid_ground_truth_embeds_without_error(tmp_path, monkeypatch):
    _write_gt(tmp_path, monkeypatch,
              {"conversation_id": "c1", "num_turns": 3, "key_moments": [dict(VALID_MOMENT)]})
    embed_module.run_embed_ground_truth(labeller="hybrid")
    out = json.loads((tmp_path / "embeddings_hybrid.json").read_text(encoding="utf-8"))
    assert out["c1"]["key_moments"][0]["action_embeddings"] == [[0.0]]


def test_empty_key_moments_list_is_allowed(tmp_path, monkeypatch):
    # An empty list is legitimate (conversation with no annotated moments).
    _write_gt(tmp_path, monkeypatch,
              {"conversation_id": "c1", "num_turns": 3, "key_moments": []})
    embed_module.run_embed_ground_truth(labeller="hybrid")  # must not raise


def test_missing_key_moments_raises(tmp_path, monkeypatch):
    _write_gt(tmp_path, monkeypatch, {"conversation_id": "c1", "num_turns": 3})
    with pytest.raises(ValueError, match="key_moments"):
        embed_module.run_embed_ground_truth(labeller="hybrid")


def test_moment_missing_action_decomposed_raises(tmp_path, monkeypatch):
    m = dict(VALID_MOMENT)
    del m["action_decomposed"]
    _write_gt(tmp_path, monkeypatch,
              {"conversation_id": "c1", "num_turns": 3, "key_moments": [m]})
    with pytest.raises(ValueError, match="action_decomposed"):
        embed_module.run_embed_ground_truth(labeller="hybrid")


def test_moment_missing_result_decomposed_raises(tmp_path, monkeypatch):
    m = dict(VALID_MOMENT)
    del m["result_decomposed"]
    _write_gt(tmp_path, monkeypatch,
              {"conversation_id": "c1", "num_turns": 3, "key_moments": [m]})
    with pytest.raises(ValueError, match="result_decomposed"):
        embed_module.run_embed_ground_truth(labeller="hybrid")
