"""Tests for the --only-overscaffold path in annotator/core/decompose.py.

Re-running decomposition just to add the over-scaffold pass must NOT re-run (or
clobber) the existing action/result facets. These tests pin down:
  - which input file the run reads (the already-decomposed file, not raw
    annotations -- reading raw would drop action_decomposed/result_decomposed
    from the output), via _input_prefix and _result_filename, and
  - the end-to-end behavior: action_decomposed/result_decomposed are preserved
    untouched while overscaffold_decomposed is populated, and only the
    over-scaffold pass makes API calls.
"""

import annotator.core.decompose as dc
from annotator.core.decompose import _input_prefix, _result_filename, run_decompose


# --- which file gets read/written --------------------------------------------

def test_input_prefix_only_overscaffold_reads_decomposed_file():
    # The whole point: in only-overscaffold mode we read the decomposed output
    # (which carries action/result facets), never the raw annotations file.
    assert _input_prefix(only_overscaffold=True, gold=True) == "decomposed_gold"
    assert _input_prefix(only_overscaffold=True, gold=False) == "decomposed"


def test_input_prefix_normal_run_reads_raw_annotations():
    assert _input_prefix(only_overscaffold=False, gold=True) == "annotations_gold"
    assert _input_prefix(only_overscaffold=False, gold=False) == "annotations"


def test_result_filename_assembles_suffixes():
    assert _result_filename("decomposed_gold", "scaffolding",
                            profile="anthropic", style=None, split="train") \
        == "decomposed_gold_anthropic_scaffolding.json"
    # non-train split gets a suffix; train does not
    assert _result_filename("decomposed", "rapport",
                            profile=None, style=None, split="test") \
        == "decomposed_test_rapport.json"
    assert _result_filename("annotations", "scaffolding",
                            profile=None, style="socratic", split="train") \
        == "annotations_socratic_scaffolding.json"


# --- guard -------------------------------------------------------------------

def test_only_overscaffold_rejects_non_scaffolding_target():
    # Over-scaffolding is scaffolding-specific; the flag is meaningless for other
    # targets and must fail fast before touching any files.
    result = run_decompose(version="v13", model="m", mode="sync", phase_cfg={},
                           only_overscaffold=True, target="rapport")
    assert result is None


# --- end-to-end: preserve action/result, add overscaffold --------------------

def test_only_overscaffold_preserves_action_result_and_adds_overscaffold(monkeypatch):
    conv_id = "abc"
    decomposed_input = {
        "version": "v13",
        "decomposed": True,
        "results": {
            conv_id: {
                "annotations": [
                    {
                        "annotation_type": "scaffolding",
                        "situation": "Student is stuck.",
                        "action": "Tutor gives away the full answer.",
                        "result": "Student copies it.",
                        "action_decomposed": ["a1", "a2"],
                        "result_decomposed": ["r1"],
                    }
                ]
            }
        },
        "decompose_stats": {
            "action_entries": 1, "result_entries": 1, "overscaffold_entries": 0,
            "skipped_action": 0, "skipped_result": 0, "skipped_overscaffold": 0,
            "total_action_facets": 2, "total_result_facets": 1,
            "total_overscaffold_facets": 0,
        },
        "token_summary": {"total_input_tokens": 10, "total_output_tokens": 20,
                          "total_tokens": 30, "errors": 0},
    }

    saved = {}

    monkeypatch.setattr(dc, "load_annotator_result", lambda v, f: decomposed_input)
    monkeypatch.setattr(dc, "load_split_ids", lambda split: {conv_id})
    monkeypatch.setattr(dc, "ModelClient", lambda model: object())
    monkeypatch.setattr(dc, "write_jsonl", lambda entries, path: None)
    monkeypatch.setattr(dc, "get_annotator_result_path", lambda v: __import__("pathlib").Path("/tmp"))

    def fake_run_sync(client, entries, json_mode=True):
        # Exactly one over-scaffold entry should be submitted; nothing else.
        assert len(entries) == 1
        key = entries[0]["key"]
        assert key.startswith("overscaffold__")
        return {key: {"text": '["over-scaffolded: gave away the answer"]',
                      "usage": {"input_tokens": 3, "output_tokens": 4}}}

    monkeypatch.setattr(dc, "run_sync_entries", fake_run_sync)

    def fake_save(version, filename, output):
        saved["filename"] = filename
        saved["output"] = output

    monkeypatch.setattr(dc, "save_annotator_result", fake_save)

    run_decompose(version="v13", model="m", mode="sync", phase_cfg={},
                  gold=True, profile="anthropic", target="scaffolding",
                  split="train", only_overscaffold=True)

    ann = saved["output"]["results"][conv_id]["annotations"][0]
    # Preserved untouched:
    assert ann["action_decomposed"] == ["a1", "a2"]
    assert ann["result_decomposed"] == ["r1"]
    # Newly added:
    assert ann["overscaffold_decomposed"] == ["over-scaffolded: gave away the answer"]
    # Wrote back to the decomposed file (gold + profile):
    assert saved["filename"] == "decomposed_gold_anthropic_scaffolding.json"
    # Stats: action/result preserved, overscaffold refreshed.
    stats = saved["output"]["decompose_stats"]
    assert stats["action_entries"] == 1
    assert stats["total_action_facets"] == 2
    assert stats["overscaffold_entries"] == 1
    assert stats["total_overscaffold_facets"] == 1
