from __future__ import annotations

from pathlib import Path

import pytest

from tutor_bench.toolkit.io_utils import load_jsonl, read_stems_file, resolve_path, save_jsonl
from tutor_bench.toolkit.llm_utils import compute_cost_usd, extract_json_array, extract_json_object


def test_read_stems_file_filters_comments_and_max(tmp_path: Path) -> None:
    p = tmp_path / "stems.txt"
    p.write_text("# comment\n\nstem_a\nstem_b\n", encoding="utf-8")
    assert read_stems_file(p) == ["stem_a", "stem_b"]
    assert read_stems_file(p, max_stems=1) == ["stem_a"]


def test_save_and_load_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    rows = [{"a": 1}, {"b": "x"}]
    save_jsonl(p, rows)
    loaded = load_jsonl(p)
    assert loaded == rows


def test_storage_root_resolves_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    resolved = resolve_path("sub/file.jsonl")
    assert str(resolved) == str(tmp_path / "sub" / "file.jsonl")


def test_storage_root_passthrough_for_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "other_root"))
    abs_path = tmp_path / "explicit.jsonl"
    resolved = resolve_path(abs_path)
    assert str(resolved) == str(abs_path)


def test_storage_root_unset_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STORAGE_ROOT", raising=False)
    resolved = resolve_path(tmp_path / "x.jsonl")
    assert str(resolved) == str(tmp_path / "x.jsonl")


def test_save_and_load_jsonl_with_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    rows = [{"a": 1}, {"b": "x"}]
    save_jsonl("nested/rows.jsonl", rows)
    assert (tmp_path / "nested" / "rows.jsonl").exists()
    assert load_jsonl("nested/rows.jsonl") == rows


def test_extract_json_array_and_object_with_wrapped_text() -> None:
    arr_text = 'prefix\n[{"x":1},{"y":2}]\nsuffix'
    obj_text = 'garbage {"dense_caption":"abc"} trailing'
    assert extract_json_array(arr_text) == [{"x": 1}, {"y": 2}]
    assert extract_json_object(obj_text) == {"dense_caption": "abc"}


def test_compute_cost_usd_with_pricing() -> None:
    usage = {"prompt_tokens": 1_000.0, "completion_tokens": 2_000.0}
    pricing = {
        "openai:gpt-5.2": {
            "input_per_1m": 2.0,
            "output_per_1m": 10.0,
        }
    }
    out = compute_cost_usd(usage, "openai", "gpt-5.2", pricing)
    assert out["pricing_key"] == "openai:gpt-5.2"
    assert out["has_pricing"] is True
    # 1000/1e6*2 + 2000/1e6*10 = 0.022
    assert abs(out["total_cost_usd"] - 0.022) < 1e-9
