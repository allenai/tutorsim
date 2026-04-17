import json
import pytest


@pytest.fixture
def temp_data(tmp_path):
    """Create a temp data layout matching config paths."""
    t_dir = tmp_path / "data" / "transcripts"
    t_dir.mkdir(parents=True)
    conv = {"conversation_id": "conv_001", "turns": [
        {"turn_number": 1, "role": "TUTOR", "text": "Hi"}
    ]}
    (t_dir / "conv_001.json").write_text(json.dumps(conv), encoding="utf-8")

    gt_dir = tmp_path / "data" / "ground_truth"
    gt_dir.mkdir(parents=True)
    gt = {"conversation_id": "conv_001", "num_turns": 1, "key_moments": []}
    (gt_dir / "conv_001.json").write_text(json.dumps(gt), encoding="utf-8")

    (tmp_path / "results" / "annotator" / "v1").mkdir(parents=True)
    (tmp_path / "results" / "benchmark" / "v1").mkdir(parents=True)

    return tmp_path


@pytest.fixture
def local_storage(temp_data, monkeypatch):
    """Configure storage for local backend against temp dir."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(temp_data))
    monkeypatch.setenv("STORAGE_GROUND_TRUTH", "data/ground_truth")
    import annotator.core.config as cfg_mod
    cfg_mod._loaded_config = None
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None
    yield temp_data
    st._backend = None
    st._cache.clear()
