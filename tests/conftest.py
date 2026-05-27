import json
import pytest


@pytest.fixture
def temp_data(tmp_path):
    """Create a temp data layout matching config paths."""
    t_dir = tmp_path / "data" / "transcripts"
    t_dir.mkdir(parents=True)
    conv = {"conversation_id": "2024-t1_2024-s1_099bf759-abcd", "turns": [
        {"turn_number": 1, "role": "TUTOR", "text": "Hi", "timestamp": "00:00-00:03", "type": "DIALOGUE"},
        {"turn_number": 2, "role": "STUDENT", "text": "hey", "timestamp": "00:03-00:05", "type": "DIALOGUE"},
        {"turn_number": 3, "role": "TUTOR", "text": "look here", "timestamp": "00:10-00:12", "type": "DIALOGUE"},
    ]}
    (t_dir / "2024-t1_2024-s1_099bf759-abcd.json").write_text(
        json.dumps(conv), encoding="utf-8"
    )
    # Keep the older conv_001 fixture for backward-compat with existing tests
    conv_simple = {"conversation_id": "conv_001", "turns": [
        {"turn_number": 1, "role": "TUTOR", "text": "Hi", "timestamp": "", "type": "DIALOGUE"},
    ]}
    (t_dir / "conv_001.json").write_text(json.dumps(conv_simple), encoding="utf-8")

    gt_dir = tmp_path / "data" / "ground_truth"
    gt_dir.mkdir(parents=True)
    gt = {"conversation_id": "conv_001", "num_turns": 1, "key_moments": []}
    (gt_dir / "conv_001.json").write_text(json.dumps(gt), encoding="utf-8")

    # Screenshot layout keyed by UUID (matches S3 convention)
    ss_dir = tmp_path / "deidentified" / "screenshots" / "099bf759-abcd"
    ss_dir.mkdir(parents=True)
    (ss_dir / "4.000.jpg").write_bytes(b"fake-jpg-1")
    (ss_dir / "11.500.jpg").write_bytes(b"fake-jpg-2")
    (ss_dir / "_metadata.json").write_text(json.dumps({
        "transcript_id": "099bf759-abcd",
        "images": {
            "4.000.jpg":  {"verified": True, "flagged": False, "eedi_ip": False},
            "11.500.jpg": {"verified": True, "flagged": False, "eedi_ip": True,
                           "eedi_ip_evidence": "Eedi branding visible"},
        },
    }), encoding="utf-8")

    (tmp_path / "results" / "annotator" / "v1").mkdir(parents=True)
    (tmp_path / "results" / "benchmark" / "v1").mkdir(parents=True)

    return tmp_path


@pytest.fixture
def local_storage(temp_data, monkeypatch):
    """Configure storage for local backend against temp dir."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_ROOT", str(temp_data))
    monkeypatch.setenv("STORAGE_GROUND_TRUTH", "data/ground_truth")
    monkeypatch.setenv("STORAGE_SCREENSHOTS", "deidentified/screenshots")
    import annotator.core.config as cfg_mod
    cfg_mod._loaded_config = None
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None
    yield temp_data
    st._backend = None
    st._cache.clear()
