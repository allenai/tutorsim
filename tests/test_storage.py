"""Storage layer tests -- verifies Factor IV compliance."""
import json
import pytest


class TestLocalBackend:
    def test_load_all_transcripts(self, local_storage):
        from annotator.core.storage import load_all_transcripts
        result = load_all_transcripts()
        assert "conv_001" in result
        assert result["conv_001"]["conversation_id"] == "conv_001"

    def test_load_transcript(self, local_storage):
        from annotator.core.storage import load_transcript
        assert load_transcript("conv_001") is not None
        assert load_transcript("nonexistent") is None

    def test_load_all_ground_truth(self, local_storage):
        from annotator.core.storage import load_all_ground_truth_files
        result = load_all_ground_truth_files()
        assert len(result) == 1
        assert result[0]["conversation_id"] == "conv_001"

    def test_save_and_load_annotator_result(self, local_storage):
        from annotator.core.storage import save_annotator_result, load_annotator_result
        save_annotator_result("v1", "test.json", {"ok": True})
        assert load_annotator_result("v1", "test.json")["ok"] is True

    def test_annotator_result_exists(self, local_storage):
        from annotator.core.storage import save_annotator_result, annotator_result_exists
        save_annotator_result("v1", "exists.json", {"ok": True})
        assert annotator_result_exists("v1", "exists.json")
        assert not annotator_result_exists("v1", "nope.json")

    def test_list_annotator_result_files(self, local_storage):
        from annotator.core.storage import save_annotator_result, list_annotator_result_files
        save_annotator_result("v1", "detections.json", {})
        save_annotator_result("v1", "annotations.json", {})
        files = list_annotator_result_files("v1")
        assert "detections.json" in files
        assert "annotations.json" in files

    def test_save_and_load_benchmark_result(self, local_storage):
        from annotator.core.storage import save_benchmark_result, load_benchmark_result
        save_benchmark_result("v1", "exchanges", "anthropic", "s1.json", data={"id": "s1"})
        loaded = load_benchmark_result("v1", "exchanges", "anthropic", "s1.json")
        assert loaded["id"] == "s1"

    def test_list_benchmark_result_files(self, local_storage):
        from annotator.core.storage import save_benchmark_result, list_benchmark_result_files
        save_benchmark_result("v1", "exchanges", "anthropic", "s1.json", data={})
        save_benchmark_result("v1", "exchanges", "anthropic", "s2.json", data={})
        files = list_benchmark_result_files("v1", "exchanges", "anthropic")
        assert "s1.json" in files
        assert "s2.json" in files

    def test_get_annotator_result_path(self, local_storage):
        from annotator.core.storage import get_annotator_result_path
        path = get_annotator_result_path("v1")
        assert path.exists()
        assert path.is_dir()

    def test_get_benchmark_result_path(self, local_storage):
        from annotator.core.storage import get_benchmark_result_path
        path = get_benchmark_result_path("v1")
        assert path.exists()

    def test_env_var_path_override(self, temp_data, monkeypatch):
        """Factor IV: paths overridable via env vars."""
        custom_dir = temp_data / "custom_transcripts"
        custom_dir.mkdir()
        conv = {"conversation_id": "custom_001", "turns": []}
        (custom_dir / "custom_001.json").write_text(json.dumps(conv), encoding="utf-8")

        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("STORAGE_ROOT", str(temp_data))
        monkeypatch.setenv("STORAGE_TRANSCRIPTS", "custom_transcripts")
        import annotator.core.config as cfg_mod
        cfg_mod._loaded_config = None
        import annotator.core.storage as st
        st._cache.clear()
        st._backend = None

        from annotator.core.storage import load_all_transcripts
        result = load_all_transcripts()
        assert "custom_001" in result

        st._backend = None
        st._cache.clear()


class TestS3Backend:
    @pytest.fixture
    def s3_env(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("S3_PREFIX", "test-prefix")
        import annotator.core.config as cfg_mod
        cfg_mod._loaded_config = None
        import annotator.core.storage as st
        st._cache.clear()
        st._backend = None
        yield
        st._backend = None
        st._cache.clear()

    def test_s3_save_and_load(self, s3_env):
        import boto3
        from moto import mock_aws
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")
            s3.put_object(Bucket="test-bucket",
                          Key="test-prefix/data/transcripts/conv_s3.json",
                          Body=json.dumps({"conversation_id": "conv_s3", "turns": []}))

            import annotator.core.storage as st
            st._backend = None

            from annotator.core.storage import load_transcript, save_annotator_result, load_annotator_result
            assert load_transcript("conv_s3")["conversation_id"] == "conv_s3"

            save_annotator_result("v1", "test.json", {"ok": True})
            assert load_annotator_result("v1", "test.json")["ok"] is True

    def test_s3_get_local_path_raises(self, s3_env):
        from moto import mock_aws
        import boto3
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
            import annotator.core.storage as st
            st._backend = None

            from annotator.core.storage import get_annotator_result_path
            with pytest.raises(RuntimeError, match="S3 mode"):
                get_annotator_result_path("v1")
