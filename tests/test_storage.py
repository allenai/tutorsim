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

    def test_read_write_bytes_roundtrip(self, local_storage):
        from annotator.core.storage import _get_backend
        be = _get_backend()
        payload = b"\x89PNG\r\n\x1a\nfake image bytes"
        be.write_bytes("screenshots/test/hello.jpg", payload)
        assert be.read_bytes("screenshots/test/hello.jpg") == payload

    def test_read_bytes_missing_raises(self, local_storage):
        from annotator.core.storage import _get_backend
        be = _get_backend()
        with pytest.raises(FileNotFoundError):
            be.read_bytes("nope/nope.jpg")

    def test_local_presigned_url_is_file_uri(self, local_storage):
        from annotator.core.storage import _get_backend
        be = _get_backend()
        be.write_bytes("screenshots/x/y.jpg", b"abc")
        url = be.get_presigned_url("screenshots/x/y.jpg")
        assert url.startswith("file://")
        assert url.endswith("y.jpg")

    def test_consolidated_transcript_has_start_seconds(self, local_storage, temp_data):
        # Overwrite conv_001 with a timestamp string and reload
        import json
        t_path = temp_data / "data" / "transcripts" / "conv_001.json"
        conv = {
            "conversation_id": "conv_001",
            "turns": [
                {"turn_number": 1, "timestamp": "02:26-02:27", "role": "TUTOR", "text": "Hi", "type": "DIALOGUE"},
                {"turn_number": 2, "timestamp": "02:30-02:31", "role": "STUDENT", "text": "Hi back", "type": "DIALOGUE"},
            ],
        }
        t_path.write_text(json.dumps(conv), encoding="utf-8")

        import annotator.core.storage as st
        st._cache.clear()

        loaded = st.load_transcript("conv_001")
        assert loaded["turns"][0]["start_seconds"] == pytest.approx(146.0)  # 2*60+26
        assert loaded["turns"][1]["start_seconds"] == pytest.approx(150.0)  # 2*60+30
        # Existing timestamp string is preserved
        assert loaded["turns"][0]["timestamp"] == "02:26-02:27"

    def test_malformed_timestamp_yields_zero(self, local_storage, temp_data):
        import json
        t_path = temp_data / "data" / "transcripts" / "conv_001.json"
        conv = {
            "conversation_id": "conv_001",
            "turns": [
                {"turn_number": 1, "timestamp": "", "role": "TUTOR", "text": "Hi", "type": "DIALOGUE"},
                {"turn_number": 2, "timestamp": "junk", "role": "STUDENT", "text": "Hi", "type": "DIALOGUE"},
            ],
        }
        t_path.write_text(json.dumps(conv), encoding="utf-8")

        import annotator.core.storage as st
        st._cache.clear()

        loaded = st.load_transcript("conv_001")
        assert loaded["turns"][0]["start_seconds"] == 0.0
        assert loaded["turns"][1]["start_seconds"] == 0.0

    def test_conv_id_to_uuid_bench_composite(self):
        """Bench-schema composite is {tutor_uuid}_{student_uuid}_{transcript_uuid}.

        Screenshots and ground truth are keyed by transcript_id (the LAST UUID),
        so the helper must return that, not the first UUID (tutor).
        """
        from annotator.core.storage import _conv_id_to_uuid
        tutor = "007af6e2-a810-56ff-82e2-56b93c2f9e32"
        student = "3dd84243-bf23-52a8-b189-8b5cf6340dc7"
        transcript = "74a3d8fd-a544-5447-9a61-b8929235372a"
        composite = f"{tutor}_{student}_{transcript}"
        assert _conv_id_to_uuid(composite) == transcript

    def test_conv_id_to_uuid_legacy_composite(self):
        """Legacy composite {yyyy-tNN}_{yyyy-sNN}_{transcript_uuid} still works."""
        from annotator.core.storage import _conv_id_to_uuid
        legacy = "2024-t1_2024-s1_099bf759-2426-549b-8dff-ad3f4be80db2"
        assert _conv_id_to_uuid(legacy) == "099bf759-2426-549b-8dff-ad3f4be80db2"

    def test_conv_id_to_uuid_bare(self):
        from annotator.core.storage import _conv_id_to_uuid
        bare = "099bf759-2426-549b-8dff-ad3f4be80db2"
        assert _conv_id_to_uuid(bare) == bare

    def test_conv_id_to_uuid_benchmark_scenario_id(self):
        """Benchmark scenario IDs append `__{type}_{idx}` to the composite.
        The helper must still return the transcript_id (the last UUID match),
        ignoring the trailing scenario-index suffix."""
        from annotator.core.storage import _conv_id_to_uuid
        tutor = "04f1df12-f52c-56d5-8577-48fd770c6809"
        student = "e14d9ebe-c5d9-5582-90e0-30385628f56e"
        transcript = "b2a884a4-ac23-524f-a2ed-2f97a1b9ce85"
        scenario_id = f"{tutor}_{student}_{transcript}__rapport_0"
        assert _conv_id_to_uuid(scenario_id) == transcript

    def test_load_ground_truth_file_by_composite_conv_id(self, temp_data, monkeypatch):
        """GT files keyed by transcript_id should resolve when looked up via the
        composite conv_id produced for bench-schema transcripts."""
        gt_dir = temp_data / "data" / "ground_truth_hybrid"
        gt_dir.mkdir(parents=True)
        transcript = "74a3d8fd-a544-5447-9a61-b8929235372a"
        gt = {"conversation_id": transcript, "num_turns": 1, "key_moments": []}
        (gt_dir / f"{transcript}.json").write_text(json.dumps(gt), encoding="utf-8")

        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("STORAGE_ROOT", str(temp_data))
        monkeypatch.setenv("STORAGE_GROUND_TRUTH", "data/ground_truth_hybrid")
        import annotator.core.config as cfg_mod
        cfg_mod._loaded_config = None
        import annotator.core.storage as st
        st._cache.clear()
        st._backend = None

        composite = f"007af6e2-a810-56ff-82e2-56b93c2f9e32_3dd84243-bf23-52a8-b189-8b5cf6340dc7_{transcript}"
        from annotator.core.storage import load_ground_truth_file
        loaded = load_ground_truth_file(composite)
        assert loaded is not None
        assert loaded["conversation_id"] == transcript

        st._backend = None
        st._cache.clear()

    def test_list_screenshots(self, local_storage):
        from annotator.core.storage import list_screenshots
        files = list_screenshots("2024-t1_2024-s1_099bf759-abcd")
        assert sorted(files) == ["11.500.jpg", "4.000.jpg"]

    def test_list_screenshots_missing_conv_returns_empty(self, local_storage):
        from annotator.core.storage import list_screenshots
        assert list_screenshots("nonexistent_conv") == []

    def test_load_screenshot_bytes(self, local_storage):
        from annotator.core.storage import load_screenshot_bytes
        data = load_screenshot_bytes("2024-t1_2024-s1_099bf759-abcd", "4.000.jpg")
        assert data == b"fake-jpg-1"

    def test_load_screenshot_verification(self, local_storage):
        from annotator.core.storage import load_screenshot_verification
        meta = load_screenshot_verification("2024-t1_2024-s1_099bf759-abcd")
        assert meta["images"]["11.500.jpg"]["eedi_ip"] is True
        assert meta["images"]["4.000.jpg"]["flagged"] is False

    def test_load_screenshot_verification_missing_returns_empty(self, local_storage):
        from annotator.core.storage import load_screenshot_verification
        assert load_screenshot_verification("no_such_conv") == {}

    def test_get_screenshot_uri_local(self, local_storage):
        from annotator.core.storage import get_screenshot_uri
        uri = get_screenshot_uri("2024-t1_2024-s1_099bf759-abcd", "4.000.jpg")
        assert uri.startswith("file://")
        assert uri.endswith("4.000.jpg")

    def test_save_and_load_annotator_shard(self, local_storage):
        from annotator.core.storage import (
            save_annotator_shard, load_annotator_shards, list_annotator_shard_ids,
        )
        save_annotator_shard("v1", "detections", "conv_a", {"detections": [{"x": 1}]})
        save_annotator_shard("v1", "detections", "conv_b", {"detections": [{"x": 2}]})

        ids = list_annotator_shard_ids("v1", "detections")
        assert sorted(ids) == ["conv_a", "conv_b"]

        shards = load_annotator_shards("v1", "detections")
        assert shards["conv_a"]["detections"][0]["x"] == 1
        assert shards["conv_b"]["detections"][0]["x"] == 2

    def test_shards_isolated_by_basename(self, local_storage):
        from annotator.core.storage import (
            save_annotator_shard, list_annotator_shard_ids,
        )
        save_annotator_shard("v1", "detections", "conv_a", {"k": "det"})
        save_annotator_shard("v1", "annotations_generous", "conv_a", {"k": "ann"})

        det_ids = list_annotator_shard_ids("v1", "detections")
        ann_ids = list_annotator_shard_ids("v1", "annotations_generous")
        assert det_ids == ["conv_a"]
        assert ann_ids == ["conv_a"]

    def test_list_shard_ids_empty_when_dir_missing(self, local_storage):
        from annotator.core.storage import list_annotator_shard_ids
        assert list_annotator_shard_ids("v_never_run", "detections") == []

    def test_top_level_results_listing_excludes_shards(self, local_storage):
        """Existing list_annotator_result_files must not pick up shard files."""
        from annotator.core.storage import (
            save_annotator_result, save_annotator_shard, list_annotator_result_files,
        )
        save_annotator_result("v1", "detections.json", {"ok": True})
        save_annotator_shard("v1", "detections", "conv_a", {"x": 1})

        files = list_annotator_result_files("v1")
        assert "detections.json" in files
        assert "conv_a.json" not in files

    def test_inflight_batch_lifecycle(self, local_storage):
        from annotator.core.storage import (
            save_inflight_batch, load_inflight_batch, clear_inflight_batch,
        )
        assert load_inflight_batch("v1", "detections") is None

        save_inflight_batch("v1", "detections", {
            "provider": "anthropic", "batch_id": "msgbatch_abc",
            "model": "claude-opus-4-6", "n_entries": 6,
            "display_name": "detect", "submitted_at": "2026-04-27T15:00:00",
        })
        rec = load_inflight_batch("v1", "detections")
        assert rec["batch_id"] == "msgbatch_abc"
        assert rec["n_entries"] == 6

        clear_inflight_batch("v1", "detections")
        assert load_inflight_batch("v1", "detections") is None

    def test_clear_inflight_is_safe_when_absent(self, local_storage):
        from annotator.core.storage import clear_inflight_batch
        clear_inflight_batch("v_never_run", "detections")  # should not raise

    def test_inflight_isolated_from_results_listing(self, local_storage):
        from annotator.core.storage import (
            save_annotator_result, save_inflight_batch, list_annotator_result_files,
        )
        save_annotator_result("v1", "detections.json", {})
        save_inflight_batch("v1", "detections", {"batch_id": "x"})
        files = list_annotator_result_files("v1")
        assert files == ["detections.json"]


class TestBenchmarkInflightBatch:
    def test_roundtrip(self, local_storage):
        from annotator.core.storage import (
            save_benchmark_inflight_batch, load_benchmark_inflight_batch,
        )
        save_benchmark_inflight_batch("v_test", "anthropic", "balanced", {
            "provider": "anthropic", "model": "claude-opus-4-6",
            "batch_id": "msgbatch_abc", "n_entries": 12,
            "entry_keys_hash": "abc123def456", "display_name": "annotate",
            "submitted_at": "2026-04-28T10:00:00",
        })
        loaded = load_benchmark_inflight_batch("v_test", "anthropic", "balanced")
        assert loaded["batch_id"] == "msgbatch_abc"
        assert loaded["n_entries"] == 12

    def test_load_missing_returns_none(self, local_storage):
        from annotator.core.storage import load_benchmark_inflight_batch
        assert load_benchmark_inflight_batch("v_nope", "anthropic", "generous") is None

    def test_clear_removes_sidecar(self, local_storage):
        from annotator.core.storage import (
            save_benchmark_inflight_batch, load_benchmark_inflight_batch,
            clear_benchmark_inflight_batch,
        )
        save_benchmark_inflight_batch("v_test", "anthropic", "balanced", {"batch_id": "x"})
        clear_benchmark_inflight_batch("v_test", "anthropic", "balanced")
        assert load_benchmark_inflight_batch("v_test", "anthropic", "balanced") is None


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

    def test_s3_read_write_bytes(self, s3_env):
        import boto3
        from moto import mock_aws
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            import annotator.core.storage as st
            st._backend = None
            be = st._get_backend()

            payload = b"\x89PNG\r\n\x1a\nbytes"
            be.write_bytes("screenshots/a/b.jpg", payload)
            assert be.read_bytes("screenshots/a/b.jpg") == payload

    def test_s3_presigned_url_is_https(self, s3_env):
        import boto3
        from moto import mock_aws
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            import annotator.core.storage as st
            st._backend = None
            be = st._get_backend()
            be.write_bytes("screenshots/a/b.jpg", b"x")
            url = be.get_presigned_url("screenshots/a/b.jpg", expires_seconds=3600)
            assert url.startswith("https://")
            assert "test-bucket" in url
            assert "b.jpg" in url
