# Factor IV Storage Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor storage.py to comply with 12-factor app Factor IV (backing services as attached resources) — backend protocol pattern eliminates all `if backend == "s3"` branching, env vars control everything needed to swap backends without code or config file changes.

**Architecture:** Introduce a `StorageBackend` ABC that both `LocalBackend` and `S3Backend` implement. All public functions delegate to a singleton backend instance — zero branching. Paths come from config.yaml with env var overrides (`STORAGE_TRANSCRIPTS`, `STORAGE_GROUND_TRUTH`, etc.) so that swapping from local to S3 is purely env var changes. The S3 bucket has different directory names than local (`de-identified/transfer_1` vs `data/transcripts`), so path overrides are essential for a true backend swap.

**Tech Stack:** Python 3.10+, boto3, pytest (new), moto (S3 mocking for tests)

---

## File Structure

```
annotator/core/storage.py          — REWRITE: backend ABC + implementations + public API
config.yaml                        — MODIFY: simplify to single paths section + env var docs
.env.example                       — CREATE: document all env vars needed for backend swap
tests/                             — CREATE: new test directory
tests/__init__.py                  — CREATE: empty
tests/conftest.py                  — CREATE: shared fixtures
tests/test_storage.py              — CREATE: local + S3 tests
requirements.txt                   — MODIFY: add pytest, moto[s3]
```

Callers (detect.py, annotate.py, label.py, eval.py, view.py, benchmark/*.py) are **unchanged** — the public API stays identical.

---

## Key Design Decision: Path Env Var Overrides

The S3 bucket and local filesystem have different directory layouts:
- Local: `data/transcripts`, `data/ground_truth`
- S3: `de-identified/transfer_1`, `de-identified/transfer_2`, `annotations`

A true Factor IV swap means changing ONLY env vars, not editing config.yaml. So paths must be overridable via env vars:

| Env Var | Purpose | Example (local) | Example (S3) |
|---------|---------|-----------------|--------------|
| `STORAGE_BACKEND` | Backend type | `local` | `s3` |
| `STORAGE_ROOT` | Local filesystem root | `/path/to/repo` | (ignored) |
| `S3_BUCKET` | S3 bucket name | (ignored) | `kylel-alexisr-edu` |
| `S3_PREFIX` | S3 key prefix | (ignored) | `synth-students/tutor-bench` |
| `STORAGE_TRANSCRIPTS` | Comma-separated transcript paths | (use config default) | `de-identified/transfer_1,de-identified/transfer_2` |
| `STORAGE_GROUND_TRUTH` | Comma-separated GT paths | (use config default) | `annotations` |
| `STORAGE_ANNOTATOR_RESULTS` | Annotator results path | (use config default) | `results/annotator` |
| `STORAGE_BENCHMARK_RESULTS` | Benchmark results path | (use config default) | `results/benchmark` |

Config.yaml provides sensible defaults for local dev. Env vars override for S3 deployment.

---

### Task 1: Rewrite storage.py with backend protocol

**Files:**
- Modify: `annotator/core/storage.py`

- [ ] **Step 1: Rewrite storage.py**

Replace the entire file with the backend ABC pattern. Key changes from current:
- `StorageBackend` ABC with `read_json`, `write_json`, `list_files`, `exists`, `get_local_path`
- `LocalBackend(root: Path)` and `S3Backend(bucket, prefix)` implementations
- Singleton `_backend` initialized once from env/config
- Path resolution reads env vars first, then config.yaml `paths:` section
- All public functions delegate to `_get_backend()` — zero branching
- S3 error handling uses `botocore.exceptions.ClientError` with error code check

```python
"""
S3/local storage layer for pipeline data (Factor IV compliant).

One set of paths in config, overridable by env vars per deploy.
Backend (local/s3) determines how paths resolve.
Swapping backends is env var changes only — no code or config file edits.

Env vars (override config.yaml):
    STORAGE_BACKEND           — 'local' or 's3' (default: local)
    STORAGE_ROOT              — local filesystem root (default: repo root)
    S3_BUCKET                 — S3 bucket name
    S3_PREFIX                 — S3 key prefix
    STORAGE_TRANSCRIPTS       — comma-separated transcript paths (override config)
    STORAGE_GROUND_TRUTH      — comma-separated ground truth paths (override config)
    STORAGE_ANNOTATOR_RESULTS — annotator results path (override config)
    STORAGE_BENCHMARK_RESULTS — benchmark results path (override config)
"""

import json
import os
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from pathlib import Path

from .config import load_config

REPO_ROOT = Path(__file__).parent.parent.parent

_cache: dict[str, object] = {}


# ===================================================================
# Config resolution (env vars take precedence over config.yaml)
# ===================================================================

def _get_storage_config() -> dict:
    config = load_config()
    return config.get("storage", {})


def get_backend() -> str:
    """Return 's3' or 'local'. Env var STORAGE_BACKEND overrides config."""
    env = os.environ.get("STORAGE_BACKEND", "").lower()
    if env in ("s3", "local"):
        return env
    return _get_storage_config().get("backend", "local")


def _get_root() -> Path:
    """Local filesystem root. Env var STORAGE_ROOT overrides config."""
    env = os.environ.get("STORAGE_ROOT", "")
    if env:
        return Path(env)
    cfg_root = _get_storage_config().get("root", "")
    if cfg_root:
        return REPO_ROOT / cfg_root
    return REPO_ROOT


def _get_bucket() -> str:
    return os.environ.get("S3_BUCKET", "") or _get_storage_config().get("bucket", "")


def _get_prefix() -> str:
    return os.environ.get("S3_PREFIX", "") or _get_storage_config().get("prefix", "")


def _get_path_list(category: str) -> list[str]:
    """Get paths for a category. Env var overrides config.yaml.

    Env var names: STORAGE_TRANSCRIPTS, STORAGE_GROUND_TRUTH, etc.
    Env var value is comma-separated list of paths.
    """
    env_key = f"STORAGE_{category.upper()}"
    env_val = os.environ.get(env_key, "")
    if env_val:
        return [p.strip() for p in env_val.split(",") if p.strip()]
    paths = _get_storage_config().get("paths", {})
    raw = paths.get(category, [])
    if isinstance(raw, str):
        raw = [raw]
    return raw


def _get_result_path(category: str) -> str:
    """Get single result path for a category. Env var overrides config."""
    env_key = f"STORAGE_{category.upper()}"
    env_val = os.environ.get(env_key, "")
    if env_val:
        return env_val.strip()
    paths = _get_storage_config().get("paths", {})
    return paths.get(category, "")


# ===================================================================
# Backend ABC + implementations
# ===================================================================

class StorageBackend(ABC):
    """Abstract storage backend. Both local and S3 implement this."""

    @abstractmethod
    def read_json(self, rel_path: str) -> dict | None: ...

    @abstractmethod
    def write_json(self, rel_path: str, data: dict) -> None: ...

    @abstractmethod
    def list_files(self, rel_prefix: str) -> list[str]: ...

    @abstractmethod
    def exists(self, rel_path: str) -> bool: ...

    @abstractmethod
    def get_local_path(self, rel_path: str) -> Path: ...


class LocalBackend(StorageBackend):
    def __init__(self, root: Path):
        self.root = root

    def read_json(self, rel_path: str) -> dict | None:
        path = self.root / rel_path
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, rel_path: str, data: dict) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def list_files(self, rel_prefix: str) -> list[str]:
        directory = self.root / rel_prefix
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.glob("*.json"))

    def exists(self, rel_path: str) -> bool:
        return (self.root / rel_path).exists()

    def get_local_path(self, rel_path: str) -> Path:
        path = self.root / rel_path
        # If it looks like a file (has extension), ensure parent exists
        # Otherwise ensure the directory itself exists
        if Path(rel_path).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        return path


class S3Backend(StorageBackend):
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("s3")
        return self._client

    def _key(self, rel_path: str) -> str:
        return f"{self.prefix}/{rel_path}" if self.prefix else rel_path

    def read_json(self, rel_path: str) -> dict | None:
        key = self._key(rel_path)
        cache_key = f"s3://{self.bucket}/{key}"
        if cache_key in _cache:
            return _cache[cache_key]
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            _cache[cache_key] = data
            return data
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            # Handle ClientError with 404 code
            if hasattr(e, "response"):
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "404"):
                    return None
            raise

    def write_json(self, rel_path: str, data: dict) -> None:
        key = self._key(rel_path)
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body.encode("utf-8"))
        _cache[f"s3://{self.bucket}/{key}"] = data

    def list_files(self, rel_prefix: str) -> list[str]:
        prefix = self._key(rel_prefix.rstrip("/") + "/")
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return [k.rsplit("/", 1)[-1] for k in keys if k.endswith(".json")]

    def exists(self, rel_path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(rel_path))
            return True
        except Exception:
            return False

    def get_local_path(self, rel_path: str) -> Path:
        raise RuntimeError(
            "get_local_path not available in S3 mode. "
            "Use read_json/write_json for data access."
        )


# ===================================================================
# Singleton backend instance
# ===================================================================

_backend: StorageBackend | None = None


def _get_backend() -> StorageBackend:
    global _backend
    if _backend is None:
        if get_backend() == "s3":
            _backend = S3Backend(_get_bucket(), _get_prefix())
        else:
            _backend = LocalBackend(_get_root())
    return _backend


# ===================================================================
# Public API — Transcripts
# ===================================================================

def load_transcript(conv_id: str) -> dict | None:
    be = _get_backend()
    for rel_dir in _get_path_list("transcripts"):
        data = be.read_json(f"{rel_dir}/{conv_id}.json")
        if data is not None:
            return data
    return None


def load_all_transcripts() -> dict[str, dict]:
    be = _get_backend()
    transcripts = {}
    for rel_dir in _get_path_list("transcripts"):
        for fname in be.list_files(rel_dir):
            data = be.read_json(f"{rel_dir}/{fname}")
            if data and "conversation_id" in data:
                transcripts[data["conversation_id"]] = data
            elif data:
                transcripts[fname.replace(".json", "")] = data
    return transcripts


def list_transcript_ids() -> list[str]:
    be = _get_backend()
    ids = set()
    for rel_dir in _get_path_list("transcripts"):
        for fname in be.list_files(rel_dir):
            ids.add(fname.replace(".json", ""))
    return sorted(ids)


# ===================================================================
# Public API — Ground Truth
# ===================================================================

def load_ground_truth_file(conv_id: str) -> dict | None:
    be = _get_backend()
    for rel_dir in _get_path_list("ground_truth"):
        data = be.read_json(f"{rel_dir}/{conv_id}.json")
        if data is not None:
            return data
    return None


def load_all_ground_truth_files() -> list[dict]:
    be = _get_backend()
    files = []
    for rel_dir in _get_path_list("ground_truth"):
        for fname in be.list_files(rel_dir):
            data = be.read_json(f"{rel_dir}/{fname}")
            if data is not None:
                files.append(data)
    return files


# ===================================================================
# Public API — Results (annotator)
# ===================================================================

def _ann_rel(version: str, filename: str) -> str:
    base = _get_result_path("annotator_results")
    return f"{base}/{version}/{filename}"


def load_annotator_result(version: str, filename: str) -> dict | None:
    return _get_backend().read_json(_ann_rel(version, filename))


def save_annotator_result(version: str, filename: str, data: dict) -> None:
    _get_backend().write_json(_ann_rel(version, filename), data)


def annotator_result_exists(version: str, filename: str) -> bool:
    return _get_backend().exists(_ann_rel(version, filename))


def list_annotator_result_files(version: str, pattern: str = "*.json") -> list[str]:
    base = _get_result_path("annotator_results")
    files = _get_backend().list_files(f"{base}/{version}")
    return [f for f in files if fnmatch(f, pattern)]


def get_annotator_result_path(version: str, filename: str = "") -> Path:
    base = _get_result_path("annotator_results")
    rel = f"{base}/{version}/{filename}" if filename else f"{base}/{version}"
    return _get_backend().get_local_path(rel)


# ===================================================================
# Public API — Results (benchmark)
# ===================================================================

def _bench_rel(version: str, *parts: str) -> str:
    base = _get_result_path("benchmark_results")
    return "/".join([base, version] + list(parts))


def load_benchmark_result(version: str, *path_parts: str) -> dict | None:
    return _get_backend().read_json(_bench_rel(version, *path_parts))


def save_benchmark_result(version: str, *path_parts: str, data: dict) -> None:
    _get_backend().write_json(_bench_rel(version, *path_parts), data)


def list_benchmark_result_files(version: str, *prefix_parts: str) -> list[str]:
    base = _get_result_path("benchmark_results")
    rel = "/".join([base, version] + list(prefix_parts))
    return _get_backend().list_files(rel)


def get_benchmark_result_path(version: str, *path_parts: str) -> Path:
    return _get_backend().get_local_path(_bench_rel(version, *path_parts))
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from annotator.core.storage import get_backend, load_all_transcripts; print(f'Backend: {get_backend()}, Transcripts: {len(load_all_transcripts())}')"`
Expected: `Backend: local, Transcripts: 104`

- [ ] **Step 3: Verify eval still works**

Run: `python -m annotator.eval.eval --version v4 --mode detections 2>&1 | head -5`
Expected: Loads detections, prints scorecard.

- [ ] **Step 4: Commit**

```bash
git add annotator/core/storage.py
git commit -m "refactor: Factor IV storage — backend ABC, env var path overrides"
```

---

### Task 2: Simplify config.yaml and create .env.example

**Files:**
- Modify: `config.yaml`
- Create: `.env.example`

- [ ] **Step 1: Rewrite storage config section**

Replace the current triple-section (local/s3/paths) with one section:

```yaml
# Storage backend config (Factor IV: backing services as attached resources)
#
# Config provides defaults for local development.
# All values overridable by env vars for deployment — see .env.example.
storage:
  backend: local          # Override: STORAGE_BACKEND
  bucket: kylel-alexisr-edu  # Override: S3_BUCKET
  prefix: synth-students/tutor-bench  # Override: S3_PREFIX
  root: ""                # Override: STORAGE_ROOT (local only, empty = repo root)
  paths:
    transcripts:
      - data/transcripts
    ground_truth:
      - data/ground_truth
    annotator_results: results/annotator
    benchmark_results: results/benchmark
```

- [ ] **Step 2: Create .env.example**

```
# Storage backend: 'local' or 's3'
STORAGE_BACKEND=local

# Local mode: filesystem root (absolute path, or empty for repo root)
# STORAGE_ROOT=

# S3 mode
# S3_BUCKET=kylel-alexisr-edu
# S3_PREFIX=synth-students/tutor-bench

# Path overrides (comma-separated for lists, override config.yaml paths)
# For S3 deployment, uncomment these to map to S3 directory layout:
# STORAGE_TRANSCRIPTS=de-identified/transfer_1,de-identified/transfer_2
# STORAGE_GROUND_TRUTH=annotations
# STORAGE_ANNOTATOR_RESULTS=results/annotator
# STORAGE_BENCHMARK_RESULTS=results/benchmark

# API keys
# GOOGLE_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml .env.example
git commit -m "config: simplify storage to single paths section, add .env.example"
```

---

### Task 3: Add tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_storage.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create test infrastructure**

```bash
mkdir -p tests
touch tests/__init__.py
```

Add to requirements.txt:
```
pytest
moto[s3]
```

- [ ] **Step 2: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
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
    # Clear cached state
    import annotator.core.config as cfg_mod
    cfg_mod._loaded_config = None
    import annotator.core.storage as st
    st._cache.clear()
    st._backend = None
    yield temp_data
    # Teardown
    st._backend = None
    st._cache.clear()
```

- [ ] **Step 3: Create test_storage.py**

```python
# tests/test_storage.py
"""Storage layer tests — verifies Factor IV compliance."""
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
            # Seed transcript
            s3.put_object(Bucket="test-bucket",
                          Key="test-prefix/data/transcripts/conv_s3.json",
                          Body=json.dumps({"conversation_id": "conv_s3", "turns": []}))

            import annotator.core.storage as st
            st._backend = None  # reset to pick up mocked client

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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/ requirements.txt
git commit -m "test: add storage layer tests (local + S3 with moto)"
```

---

### Task 4: Final verification

**Files:** None

- [ ] **Step 1: Verify all pipeline modules import**

```bash
python -c "
from annotator.core.detect import main; print('detect OK')
from annotator.core.annotate import main; print('annotate OK')
from annotator.core.label import main; print('label OK')
from annotator.eval.eval import main; print('eval OK')
from annotator.eval.view import main; print('view OK')
from benchmark.run import main; print('benchmark OK')
from benchmark.eval.eval import main; print('bench eval OK')
from benchmark.eval.view import main; print('bench view OK')
"
```

- [ ] **Step 2: Run eval end-to-end**

```bash
python -m annotator.eval.eval --version v4 --mode detections 2>&1 | head -15
```

Expected: Same output as before refactor (Cluster Recall 0.642, etc.)

- [ ] **Step 3: Run all tests**

```bash
pytest tests/ -v
```

- [ ] **Step 4: Commit and push**

```bash
git add -A
git commit -m "Factor IV storage refactor complete"
git push
```

---

## Factor IV Compliance After This Refactor

| Principle | Before | After |
|-----------|--------|-------|
| **Config sections** | 3 (local/s3/paths) with different path schemas | 1 `paths:` section, env var overrides |
| **Backend branching** | `if backend == "s3"` in every public function | Backend ABC — zero branching |
| **Swap test** | Change `backend:` + remap paths in config | Set env vars: `STORAGE_BACKEND=s3 S3_BUCKET=... STORAGE_TRANSCRIPTS=...` |
| **Resource handles** | Spread across config sections | Env vars per Factor III |
| **Code changes to swap** | None | None |
| **Config file changes to swap** | Yes (different paths per section) | No (env vars override) |
