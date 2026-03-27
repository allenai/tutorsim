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

from dotenv import load_dotenv

from .config import load_config

load_dotenv()

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
# Public API -- Transcripts
# ===================================================================

def load_transcript(conv_id: str) -> dict | None:
    """Load a single transcript JSON by conversation ID.
    Searches all configured transcript paths."""
    be = _get_backend()
    for rel_dir in _get_path_list("transcripts"):
        data = be.read_json(f"{rel_dir}/{conv_id}.json")
        if data is not None:
            return data
    return None


def load_all_transcripts() -> dict[str, dict]:
    """Load all transcripts from all configured transcript paths, merged into one pool.
    Returns {conv_id: conversation_dict}."""
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
    """List all available conversation IDs across all configured transcript paths."""
    be = _get_backend()
    ids = set()
    for rel_dir in _get_path_list("transcripts"):
        for fname in be.list_files(rel_dir):
            ids.add(fname.replace(".json", ""))
    return sorted(ids)


# ===================================================================
# Public API -- Ground Truth
# ===================================================================

def load_ground_truth_file(conv_id: str) -> dict | None:
    """Load a single ground truth JSON by conversation ID."""
    be = _get_backend()
    for rel_dir in _get_path_list("ground_truth"):
        data = be.read_json(f"{rel_dir}/{conv_id}.json")
        if data is not None:
            return data
    return None


def load_all_ground_truth_files() -> list[dict]:
    """Load all ground truth JSON files from all configured ground truth paths.
    Returns list of dicts."""
    be = _get_backend()
    files = []
    for rel_dir in _get_path_list("ground_truth"):
        for fname in be.list_files(rel_dir):
            data = be.read_json(f"{rel_dir}/{fname}")
            if data is not None:
                files.append(data)
    return files


# ===================================================================
# Public API -- Results (annotator)
# ===================================================================

def _ann_rel(version: str, filename: str) -> str:
    base = _get_result_path("annotator_results")
    return f"{base}/{version}/{filename}"


def load_annotator_result(version: str, filename: str) -> dict | None:
    """Load a result file from results/annotator/{version}/{filename}."""
    return _get_backend().read_json(_ann_rel(version, filename))


def save_annotator_result(version: str, filename: str, data: dict) -> None:
    """Save a result file to results/annotator/{version}/{filename}."""
    _get_backend().write_json(_ann_rel(version, filename), data)


def annotator_result_exists(version: str, filename: str) -> bool:
    """Check if a result file exists."""
    return _get_backend().exists(_ann_rel(version, filename))


def list_annotator_result_files(version: str, pattern: str = "*.json") -> list[str]:
    """List result filenames matching pattern in results/annotator/{version}/."""
    base = _get_result_path("annotator_results")
    files = _get_backend().list_files(f"{base}/{version}")
    return [f for f in files if fnmatch(f, pattern)]


def get_annotator_result_path(version: str, filename: str = "") -> Path:
    """Get the local Path for an annotator result (for local backend only).
    Used by code that needs to write non-JSON files (e.g., JSONL, HTML)."""
    base = _get_result_path("annotator_results")
    rel = f"{base}/{version}/{filename}" if filename else f"{base}/{version}"
    return _get_backend().get_local_path(rel)


# ===================================================================
# Public API -- Results (benchmark)
# ===================================================================

def _bench_rel(version: str, *parts: str) -> str:
    base = _get_result_path("benchmark_results")
    return "/".join([base, version] + list(parts))


def load_benchmark_result(version: str, *path_parts: str) -> dict | None:
    """Load a result file from results/benchmark/{version}/{path_parts joined}.
    Example: load_benchmark_result('v1', 'exchanges', 'anthropic', 'scenario_123.json')"""
    return _get_backend().read_json(_bench_rel(version, *path_parts))


def save_benchmark_result(version: str, *path_parts: str, data: dict) -> None:
    """Save a result file to results/benchmark/{version}/{path_parts joined}."""
    _get_backend().write_json(_bench_rel(version, *path_parts), data)


def list_benchmark_result_files(version: str, *prefix_parts: str) -> list[str]:
    """List files under results/benchmark/{version}/{prefix_parts joined}/."""
    base = _get_result_path("benchmark_results")
    rel = "/".join([base, version] + list(prefix_parts))
    return _get_backend().list_files(rel)


def get_benchmark_result_path(version: str, *path_parts: str) -> Path:
    """Get the local Path for a benchmark result (for local backend only)."""
    return _get_backend().get_local_path(_bench_rel(version, *path_parts))
