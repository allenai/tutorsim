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
import logging
import os
import re
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from .config import load_config

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent


def _parse_timestamp_seconds(ts: str) -> float:
    """Best-effort: turn a timestamp string into seconds.

    Accepts 'MM:SS-MM:SS', 'M:SS', '{seconds}s', or a bare number.
    Returns 0.0 on any failure so callers don't have to handle malformed data.
    """
    if not ts:
        return 0.0
    s = ts.strip()
    # range form: keep only the first half
    if "-" in s:
        s = s.split("-", 1)[0].strip()
    # '123.4s' form
    if s.endswith("s") and s[:-1].replace(".", "", 1).isdigit():
        try:
            return float(s[:-1])
        except ValueError:
            return 0.0
    # 'MM:SS' or 'HH:MM:SS' form
    if ":" in s:
        parts = s.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return 0.0
        # MM:SS -> 60*M + S; HH:MM:SS -> 3600*H + 60*M + S
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        return 0.0
    # bare number
    try:
        return float(s)
    except ValueError:
        return 0.0


_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_CONV_ID_PREFIX_RE = re.compile(r"^\d{4}-t\d+_\d{4}-s\d+_")


def _conv_id_to_uuid(conv_id: str) -> str:
    """Extract the transcript-UUID component from a full conv_id.

    Accepts:
      - bare UUIDs (returned as-is)
      - legacy composites `2024-tN_2024-sN_{uuid}` (UUID extracted)
      - bench composites `{tutor_uuid}_{student_uuid}_{transcript_uuid}`
        (LAST UUID returned — transcript_id is the join key for GT/screenshots)
    """
    matches = _UUID_RE.findall(conv_id)
    if matches:
        return matches[-1]
    # Strip '{year-tN}_{year-sN}_' prefix if present (handles shortened test UUIDs)
    stripped = _CONV_ID_PREFIX_RE.sub("", conv_id)
    if stripped != conv_id:
        return stripped
    return conv_id


def _annotate_turns_with_start_seconds(conv: dict) -> dict:
    """Add a start_seconds float to each turn if missing. Mutates and returns conv."""
    for turn in conv.get("turns", []):
        if "start_seconds" not in turn:
            turn["start_seconds"] = _parse_timestamp_seconds(turn.get("timestamp", ""))
    return conv


_cache: dict[str, object] = {}
_jsonl_indexes: dict[str, dict[str, dict]] = {}  # path -> {conv_id: transformed_record}


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

    @abstractmethod
    def read_bytes(self, rel_path: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, rel_path: str, data: bytes) -> None: ...

    @abstractmethod
    def get_presigned_url(self, rel_path: str, expires_seconds: int = 172800) -> str: ...


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
        # Atomic write: write to temp file then rename, so a crash mid-write
        # doesn't leave a truncated/corrupted target file.
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        # OneDrive can hold a transient lock on the target file while syncing.
        # Retry the rename a few times before giving up.
        last_err = None
        for delay in (0, 0.2, 0.5, 1.0, 2.0):
            if delay:
                import time
                time.sleep(delay)
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as e:
                last_err = e
        raise last_err

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

    def read_bytes(self, rel_path: str) -> bytes:
        path = self.root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
        with open(path, "rb") as f:
            return f.read()

    def write_bytes(self, rel_path: str, data: bytes) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)

    def get_presigned_url(self, rel_path: str, expires_seconds: int = 172800) -> str:
        path = (self.root / rel_path).resolve()
        return path.as_uri()


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

    def read_bytes(self, rel_path: str) -> bytes:
        key = self._key(rel_path)
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"s3://{self.bucket}/{key}")

    def write_bytes(self, rel_path: str, data: bytes) -> None:
        key = self._key(rel_path)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_presigned_url(self, rel_path: str, expires_seconds: int = 172800) -> str:
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": self._key(rel_path)},
            ExpiresIn=expires_seconds,
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
# JSONL transcript support
# ===================================================================

def _transform_normalized_record(rec: dict) -> dict:
    """Transform an S3 normalized JSONL record to our internal transcript format.

    S3 format: turns[] (dialogue only) + enrichments[] (non-dialogue, with before_turn).
    Our format: turns[] with dialogue turns keeping their ORIGINAL turn_number (so they
    match human-annotated turn ranges in the ground truth). Enrichments are inserted
    before their associated dialogue turn with the same turn_number and is_enrichment=True,
    so they appear in excerpts as context but don't shift turn numbering.
    """
    sess = rec.get("session", {})
    source_id = rec.get("transcript_id", "") or rec.get("source_id", "")
    tutor_id = sess.get("tutor_id", "") or sess.get("source_tutor_id", "")
    student_id = sess.get("student_id", "") or sess.get("source_student_id", "")

    # source_id format varies by batch:
    #   older batches: UUID only (e.g. "69b80b21-...")
    #   newer batches: full conv_id (e.g. "2025-t27247_2025-s12069_69b80b21-...")
    if tutor_id and tutor_id in source_id:
        conv_id = source_id  # already includes tutor_student prefix
    else:
        conv_id = f"{tutor_id}_{student_id}_{source_id}"

    # Index dialogue turns by their original turn_number
    raw_turns = rec.get("turns", [])
    max_dialogue_turn = max((t["turn_number"] for t in raw_turns), default=0)

    # Group enrichments by the dialogue turn they precede
    from collections import defaultdict as _defaultdict
    enrichments_by_turn = _defaultdict(list)
    trailing_enrichments = []
    for e in rec.get("enrichments", []):
        etype = e.get("type", "").upper().replace(" ", "_")
        label = e.get("label", "")
        content = e.get("content", "")
        text = f"[{etype}]"
        if label:
            text = f"[{etype}: {label}]"
        if content:
            text = f"{text} {content}"
        ss = float(e.get("start_seconds", 0) or 0)
        entry = {
            "role": "TUTOR",
            "text": text,
            "type": etype,
            "timestamp": f"{ss}s",
            "start_seconds": ss,
            "is_enrichment": True,
        }
        before = e.get("before_turn") or 1  # None/0 → before turn 1
        if before <= max_dialogue_turn:
            enrichments_by_turn[before].append(entry)
        else:
            trailing_enrichments.append(entry)

    # Build final list: enrichments before each dialogue turn, preserving original numbering
    final_turns = []
    for t in raw_turns:
        turn_num = t["turn_number"]
        for e_entry in enrichments_by_turn.get(turn_num, []):
            final_turns.append({**e_entry, "turn_number": turn_num})
        ss = float(t.get("start_seconds", 0) or 0)
        final_turns.append({
            "turn_number": turn_num,
            "role": t["role"].upper(),
            "text": t["text"],
            "type": "DIALOGUE",
            "timestamp": f"{ss}s",
            "start_seconds": ss,
            "is_enrichment": False,
        })
    for e_entry in trailing_enrichments:
        final_turns.append({**e_entry, "turn_number": max_dialogue_turn})

    # Build context from demographics if available
    demo = rec.get("demographics", {})
    student = demo.get("student", {})
    context_parts = []
    if student.get("grade"):
        context_parts.append(f"Grade {student['grade']}")
    if student.get("subject"):
        context_parts.append(student["subject"])
    context = ", ".join(context_parts)

    return {
        "conversation_id": conv_id,
        "transcript_id": source_id,
        "tutor_id": tutor_id,
        "student_id": student_id,
        "context": context,
        "platform": rec.get("source", "step_up"),
        "num_turns": len(raw_turns),  # dialogue turns only
        "turns": final_turns,
    }


def _load_jsonl_index(path: str) -> dict[str, dict]:
    """Load a JSONL file from the backend, transform each record, index by conv_id.

    Cached in _jsonl_indexes so the file is only loaded once per process.
    """
    if path in _jsonl_indexes:
        return _jsonl_indexes[path]

    be = _get_backend()
    logger.info("Loading JSONL transcript index from %s...", path)

    import io

    stream = None

    # Try local file first
    if isinstance(be, LocalBackend):
        full_path = be.root / path
        if full_path.exists():
            stream = open(full_path, "r", encoding="utf-8")

    # Try S3 (either as primary backend or fallback for JSONL paths not found locally)
    if stream is None:
        bucket = _get_bucket()
        if bucket:
            try:
                import boto3
                s3_client = boto3.client("s3")
                prefix = _get_prefix()
                key = f"{prefix}/{path}" if prefix else path
                resp = s3_client.get_object(Bucket=bucket, Key=key)
                stream = io.TextIOWrapper(resp["Body"], encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to load JSONL from S3: %s", e)

    if stream is None:
        logger.warning("JSONL file not found: %s", path)
        _jsonl_indexes[path] = {}
        return {}

    index = {}
    errors = 0
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                transformed = _transform_normalized_record(rec)
                conv_id = transformed["conversation_id"]
                index[conv_id] = transformed
            except Exception:
                errors += 1
    finally:
        stream.close()

    logger.info("Indexed %d transcripts (%d parse errors)", len(index), errors)
    _jsonl_indexes[path] = index
    return index


# ===================================================================
# Public API -- Transcripts
# ===================================================================

def load_transcript(conv_id: str) -> dict | None:
    """Load a single transcript JSON by conversation ID.
    Searches all configured transcript paths. Supports JSONL sources.
    Adds start_seconds to each turn if missing."""
    be = _get_backend()
    for rel_path in _get_path_list("transcripts"):
        if rel_path.endswith(".jsonl"):
            index = _load_jsonl_index(rel_path)
            if conv_id in index:
                return _annotate_turns_with_start_seconds(index[conv_id])
        else:
            data = be.read_json(f"{rel_path}/{conv_id}.json")
            if data is not None:
                return _annotate_turns_with_start_seconds(data)
    return None


def load_all_transcripts() -> dict[str, dict]:
    """Load all transcripts from all configured transcript paths, merged into one pool.
    Returns {conv_id: conversation_dict}. Supports JSONL sources.
    Adds start_seconds to each turn if missing."""
    be = _get_backend()
    transcripts = {}
    for rel_path in _get_path_list("transcripts"):
        if rel_path.endswith(".jsonl"):
            for conv_id, conv in _load_jsonl_index(rel_path).items():
                transcripts[conv_id] = _annotate_turns_with_start_seconds(conv)
        else:
            for fname in be.list_files(rel_path):
                data = be.read_json(f"{rel_path}/{fname}")
                if data and "conversation_id" in data:
                    transcripts[data["conversation_id"]] = _annotate_turns_with_start_seconds(data)
                elif data:
                    transcripts[fname.replace(".json", "")] = _annotate_turns_with_start_seconds(data)
    return transcripts


def list_transcript_ids() -> list[str]:
    """List all available conversation IDs across all configured transcript paths.
    Supports JSONL sources."""
    be = _get_backend()
    ids = set()
    for rel_path in _get_path_list("transcripts"):
        if rel_path.endswith(".jsonl"):
            ids.update(_load_jsonl_index(rel_path).keys())
        else:
            for fname in be.list_files(rel_path):
                ids.add(fname.replace(".json", ""))
    return sorted(ids)


# ===================================================================
# Public API -- Ground Truth
# ===================================================================

def load_ground_truth_file(conv_id: str) -> dict | None:
    """Load a single ground truth JSON by conversation ID.

    GT files are keyed by transcript_id (UUID), while loaded transcripts may
    use composite conv_ids like `{tutor}_{student}_{transcript}`. Falls back
    to the extracted UUID if the direct lookup misses.
    """
    be = _get_backend()
    fallback_id = _conv_id_to_uuid(conv_id)
    for rel_dir in _get_path_list("ground_truth"):
        data = be.read_json(f"{rel_dir}/{conv_id}.json")
        if data is not None:
            return data
        if fallback_id != conv_id:
            data = be.read_json(f"{rel_dir}/{fallback_id}.json")
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
# Public API -- Annotator shards (incremental per-conv writes)
# ===================================================================
#
# Shards live under results/annotator/{version}/shards/{basename}/{conv_id}.json
# where `basename` is the output filename without extension (e.g. "detections",
# "annotations", "annotations_generous"). Each shard stores the per-conv slice
# of what would otherwise be inside the monolithic output's `results` dict.
#
# This enables Ctrl-C resume: a re-run skips conv_ids that already have shards
# and only sends the remaining work to the model.

def _ann_shard_dir(version: str, basename: str) -> str:
    base = _get_result_path("annotator_results")
    return f"{base}/{version}/shards/{basename}"


def save_annotator_shard(version: str, basename: str, conv_id: str, data: dict) -> None:
    """Write one conv's slice to its own file. Called incrementally as results parse."""
    rel = f"{_ann_shard_dir(version, basename)}/{conv_id}.json"
    _get_backend().write_json(rel, data)


def list_annotator_shard_ids(version: str, basename: str) -> list[str]:
    """Return conv_ids that already have a shard for (version, basename)."""
    files = _get_backend().list_files(_ann_shard_dir(version, basename))
    return sorted(f[:-5] for f in files if f.endswith(".json"))


def load_annotator_shards(version: str, basename: str) -> dict[str, dict]:
    """Load every shard for (version, basename) as {conv_id: data}."""
    be = _get_backend()
    shard_dir = _ann_shard_dir(version, basename)
    out = {}
    for fname in be.list_files(shard_dir):
        if not fname.endswith(".json"):
            continue
        conv_id = fname[:-5]
        data = be.read_json(f"{shard_dir}/{fname}")
        if data is not None:
            out[conv_id] = data
    return out


# ===================================================================
# Public API -- In-flight batch sidecars (ctrl-C resume during poll)
# ===================================================================
#
# When a batch is submitted to a provider, the batch keeps running server-side
# even if our process exits. We persist the provider's batch ID to a sidecar
# at results/annotator/{version}/in_flight/{basename}.json BEFORE entering the
# poll loop, so a re-run after ctrl-C can resume polling on the same batch
# instead of re-submitting and double-charging compute.
#
# The sidecar is deleted only after the batch's results are successfully
# parsed and sharded.

def _inflight_rel(version: str, basename: str) -> str:
    # Subdirectory keeps the sidecar out of list_annotator_result_files results.
    base = _get_result_path("annotator_results")
    return f"{base}/{version}/in_flight/{basename}.json"


def save_inflight_batch(version: str, basename: str, data: dict) -> None:
    """Record an in-flight batch's metadata. Expected keys:
    provider, model, batch_id, n_entries, entry_keys_hash, display_name, submitted_at."""
    _get_backend().write_json(_inflight_rel(version, basename), data)


def load_inflight_batch(version: str, basename: str) -> dict | None:
    """Return the recorded in-flight batch metadata, or None if no batch is in flight."""
    return _get_backend().read_json(_inflight_rel(version, basename))


def clear_inflight_batch(version: str, basename: str) -> None:
    """Delete the in-flight sidecar after a batch completes successfully."""
    be = _get_backend()
    rel = _inflight_rel(version, basename)
    if isinstance(be, LocalBackend):
        path = be.root / rel
        if path.exists():
            path.unlink()
    else:
        try:
            be.client.delete_object(Bucket=be.bucket, Key=be._key(rel))
        except Exception:
            pass


# ===================================================================
# Public API -- Benchmark in-flight batch sidecars (per profile + style)
# ===================================================================
#
# Mirrors the annotator sidecar helpers above, but namespaced under
# results/benchmark/{version}/in_flight/{profile}_{style}.json so each
# (tutor profile, annotator style) batch tracks independently.

def _bench_inflight_rel(version: str, profile: str, style: str) -> str:
    base = _get_result_path("benchmark_results")
    return f"{base}/{version}/in_flight/{profile}_{style}.json"


def save_benchmark_inflight_batch(version: str, profile: str, style: str,
                                   data: dict) -> None:
    """Record an in-flight benchmark annotation batch's metadata."""
    _get_backend().write_json(_bench_inflight_rel(version, profile, style), data)


def load_benchmark_inflight_batch(version: str, profile: str,
                                   style: str) -> dict | None:
    """Return the recorded in-flight benchmark batch metadata, or None."""
    return _get_backend().read_json(_bench_inflight_rel(version, profile, style))


def clear_benchmark_inflight_batch(version: str, profile: str, style: str) -> None:
    """Delete the benchmark in-flight sidecar after a batch completes."""
    be = _get_backend()
    rel = _bench_inflight_rel(version, profile, style)
    if isinstance(be, LocalBackend):
        path = be.root / rel
        if path.exists():
            path.unlink()
    else:
        try:
            be.client.delete_object(Bucket=be.bucket, Key=be._key(rel))
        except Exception:
            pass


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


# ===================================================================
# Public API -- Screenshots
# ===================================================================

def _screenshot_root() -> str:
    """Return the configured screenshots prefix (e.g. 'deidentified/screenshots')."""
    paths = _get_path_list("screenshots")
    if not paths:
        return "deidentified/screenshots"
    return paths[0]


def _screenshot_rel_path(conv_id: str, filename: str) -> str:
    return f"{_screenshot_root()}/{_conv_id_to_uuid(conv_id)}/{filename}"


def list_screenshots(conv_id: str) -> list[str]:
    """Return screenshot filenames for a conversation, sorted.

    Returns [] if the conversation has no screenshots. Skips the _metadata.json
    sidecar and any non-image entries.
    """
    be = _get_backend()
    uuid = _conv_id_to_uuid(conv_id)
    prefix = f"{_screenshot_root()}/{uuid}"

    if isinstance(be, LocalBackend):
        directory = be.root / prefix
        if not directory.exists():
            return []
        names = [p.name for p in directory.iterdir() if p.is_file()]
    else:
        # S3: use a dedicated listing since list_files is json-only
        prefix_key = be._key(prefix.rstrip("/") + "/")
        names = []
        paginator = be.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=be.bucket, Prefix=prefix_key):
            for obj in page.get("Contents", []):
                names.append(obj["Key"].rsplit("/", 1)[-1])

    return sorted(
        n for n in names
        if not n.startswith("_") and n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )


def load_screenshot_bytes(conv_id: str, filename: str) -> bytes:
    return _get_backend().read_bytes(_screenshot_rel_path(conv_id, filename))


def get_screenshot_uri(conv_id: str, filename: str) -> str:
    return _get_backend().get_presigned_url(_screenshot_rel_path(conv_id, filename))


def load_screenshot_verification(conv_id: str) -> dict:
    """Read _metadata.json for a conv. Returns {} if absent."""
    be = _get_backend()
    uuid = _conv_id_to_uuid(conv_id)
    rel = f"{_screenshot_root()}/{uuid}/_metadata.json"
    data = be.read_json(rel)
    return data or {}
