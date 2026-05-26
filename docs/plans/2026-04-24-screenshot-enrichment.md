# Screenshot Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end support for passing anchored screenshots from S3 into the detection and annotation passes of the annotator pipeline, with the image-attachment capability built as a generic `ModelClient` feature so benchmark stages can adopt it later without client-layer changes.

**Architecture:** Screenshots already live on S3 under `deidentified/screenshots/{uuid}/{timestamp_seconds}.jpg` with a sibling `_metadata.json`. We add (1) binary + presigned-URL methods to the existing `StorageBackend` ABC, (2) a new `screenshots.py` module that anchors each image to the latest transcript turn whose timestamp ≤ the image's filename-encoded timestamp, (3) an optional `images=[storage_path, ...]` kwarg on `ModelClient.generate` and `build_batch_entry` that resolves per-provider (base64 inline for local/Gemini, presigned URL for S3+Claude/OpenAI), and (4) `--with-screenshots` CLI flags on `detect.py` and `annotate.py`. Detection attaches every image for a conversation. Annotation filters per-moment to images whose anchor turn falls inside the excerpt window. Default off everywhere; existing runs are byte-for-byte unchanged.

**Tech Stack:** Python 3, existing `annotator` package, `pytest` + `moto` for tests, Anthropic/OpenAI/Gemini SDKs already pinned in the project.

**Reference spec:** `docs/superpowers/specs/2026-04-24-screenshot-enrichment-design.md`

---

## File Structure

**Created:**
- `annotator/core/screenshots.py` — anchoring logic and the `load_anchored_screenshots` entry point
- `tests/test_screenshots.py` — unit tests for the new module

**Modified:**
- `annotator/core/storage.py` — add `read_bytes`, `write_bytes`, `get_presigned_url` to `StorageBackend` ABC + both backends; add screenshot helpers; preserve `start_seconds` on transformed turns
- `annotator/core/client.py` — add `images` kwarg to `generate` + all three `_generate_*`; add `images` to `build_batch_entry` + all three batch runners; add `_mime_from_path`, `validate_vision_support`, `_build_image_blocks_*` helpers; add `enable_cache` kwarg for Anthropic
- `annotator/core/utils.py` — parse `timestamp` string to `start_seconds` at load; extend `format_transcript` and `format_excerpt` with `screenshots` param
- `annotator/core/detect.py` — `--with-screenshots` flag, vision validation, image-aware output JSON fields
- `annotator/core/annotate.py` — `--with-screenshots` flag, per-moment image filtering, cache on for screenshot runs
- `config.yaml` — add `storage.paths.screenshots`
- `.env.example` — add `STORAGE_SCREENSHOTS`
- `tests/test_storage.py` — add tests for new binary/URI methods and screenshot helpers
- `tests/test_client.py` — add tests for `_mime_from_path`, `validate_vision_support`, images in `build_batch_entry`
- `tests/conftest.py` — add a screenshot fixture layout to `temp_data`
- `tests/test_utils.py` — add tests for `format_transcript` and `format_excerpt` with screenshots

---

## Task 1: Storage binary I/O and presigned URL methods

**Files:**
- Modify: `annotator/core/storage.py` — extend `StorageBackend` ABC at lines 104–120, `LocalBackend` at 123–161, `S3Backend` at 164–226
- Modify: `tests/test_storage.py` — extend `TestLocalBackend` and `TestS3Backend` classes

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_storage.py`, inside `class TestLocalBackend`:

```python
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
```

Append to `class TestS3Backend`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -k "bytes or presigned" -v`
Expected: FAIL with "has no attribute 'read_bytes'" / "has no attribute 'write_bytes'" / "has no attribute 'get_presigned_url'"

- [ ] **Step 3: Extend the ABC**

In `annotator/core/storage.py`, replace the `class StorageBackend(ABC)` block (lines 104–120) with:

```python
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
```

- [ ] **Step 4: Implement on LocalBackend**

In `annotator/core/storage.py`, append to `class LocalBackend` (after `get_local_path`):

```python
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
```

- [ ] **Step 5: Implement on S3Backend**

In `annotator/core/storage.py`, append to `class S3Backend` (after `get_local_path`):

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -k "bytes or presigned" -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add annotator/core/storage.py tests/test_storage.py
git commit -m "feat(storage): add read_bytes, write_bytes, get_presigned_url to backends"
```

---

## Task 2: Turn `start_seconds` normalization

**Context:** Anchoring screenshots requires each turn to carry a numeric start-time in seconds. Today, JSONL-sourced turns get a string like `"603.834s"` in their `timestamp` field; consolidated-JSON turns get `"MM:SS-MM:SS"`. This task adds `start_seconds: float` alongside the existing `timestamp` string without removing it.

**Files:**
- Modify: `annotator/core/storage.py:272-278` (JSONL transform — stop stringifying start_seconds; add it as float alongside)
- Modify: `annotator/core/storage.py` public API — add a `_parse_timestamp_seconds` helper and apply it in `load_transcript` / `load_all_transcripts` when loading JSON-format transcripts
- Modify: `tests/test_storage.py` — add tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_storage.py`, inside `class TestLocalBackend`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -k "start_seconds" -v`
Expected: FAIL with `KeyError: 'start_seconds'`.

- [ ] **Step 3: Add the parse helper**

In `annotator/core/storage.py`, after the `REPO_ROOT = ...` line (around line 31), add:

```python
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


def _annotate_turns_with_start_seconds(conv: dict) -> dict:
    """Add a start_seconds float to each turn if missing. Mutates and returns conv."""
    for turn in conv.get("turns", []):
        if "start_seconds" not in turn:
            turn["start_seconds"] = _parse_timestamp_seconds(turn.get("timestamp", ""))
    return conv
```

- [ ] **Step 4: Apply it in `load_transcript` and `load_all_transcripts`**

In `annotator/core/storage.py`, find `def load_transcript(conv_id: str) -> dict | None:` (around line 402). Replace its body with:

```python
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
```

Find `def load_all_transcripts()` (just below it) and wrap each loaded conv in `_annotate_turns_with_start_seconds` the same way. Its existing body loops over paths and accumulates into a dict — apply the helper on each value before inserting.

- [ ] **Step 5: Update the JSONL transform to preserve start_seconds**

In `annotator/core/storage.py:272-278` (inside `_transform_normalized_record`), replace the dialogue_turns append with:

```python
    dialogue_turns = []
    for t in rec.get("turns", []):
        ss = float(t.get("start_seconds", 0) or 0)
        dialogue_turns.append({
            "_sort_key": (t["turn_number"], 1),  # dialogue after enrichments at same position
            "role": t["role"].upper(),
            "text": t["text"],
            "type": "DIALOGUE",
            "timestamp": f"{ss}s",
            "start_seconds": ss,
        })
```

And the enrichment_turns append (lines 281-298) similarly to include a numeric `start_seconds` on every emitted turn:

```python
    enrichment_turns = []
    for e in rec.get("enrichments", []):
        etype = e.get("type", "").upper().replace(" ", "_")
        role = "TUTOR"
        label = e.get("label", "")
        content = e.get("content", "")
        text = f"[{etype}]"
        if label:
            text = f"[{etype}: {label}]"
        if content:
            text = f"{text} {content}"
        ss = float(e.get("start_seconds", 0) or 0)

        enrichment_turns.append({
            "_sort_key": (e.get("before_turn") or 0, 0),
            "role": role,
            "text": text,
            "type": etype,
            "timestamp": f"{ss}s",
            "start_seconds": ss,
        })
```

Finally, inside the numbered-turn builder a few lines below (lines 305-313), carry `start_seconds` through:

```python
    numbered = []
    for i, t in enumerate(all_turns, start=1):
        numbered.append({
            "turn_number": i,
            "role": t["role"],
            "text": t["text"],
            "type": t["type"],
            "timestamp": t["timestamp"],
            "start_seconds": t["start_seconds"],
        })
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: all pass (including the two new `start_seconds` tests).

- [ ] **Step 7: Commit**

```bash
git add annotator/core/storage.py tests/test_storage.py
git commit -m "feat(storage): add start_seconds to every turn at load time"
```

---

## Task 3: Screenshot storage helpers

**Files:**
- Modify: `annotator/core/storage.py` — add screenshot helpers; add `screenshots` path category
- Modify: `tests/conftest.py` — extend `temp_data` fixture with a screenshot layout
- Modify: `tests/test_storage.py` — add screenshot helper tests
- Modify: `config.yaml` — add `screenshots` path
- Modify: `.env.example` — add `STORAGE_SCREENSHOTS`

- [ ] **Step 1: Add the fixture layout**

In `tests/conftest.py`, extend `temp_data` to create screenshots:

```python
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
```

Also update the `local_storage` fixture to set the screenshots path:

```python
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
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_storage.py`, inside `class TestLocalBackend`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -k "screenshot" -v`
Expected: FAIL with `ImportError: cannot import name 'list_screenshots'`.

- [ ] **Step 4: Add the `screenshots` path category and UUID extraction helper**

In `annotator/core/storage.py`, add a small UUID helper near the top, below `_parse_timestamp_seconds`:

```python
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def _conv_id_to_uuid(conv_id: str) -> str:
    """Extract the UUID component from a full conv_id.

    Accepts bare UUIDs (returned as-is) and full conv_ids like
    '2024-tN_2024-sN_099bf759-...' (UUID extracted).
    """
    m = _UUID_RE.search(conv_id)
    if m:
        return m.group(0)
    # Fallback: assume it's already a UUID-like string
    return conv_id
```

Add the required import at the top of the file (if not present):
```python
import re
```

- [ ] **Step 5: Implement the screenshot helpers**

In `annotator/core/storage.py`, add a new section after the existing public API sections (near the bottom of the file):

```python
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
```

- [ ] **Step 6: Update config.yaml and .env.example**

In `config.yaml`, under `storage.paths`, add one line:

```yaml
    screenshots: deidentified/screenshots
```

In `.env.example`, after the other `STORAGE_*` examples, add:

```
# STORAGE_SCREENSHOTS=deidentified/screenshots
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -k "screenshot" -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add annotator/core/storage.py tests/test_storage.py tests/conftest.py config.yaml .env.example
git commit -m "feat(storage): add screenshot helpers and screenshots path category"
```

---

## Task 4: New `screenshots.py` module — anchoring logic

**Files:**
- Create: `annotator/core/screenshots.py`
- Create: `tests/test_screenshots.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_screenshots.py`:

```python
"""Tests for annotator.core.screenshots."""
import pytest


class TestTimestampFromFilename:
    def test_parses_decimal(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        assert timestamp_seconds_from_filename("603.834.jpg") == pytest.approx(603.834)

    def test_parses_integer(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        assert timestamp_seconds_from_filename("120.jpg") == pytest.approx(120.0)

    def test_raises_on_junk(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        with pytest.raises(ValueError):
            timestamp_seconds_from_filename("notanumber.jpg")

    def test_accepts_png(self):
        from annotator.core.screenshots import timestamp_seconds_from_filename
        assert timestamp_seconds_from_filename("50.5.png") == pytest.approx(50.5)


class TestAnchorScreenshots:
    def _turns(self, *ss_pairs):
        return [
            {"turn_number": n, "start_seconds": s}
            for n, s in ss_pairs
        ]

    def test_anchors_to_latest_turn_at_or_before(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 5.0), (3, 10.0))
        # Screenshot at 7s should anchor to turn 2 (5.0 <= 7 < 10.0)
        result = anchor_screenshots(["7.000.jpg"], turns)
        assert len(result) == 1
        assert result[0]["anchor_turn"] == 2
        assert result[0]["filename"] == "7.000.jpg"
        assert result[0]["timestamp_seconds"] == pytest.approx(7.0)

    def test_boundary_exact_match(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 5.0), (3, 10.0))
        result = anchor_screenshots(["5.000.jpg"], turns)
        assert result[0]["anchor_turn"] == 2  # <= is inclusive

    def test_before_all_turns_anchors_to_first(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 10.0), (2, 20.0))
        result = anchor_screenshots(["3.000.jpg"], turns)
        assert result[0]["anchor_turn"] == 1

    def test_after_all_turns_anchors_to_last(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 5.0))
        result = anchor_screenshots(["100.000.jpg"], turns)
        assert result[0]["anchor_turn"] == 2

    def test_sorted_by_anchor_turn(self):
        from annotator.core.screenshots import anchor_screenshots
        turns = self._turns((1, 0.0), (2, 10.0), (3, 20.0))
        result = anchor_screenshots(["25.000.jpg", "5.000.jpg", "15.000.jpg"], turns)
        assert [r["anchor_turn"] for r in result] == [1, 2, 3]

    def test_empty_screenshots_returns_empty(self):
        from annotator.core.screenshots import anchor_screenshots
        assert anchor_screenshots([], []) == []


class TestLoadAnchoredScreenshots:
    def test_filters_flagged_and_eedi_ip(self, local_storage):
        from annotator.core.screenshots import load_anchored_screenshots
        from annotator.core.storage import load_transcript

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)
        result = load_anchored_screenshots(conv_id, conv["turns"])
        # Only 4.000.jpg is usable -- 11.500.jpg has eedi_ip=True
        assert len(result) == 1
        assert result[0]["filename"] == "4.000.jpg"
        assert result[0]["anchor_turn"] == 1  # 4s falls inside turn 1 (starts at 0s)

    def test_empty_when_no_screenshots(self, local_storage):
        from annotator.core.screenshots import load_anchored_screenshots
        conv = {"turns": [{"turn_number": 1, "start_seconds": 0.0}]}
        result = load_anchored_screenshots("nonexistent_conv_id", conv["turns"])
        assert result == []

    def test_storage_path_composed(self, local_storage):
        from annotator.core.screenshots import load_anchored_screenshots
        from annotator.core.storage import load_transcript

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)
        result = load_anchored_screenshots(conv_id, conv["turns"])
        assert result[0]["storage_path"] == "deidentified/screenshots/099bf759-abcd/4.000.jpg"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_screenshots.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'annotator.core.screenshots'`.

- [ ] **Step 3: Create `annotator/core/screenshots.py`**

Create `annotator/core/screenshots.py`:

```python
"""Screenshot anchoring and loading helpers.

Screenshots are stored under `deidentified/screenshots/{uuid}/{timestamp}.jpg`
where `timestamp` is the video-timestamp in seconds (filename is the source of
truth). Anchoring a screenshot to a dialogue turn is deterministic: pick the
latest turn whose start_seconds <= screenshot timestamp.
"""

import os
from typing import Iterable

from . import storage


def timestamp_seconds_from_filename(filename: str) -> float:
    """Parse the video timestamp encoded in a screenshot filename.

    >>> timestamp_seconds_from_filename("603.834.jpg")
    603.834
    """
    stem, _ = os.path.splitext(filename)
    return float(stem)


def anchor_screenshots(filenames: Iterable[str], turns: list[dict]) -> list[dict]:
    """Anchor each screenshot to the latest turn with start_seconds <= its timestamp.

    Falls back to the first turn if the screenshot precedes all turns.
    Returns entries sorted by anchor_turn ascending:
        [{"filename", "timestamp_seconds", "anchor_turn"}, ...]
    """
    if not turns:
        return []

    # Turns sorted by start_seconds for deterministic anchoring
    sorted_turns = sorted(turns, key=lambda t: t.get("start_seconds", 0.0))

    out = []
    for fname in filenames:
        ts = timestamp_seconds_from_filename(fname)
        # Find latest turn whose start_seconds <= ts
        chosen = sorted_turns[0]
        for t in sorted_turns:
            if t.get("start_seconds", 0.0) <= ts:
                chosen = t
            else:
                break
        out.append({
            "filename": fname,
            "timestamp_seconds": ts,
            "anchor_turn": chosen["turn_number"],
        })

    out.sort(key=lambda r: (r["anchor_turn"], r["timestamp_seconds"]))
    return out


def load_anchored_screenshots(conv_id: str, turns: list[dict]) -> list[dict]:
    """Load, filter (flagged / eedi_ip), and anchor screenshots for a conv.

    Returns list of dicts with:
      - filename
      - timestamp_seconds
      - anchor_turn
      - storage_path (relative to storage backend root)
    """
    filenames = storage.list_screenshots(conv_id)
    if not filenames:
        return []

    verification = storage.load_screenshot_verification(conv_id)
    flagged = {
        f for f, m in verification.get("images", {}).items()
        if m.get("flagged") or m.get("eedi_ip")
    }
    usable = [f for f in filenames if f not in flagged]

    anchored = anchor_screenshots(usable, turns)
    for row in anchored:
        row["storage_path"] = storage._screenshot_rel_path(conv_id, row["filename"])

    return anchored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_screenshots.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add annotator/core/screenshots.py tests/test_screenshots.py
git commit -m "feat(screenshots): add timestamp-based anchoring module"
```

---

## Task 5: MIME inference and vision-model validation

**Files:**
- Modify: `annotator/core/client.py`
- Modify: `tests/test_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_client.py`:

```python
class TestMimeFromPath:
    def test_jpg(self):
        from annotator.core.client import _mime_from_path
        assert _mime_from_path("foo/bar.jpg") == "image/jpeg"
        assert _mime_from_path("X.JPEG") == "image/jpeg"

    def test_png(self):
        from annotator.core.client import _mime_from_path
        assert _mime_from_path("a/b/c.png") == "image/png"

    def test_webp(self):
        from annotator.core.client import _mime_from_path
        assert _mime_from_path("x.webp") == "image/webp"

    def test_unknown_raises(self):
        from annotator.core.client import _mime_from_path
        with pytest.raises(ValueError, match="unknown image extension"):
            _mime_from_path("foo.bmp")


class TestValidateVisionSupport:
    def test_accepts_claude_opus_4(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("claude-opus-4-6")

    def test_accepts_gemini_3(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("gemini-3.1-pro-preview")

    def test_accepts_gpt5(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("gpt-5.4")

    def test_accepts_gpt_4o(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("gpt-4o-mini")

    def test_rejects_old_text_only(self):
        from annotator.core.client import validate_vision_support
        with pytest.raises(ValueError, match="not in the vision-capable list"):
            validate_vision_support("llama-3")

    def test_case_insensitive(self):
        from annotator.core.client import validate_vision_support
        validate_vision_support("CLAUDE-OPUS-4-6")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_client.py -k "Mime or Vision" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the helpers to client.py**

In `annotator/core/client.py`, add near the top after the `MAX_OUTPUT_TOKENS` dict (~line 53):

```python
VISION_CAPABLE_PREFIXES = (
    "claude-opus-4", "claude-sonnet-4",
    "gemini-2", "gemini-3",
    "gpt-4o", "gpt-4.1", "gpt-5", "o4",
)


def validate_vision_support(model: str) -> None:
    """Raise ValueError if the model is not known to support vision input."""
    m = model.lower()
    if not any(m.startswith(p) for p in VISION_CAPABLE_PREFIXES):
        raise ValueError(
            f"Model '{model}' is not in the vision-capable list. "
            f"Vision-capable prefixes: {', '.join(VISION_CAPABLE_PREFIXES)}."
        )


_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _mime_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MIME_BY_EXT:
        raise ValueError(f"unknown image extension: {ext} (path: {path})")
    return _MIME_BY_EXT[ext]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client.py -k "Mime or Vision" -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add annotator/core/client.py tests/test_client.py
git commit -m "feat(client): add MIME inference and vision-model validation"
```

---

## Task 6: Per-provider image block builders

**Context:** Each provider expects a different shape. Isolate the shape-building in three pure functions, tested without hitting any API.

**Files:**
- Modify: `annotator/core/client.py`
- Modify: `tests/test_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_client.py`:

```python
class TestImageBlocks:
    def _mock_backend(self, monkeypatch, is_local):
        """Stub out the storage backend for deterministic tests."""
        import annotator.core.client as c

        def fake_read_bytes(path):
            return b"bytes-for-" + path.encode()

        def fake_get_presigned_url(path, expires_seconds=None):
            return f"https://example.com/{path}?sig=x"

        class FakeBE:
            pass

        fake_be = FakeBE()
        fake_be.read_bytes = fake_read_bytes
        fake_be.get_presigned_url = fake_get_presigned_url

        class FakeLocal: pass
        class FakeS3: pass
        import annotator.core.storage as s
        monkeypatch.setattr(s, "_get_backend", lambda: fake_be)
        monkeypatch.setattr(
            s, "LocalBackend",
            FakeLocal if not is_local else type(fake_be)
        )

    def test_anthropic_local_inlines_base64(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=True)
        from annotator.core.client import _build_image_blocks_anthropic
        blocks = _build_image_blocks_anthropic(["x/1.jpg"], use_url=False, enable_cache=False)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["type"] == "base64"
        assert blocks[0]["source"]["media_type"] == "image/jpeg"
        assert "data" in blocks[0]["source"]

    def test_anthropic_s3_uses_url(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=False)
        from annotator.core.client import _build_image_blocks_anthropic
        blocks = _build_image_blocks_anthropic(["x/1.jpg"], use_url=True, enable_cache=False)
        assert blocks[0]["source"]["type"] == "url"
        assert blocks[0]["source"]["url"].startswith("https://")

    def test_anthropic_cache_control(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=True)
        from annotator.core.client import _build_image_blocks_anthropic
        blocks = _build_image_blocks_anthropic(["x/1.jpg"], use_url=False, enable_cache=True)
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_openai_local_uses_data_url(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=True)
        from annotator.core.client import _build_image_blocks_openai
        blocks = _build_image_blocks_openai(["x/1.jpg"], use_url=False)
        assert blocks[0]["type"] == "image_url"
        assert blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_openai_s3_uses_https(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=False)
        from annotator.core.client import _build_image_blocks_openai
        blocks = _build_image_blocks_openai(["x/1.jpg"], use_url=True)
        assert blocks[0]["image_url"]["url"].startswith("https://")

    def test_gemini_always_inlines(self, monkeypatch):
        self._mock_backend(monkeypatch, is_local=False)  # even on S3
        from annotator.core.client import _build_image_blocks_gemini
        blocks = _build_image_blocks_gemini(["x/1.jpg"])
        assert "inline_data" in blocks[0]
        assert blocks[0]["inline_data"]["mime_type"] == "image/jpeg"
        assert "data" in blocks[0]["inline_data"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_client.py -k "ImageBlocks" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the block builders**

In `annotator/core/client.py`, add after the `_mime_from_path` helper:

```python
import base64


def _should_use_presigned_url() -> bool:
    """True when the storage backend is S3 (pre-signed URLs available)."""
    from . import storage
    be = storage._get_backend()
    return not isinstance(be, storage.LocalBackend)


def _base64_bytes(rel_path: str) -> str:
    from . import storage
    raw = storage._get_backend().read_bytes(rel_path)
    return base64.b64encode(raw).decode("ascii")


def _presigned_url(rel_path: str, expires_seconds: int = 172800) -> str:
    from . import storage
    return storage._get_backend().get_presigned_url(rel_path, expires_seconds=expires_seconds)


def _build_image_blocks_anthropic(
    image_paths: list[str], use_url: bool, enable_cache: bool,
) -> list[dict]:
    blocks = []
    for path in image_paths:
        media_type = _mime_from_path(path)
        if use_url:
            source = {"type": "url", "url": _presigned_url(path)}
        else:
            source = {
                "type": "base64",
                "media_type": media_type,
                "data": _base64_bytes(path),
            }
        block = {"type": "image", "source": source}
        if enable_cache:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def _build_image_blocks_openai(
    image_paths: list[str], use_url: bool,
) -> list[dict]:
    blocks = []
    for path in image_paths:
        if use_url:
            url = _presigned_url(path)
        else:
            b64 = _base64_bytes(path)
            url = f"data:{_mime_from_path(path)};base64,{b64}"
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


def _build_image_blocks_gemini(image_paths: list[str]) -> list[dict]:
    # Gemini does not accept S3 URIs; always inline.
    blocks = []
    for path in image_paths:
        blocks.append({
            "inline_data": {
                "mime_type": _mime_from_path(path),
                "data": _base64_bytes(path),
            }
        })
    return blocks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client.py -k "ImageBlocks" -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add annotator/core/client.py tests/test_client.py
git commit -m "feat(client): add per-provider image block builders"
```

---

## Task 7: `generate()` accepts images

**Files:**
- Modify: `annotator/core/client.py` — extend `generate`, `_generate_gemini`, `_generate_openai`, `_generate_anthropic`
- Modify: `tests/test_client.py` — test that generate routes images through correctly (no real API; mock at SDK boundary)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_client.py`:

```python
class TestGenerateWithImages:
    def test_anthropic_sends_image_blocks(self, monkeypatch):
        """generate() with images wraps them into Anthropic content blocks."""
        from annotator.core.client import ModelClient

        captured = {}

        class FakeResponse:
            class Usage:
                input_tokens = 1
                output_tokens = 1
            usage = Usage()
            content = [type("T", (), {"type": "text", "text": "ok"})()]

        class FakeAnthropic:
            class messages:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return FakeResponse()

        client = ModelClient.__new__(ModelClient)
        client.model = "claude-opus-4-6"
        client.provider = "anthropic"
        client._client = FakeAnthropic()

        # Stub image block builder to avoid storage calls
        import annotator.core.client as c
        monkeypatch.setattr(
            c, "_build_image_blocks_anthropic",
            lambda paths, use_url, enable_cache: [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "xx"}}
            ],
        )
        monkeypatch.setattr(c, "_should_use_presigned_url", lambda: False)

        resp = client.generate("hello", images=["foo.jpg"], json_mode=False)
        assert resp.text == "ok"
        content = captured["messages"][0]["content"]
        # Multimodal content is a list of blocks: text block + image block(s)
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "hello"
        assert content[1]["type"] == "image"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_client.py::TestGenerateWithImages -v`
Expected: FAIL because `generate()` doesn't accept `images`.

- [ ] **Step 3: Extend `generate` signature**

In `annotator/core/client.py`, replace the `generate` method signature and body with:

```python
    def generate(self, prompt: str,
                 images: list[str] | None = None,
                 json_mode: bool = True,
                 max_tokens: int = 0, timeout: int = 120,
                 thinking: bool = False,
                 thinking_budget: int = 0,
                 reasoning_effort: str = "",
                 enable_cache: bool = False) -> ModelResponse:
        if max_tokens <= 0:
            max_tokens = MAX_OUTPUT_TOKENS.get(self.provider, 8192)

        retry_cfg = get_retry_config()
        max_retries = retry_cfg.get("max_retries", 5)
        base_delay = retry_cfg.get("base_delay", 5)

        last_error = None
        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    return self._generate_gemini(prompt, json_mode, max_tokens, timeout,
                                                 thinking, thinking_budget, images)
                elif self.provider == "openai":
                    return self._generate_openai(prompt, json_mode, max_tokens, timeout,
                                                  thinking, thinking_budget,
                                                  reasoning_effort=reasoning_effort,
                                                  images=images)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(prompt, json_mode, max_tokens, timeout,
                                                     thinking, thinking_budget,
                                                     images=images,
                                                     enable_cache=enable_cache)
            except Exception as e:
                last_error = e
                delay = base_delay * (2 ** attempt)
                if attempt < max_retries - 1:
                    print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}. "
                          f"Retrying in {delay}s...", flush=True)
                    time.sleep(delay)
                else:
                    print(f"  API failed after {max_retries} attempts: {e}", flush=True)

        raise RuntimeError(
            f"API call failed after {max_retries} attempts: {last_error}"
        )
```

- [ ] **Step 4: Extend `_generate_anthropic`**

Replace the `_generate_anthropic` method:

```python
    def _generate_anthropic(self, prompt, json_mode, max_tokens, timeout,
                            thinking=False, thinking_budget=0,
                            images=None, enable_cache=False):
        """Anthropic API call via anthropic SDK."""
        system_parts = []
        if json_mode:
            system_parts.append(
                "You must respond with valid JSON only. "
                "Do not include markdown code fences, explanations, or any text "
                "outside the JSON object."
            )

        if images:
            content = [{"type": "text", "text": prompt}]
            content.extend(_build_image_blocks_anthropic(
                images, use_url=_should_use_presigned_url(), enable_cache=enable_cache,
            ))
        else:
            content = prompt

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "timeout": timeout,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)
        if thinking:
            budget = thinking_budget if thinking_budget > 0 else 16384
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        response = self._client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break
        if json_mode:
            text = _strip_json_fences(text)

        usage = {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
            "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
        }
        return ModelResponse(text=text, usage=usage)
```

- [ ] **Step 5: Extend `_generate_openai`**

Replace the `_generate_openai` method:

```python
    def _generate_openai(self, prompt, json_mode, max_tokens, timeout,
                         thinking=False, thinking_budget=0,
                         reasoning_effort: str = "", images=None):
        """OpenAI API call via openai SDK."""
        if images:
            content = [{"type": "text", "text": prompt}]
            content.extend(_build_image_blocks_openai(
                images, use_url=_should_use_presigned_url(),
            ))
        else:
            content = prompt

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": max_tokens,
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        response = self._client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens or 0,
            "output_tokens": response.usage.completion_tokens or 0,
            "total_tokens": response.usage.total_tokens or 0,
        }
        return ModelResponse(text=text, usage=usage)
```

- [ ] **Step 6: Extend `_generate_gemini`**

Replace the `_generate_gemini` method:

```python
    def _generate_gemini(self, prompt, json_mode, max_tokens, timeout,
                         thinking=False, thinking_budget=0, images=None):
        """Gemini API call via google-genai SDK."""
        config = {
            "max_output_tokens": max_tokens,
            "http_options": {"timeout": timeout * 1000},
        }
        if json_mode:
            config["response_mime_type"] = "application/json"
        if thinking:
            budget = thinking_budget if thinking_budget > 0 else 16384
            config["thinking_config"] = {"include_thoughts": True, "thinking_budget": budget}

        if images:
            contents = [{"role": "user",
                         "parts": [{"text": prompt}] + _build_image_blocks_gemini(images)}]
        else:
            contents = prompt

        response = self._client.models.generate_content(
            model=f"models/{self.model}",
            contents=contents,
            config=config,
        )

        text = response.text or ""
        usage_meta = response.usage_metadata
        usage = {
            "input_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
        }
        return ModelResponse(text=text, usage=usage)
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_client.py -v`
Expected: all pass. The existing text-only tests continue to work because `images` defaults to `None`.

- [ ] **Step 8: Commit**

```bash
git add annotator/core/client.py tests/test_client.py
git commit -m "feat(client): generate() accepts optional images kwarg"
```

---

## Task 8: Batch API — images in `build_batch_entry` and all three runners

**Files:**
- Modify: `annotator/core/client.py`
- Modify: `tests/test_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_client.py`:

```python
class TestBuildBatchEntryWithImages:
    def test_images_stored_in_request(self):
        from annotator.core.client import build_batch_entry
        entry = build_batch_entry("k", "p", images=["x/1.jpg", "x/2.jpg"])
        assert entry["request"]["images"] == ["x/1.jpg", "x/2.jpg"]

    def test_no_images_field_when_empty(self):
        from annotator.core.client import build_batch_entry
        entry = build_batch_entry("k", "p")
        assert "images" not in entry["request"]


class TestExtractEntryWithImages:
    def test_extract_returns_images(self):
        from annotator.core.client import build_batch_entry, _extract_entry
        entry = build_batch_entry("k", "p", images=["a.jpg"])
        key, prompt, json_mode, max_tokens, images = _extract_entry(entry)
        assert images == ["a.jpg"]

    def test_extract_empty_when_no_images(self):
        from annotator.core.client import build_batch_entry, _extract_entry
        entry = build_batch_entry("k", "p")
        _, _, _, _, images = _extract_entry(entry)
        assert images == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_client.py -k "BatchEntryWithImages or ExtractEntryWithImages" -v`
Expected: FAIL — `build_batch_entry` rejects `images` kwarg and `_extract_entry` returns a 4-tuple.

- [ ] **Step 3: Extend `build_batch_entry` and `_extract_entry`**

In `annotator/core/client.py`, replace `build_batch_entry` and `_extract_entry`:

```python
def build_batch_entry(key: str, prompt_text: str,
                      images: list[str] | None = None,
                      json_mode: bool = True,
                      max_tokens: int = 65536) -> dict:
    """Build a single batch entry from a key and prompt text.

    Uses a provider-neutral internal format. run_batch() and run_sync_entries()
    both consume these entries.
    """
    gen_config = {"max_output_tokens": max_tokens}
    if json_mode:
        gen_config["response_mime_type"] = "application/json"
    request = {
        "contents": [{
            "parts": [{"text": prompt_text}],
            "role": "user"
        }],
        "generation_config": gen_config,
    }
    if images:
        request["images"] = list(images)
    return {"key": key, "request": request}


def _extract_entry(entry: dict) -> tuple[str, str, bool, int, list[str]]:
    """Extract key, prompt, json_mode, max_tokens, images from a batch entry."""
    key = entry["key"]
    parts = entry["request"]["contents"][0]["parts"]
    prompt_text = parts[0]["text"]
    gen_config = entry["request"].get("generation_config", {})
    json_mode = "application/json" in gen_config.get("response_mime_type", "")
    max_tokens = gen_config.get("max_output_tokens", 0)
    images = entry["request"].get("images", [])
    return key, prompt_text, json_mode, max_tokens, images
```

- [ ] **Step 4: Update `run_sync_entries` to pass images through**

Replace the body of `run_sync_entries`:

```python
def run_sync_entries(client: 'ModelClient', entries: list[dict],
                     json_mode: bool = True, max_tokens: int = 0) -> dict:
    """Run entries synchronously one at a time.

    Returns {key: {text, usage}} dict (same shape as run_batch).
    """
    raw_entries = {}
    total = len(entries)
    for i, entry in enumerate(entries):
        key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
        if not entry_max_tokens:
            entry_max_tokens = max_tokens

        print(f"  [{i+1}/{total}] {key[:60]}...", flush=True)
        try:
            response = client.generate(
                prompt_text,
                images=images or None,
                json_mode=entry_json_mode if json_mode else False,
                max_tokens=entry_max_tokens,
            )
            raw_entries[key] = {
                "text": response.text,
                "usage": response.usage,
            }
        except Exception as e:
            print(f"  ERROR on {key}: {e}", flush=True)
            raw_entries[key] = {
                "text": "",
                "error": str(e),
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
    return raw_entries
```

- [ ] **Step 5: Update Gemini batch runner**

In `_run_batch_gemini`, the per-entry build currently writes `json.dumps(entry, ...)` directly. We need to transform entries that have images. Replace the body of `_run_batch_gemini` before the "Write Gemini-format JSONL" block to pre-transform entries:

Inside `_run_batch_gemini`, replace the JSONL-write block:

```python
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                      encoding="utf-8") as f:
        for entry in entries:
            key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
            parts = [{"text": prompt_text}]
            if images:
                parts.extend(_build_image_blocks_gemini(images))
            gem_entry = {
                "key": key,
                "request": {
                    "contents": [{"parts": parts, "role": "user"}],
                    "generation_config": entry["request"].get("generation_config", {}),
                },
            }
            f.write(json.dumps(gem_entry, ensure_ascii=False) + "\n")
        jsonl_path = f.name
```

- [ ] **Step 6: Update OpenAI batch runner**

In `_run_batch_openai`, replace the per-entry body building block with:

```python
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                      encoding="utf-8") as f:
        for entry in entries:
            key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
            if not entry_max_tokens or entry_max_tokens > max_tokens:
                entry_max_tokens = max_tokens

            if images:
                content = [{"type": "text", "text": prompt_text}]
                content.extend(_build_image_blocks_openai(
                    images, use_url=_should_use_presigned_url(),
                ))
            else:
                content = prompt_text

            body = {
                "model": client.model,
                "messages": [{"role": "user", "content": content}],
                "max_completion_tokens": entry_max_tokens,
            }
            if json_mode and entry_json_mode:
                body["response_format"] = {"type": "json_object"}
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
            line = {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        jsonl_path = f.name
```

- [ ] **Step 7: Update Anthropic batch runner**

`_run_batch_anthropic` needs two things: accept an `enable_cache` flag, and wrap prompts + images into content blocks. Change the runner signature and the body. Replace the signature:

```python
def _run_batch_anthropic(client, entries, json_mode, display_name, poll_interval,
                         thinking=False, thinking_budget=0, enable_cache=False):
```

And inside the per-entry build loop, replace the `messages` construction:

```python
    id_to_key = {}
    requests = []
    for i, entry in enumerate(entries):
        key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(entry)
        if not entry_max_tokens or entry_max_tokens > max_tokens:
            entry_max_tokens = max_tokens

        short_id = f"r{i}"
        id_to_key[short_id] = key

        if images:
            content = [{"type": "text", "text": prompt_text}]
            content.extend(_build_image_blocks_anthropic(
                images, use_url=_should_use_presigned_url(), enable_cache=enable_cache,
            ))
        else:
            content = prompt_text

        params = {
            "model": client.model,
            "max_tokens": entry_max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if json_mode and entry_json_mode:
            params["system"] = (
                "You must respond with valid JSON only. "
                "Do not include markdown code fences, explanations, or any text "
                "outside the JSON object."
            )
        if thinking:
            budget = thinking_budget if thinking_budget > 0 else 16384
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}

        requests.append(Request(
            custom_id=short_id,
            params=MessageCreateParamsNonStreaming(**params),
        ))
```

- [ ] **Step 8: Thread `enable_cache` through `run_batch`**

Replace `run_batch`:

```python
def run_batch(client: 'ModelClient', entries: list[dict],
              json_mode: bool = True, display_name: str = "batch",
              poll_interval: int = 60,
              thinking: bool = False, thinking_budget: int = 0,
              reasoning_effort: str = "",
              enable_cache: bool = False) -> dict:
    """Run entries as a batch job via the provider's batch API."""
    provider = client.provider
    print(f"Running batch ({provider}): {len(entries)} entries, display_name={display_name}")

    if provider == "gemini":
        return _run_batch_gemini(client, entries, json_mode, display_name, poll_interval,
                                thinking, thinking_budget)
    elif provider == "openai":
        return _run_batch_openai(client, entries, json_mode, display_name, poll_interval,
                                thinking, thinking_budget, reasoning_effort)
    elif provider == "anthropic":
        return _run_batch_anthropic(client, entries, json_mode, display_name, poll_interval,
                                   thinking, thinking_budget, enable_cache=enable_cache)
    else:
        raise ValueError(f"Batch API not supported for provider: {provider}")
```

- [ ] **Step 9: Run all client tests**

Run: `pytest tests/test_client.py -v`
Expected: all pass (including pre-existing ones).

- [ ] **Step 10: Commit**

```bash
git add annotator/core/client.py tests/test_client.py
git commit -m "feat(client): plumb images and enable_cache through batch API"
```

---

## Task 9: `format_transcript` and `format_excerpt` render screenshot markers

**Files:**
- Modify: `annotator/core/utils.py`
- Modify: `tests/test_utils.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_utils.py`:

```python
class TestFormatTranscriptWithScreenshots:
    def _conv(self):
        return {
            "conversation_id": "c1",
            "turns": [
                {"turn_number": 1, "role": "TUTOR", "text": "hi", "type": "DIALOGUE"},
                {"turn_number": 2, "role": "TUTOR", "text": "look here", "type": "DIALOGUE"},
                {"turn_number": 3, "role": "STUDENT", "text": "ok", "type": "DIALOGUE"},
            ],
        }

    def test_marker_rendered_after_anchor_turn(self):
        from annotator.core.utils import format_transcript
        ss = [{"anchor_turn": 2, "filename": "5.0.jpg", "timestamp_seconds": 5.0}]
        out = format_transcript(self._conv(), screenshots=ss)
        lines = out.split("\n")
        assert lines[0] == "Turn 1. TUTOR: hi"
        assert lines[1] == "Turn 2. TUTOR: look here"
        assert lines[2] == "  [SCREEN @ turn 2: image 1]"
        assert lines[3] == "Turn 3. STUDENT: ok"

    def test_multiple_images_numbered_positionally(self):
        from annotator.core.utils import format_transcript
        ss = [
            {"anchor_turn": 1, "filename": "0.5.jpg", "timestamp_seconds": 0.5},
            {"anchor_turn": 2, "filename": "5.0.jpg", "timestamp_seconds": 5.0},
        ]
        out = format_transcript(self._conv(), screenshots=ss)
        assert "[SCREEN @ turn 1: image 1]" in out
        assert "[SCREEN @ turn 2: image 2]" in out

    def test_no_screenshots_unchanged_output(self):
        from annotator.core.utils import format_transcript
        conv = self._conv()
        assert format_transcript(conv) == format_transcript(conv, screenshots=None)
        assert format_transcript(conv) == format_transcript(conv, screenshots=[])


class TestFormatExcerptWithScreenshots:
    def _conv(self):
        turns = [
            {"turn_number": n, "role": "TUTOR" if n % 2 else "STUDENT",
             "text": f"t{n}", "type": "DIALOGUE"}
            for n in range(1, 11)
        ]
        return {"conversation_id": "c1", "turns": turns}

    def test_screenshot_outside_excerpt_omitted(self):
        from annotator.core.utils import format_excerpt
        ss_all = [
            {"anchor_turn": 1, "filename": "a.jpg", "timestamp_seconds": 1.0},
            {"anchor_turn": 5, "filename": "b.jpg", "timestamp_seconds": 5.0},
            {"anchor_turn": 9, "filename": "c.jpg", "timestamp_seconds": 9.0},
        ]
        # The caller pre-filters to what's in window; format_excerpt renders markers
        # for the ones passed in.
        filtered = [s for s in ss_all if 4 <= s["anchor_turn"] <= 6]
        out = format_excerpt(self._conv(), turn_start=5, turn_end=5,
                             context_before=1, context_after=1,
                             screenshots=filtered)
        assert "[SCREEN @ turn 5: image 1]" in out
        assert "a.jpg" not in out
        assert "c.jpg" not in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_utils.py -k "Screenshots" -v`
Expected: FAIL — unknown kwarg `screenshots`.

- [ ] **Step 3: Extend `format_transcript`**

In `annotator/core/utils.py`, replace `format_transcript`:

```python
def format_transcript(conversation: dict, dialogue_only: bool = False,
                      screenshots: list[dict] | None = None) -> str:
    """Format conversation turns as: Turn {n}. {ROLE}: {text}

    If `screenshots` is provided, inline a marker '  [SCREEN @ turn N: image K]'
    after each anchor turn. K is the 1-based index of the screenshot in the list.
    """
    ss_by_turn: dict[int, list[int]] = {}
    if screenshots:
        for idx, s in enumerate(screenshots, start=1):
            ss_by_turn.setdefault(s["anchor_turn"], []).append(idx)

    lines = []
    for turn in _filter_turns(conversation["turns"], dialogue_only):
        n = turn["turn_number"]
        role = turn["role"]
        text = turn["text"]
        lines.append(f"Turn {n}. {role}: {text}")
        for idx in ss_by_turn.get(n, []):
            lines.append(f"  [SCREEN @ turn {n}: image {idx}]")
    return "\n".join(lines)
```

- [ ] **Step 4: Extend `format_excerpt`**

Replace `format_excerpt`:

```python
def format_excerpt(conversation: dict, turn_start: int, turn_end: int,
                   context_before: int = 20, context_after: int = 20,
                   dialogue_only: bool = False,
                   screenshots: list[dict] | None = None) -> str:
    """Extract a transcript excerpt around a detected moment, with context.

    If `screenshots` is provided, inline '  [SCREEN @ turn N: image K]' markers
    for screenshots whose anchor_turn falls inside the rendered excerpt range.
    Screenshots are numbered by their position in the passed list (K=1..len).
    """
    turns = _filter_turns(conversation["turns"], dialogue_only)
    if not turns:
        return ""

    all_turn_nums = [t["turn_number"] for t in turns]
    min_turn = min(all_turn_nums)
    max_turn = max(all_turn_nums)

    excerpt_start = max(min_turn, turn_start - context_before)
    excerpt_end = min(max_turn, turn_end + context_after)

    ss_by_turn: dict[int, list[int]] = {}
    if screenshots:
        for idx, s in enumerate(screenshots, start=1):
            if excerpt_start <= s["anchor_turn"] <= excerpt_end:
                ss_by_turn.setdefault(s["anchor_turn"], []).append(idx)

    lines = []
    if excerpt_start > min_turn:
        lines.append(f"[... turns 1-{excerpt_start - 1} omitted ...]")
        lines.append("")

    for turn in turns:
        n = turn["turn_number"]
        if n < excerpt_start or n > excerpt_end:
            continue
        text = turn["text"][:200]
        marker = " <<<" if turn_start <= n <= turn_end else ""
        lines.append(f"  Turn {n}. {turn['role']}: {text}{marker}")
        for idx in ss_by_turn.get(n, []):
            lines.append(f"  [SCREEN @ turn {n}: image {idx}]")
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_utils.py -v`
Expected: all pass (including pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add annotator/core/utils.py tests/test_utils.py
git commit -m "feat(utils): render screenshot markers in format_transcript and format_excerpt"
```

---

## Task 10: Detection integration — `--with-screenshots`

**Files:**
- Modify: `annotator/core/detect.py`
- Modify: `tests/test_detect_parse.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detect_parse.py`:

```python
class TestBuildDetectionEntriesWithScreenshots:
    def test_includes_images_when_flag_set(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.detect import build_detection_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        # Stub prompt loader to return a predictable template
        import annotator.core.detect as d
        monkeypatch.setattr(d, "load_prompt", lambda v, t: "PROMPT: {transcript}")

        entries = build_detection_entries(
            [conv], targets=["scaffolding"], version="v5",
            with_screenshots=True,
        )
        assert len(entries) == 1
        # 4.000.jpg usable; 11.500.jpg filtered (eedi_ip=True)
        assert entries[0]["request"]["images"] == [
            "deidentified/screenshots/099bf759-abcd/4.000.jpg"
        ]
        # Text marker appears in the prompt
        assert "[SCREEN @ turn 1: image 1]" in entries[0]["request"]["contents"][0]["parts"][0]["text"]

    def test_no_images_when_flag_off(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.detect import build_detection_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.detect as d
        monkeypatch.setattr(d, "load_prompt", lambda v, t: "PROMPT: {transcript}")

        entries = build_detection_entries(
            [conv], targets=["scaffolding"], version="v5",
            with_screenshots=False,
        )
        assert "images" not in entries[0]["request"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_detect_parse.py -k "Screenshots" -v`
Expected: FAIL — `with_screenshots` kwarg unknown.

- [ ] **Step 3: Extend `build_detection_entries`**

In `annotator/core/detect.py`, replace `build_detection_entries`:

```python
def build_detection_entries(conversations: list[dict], targets: list[str],
                            version: str, dialogue_only: bool = False,
                            with_screenshots: bool = False) -> list[dict]:
    """Build batch entries for detection."""
    from .screenshots import load_anchored_screenshots

    prompt_cache = {}
    entries = []

    for conv in conversations:
        conv_id = conv["conversation_id"]

        screenshots = (
            load_anchored_screenshots(conv_id, conv["turns"])
            if with_screenshots else []
        )
        image_paths = [s["storage_path"] for s in screenshots]

        transcript_text = format_transcript(
            conv, dialogue_only=dialogue_only,
            screenshots=screenshots if screenshots else None,
        )

        for target in targets:
            if target not in prompt_cache:
                prompt_cache[target] = load_prompt(version, target)

            prompt = prompt_cache[target].replace("{transcript}", transcript_text)
            key = f"{conv_id}__{target}"
            entries.append(build_batch_entry(
                key, prompt, images=image_paths or None,
            ))

    return entries
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_detect_parse.py -v`
Expected: all pass.

- [ ] **Step 5: Extend `run_detect` and CLI**

In `annotator/core/detect.py`, replace the `run_detect` signature:

```python
def run_detect(version: str, model: str, mode: str, prompt_version: str,
               targets: list[str], phase_cfg: dict,
               test: int = 0, dialogue_only: bool = False,
               with_screenshots: bool = False) -> dict:
```

Inside `run_detect`, after `client = ModelClient(model)`:

```python
    if with_screenshots:
        from .client import validate_vision_support
        validate_vision_support(model)
        print("Screenshots: enabled -- vision model validated")
```

Change the `build_detection_entries` call to pass `with_screenshots=with_screenshots`.

Count images sent across entries:
```python
    total_images_sent = sum(len(e["request"].get("images", [])) for e in entries)
    convs_with_images = sum(
        1 for e in entries if e["request"].get("images")
    )
```

Add these and the flag to the output dict:
```python
    output = {
        ...,
        "with_screenshots": with_screenshots,
        "convs_with_images": convs_with_images,
        "total_images_sent": total_images_sent,
        ...
    }
```

And thread `images_seen` into each conv's entry in `parse_detection_results`. Replace `parse_detection_results` signature:

```python
def parse_detection_results(raw_entries: dict,
                            images_per_key: dict[str, int] | None = None) -> dict[str, dict]:
```

Add, at the start:
```python
    images_per_key = images_per_key or {}
```

And where we create the per-conv entry:
```python
        if conv_id not in detections_by_conv:
            detections_by_conv[conv_id] = {
                "detections": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "images_seen": 0,
            }
        # accumulate max images seen for this conv across targets (same images per target)
        detections_by_conv[conv_id]["images_seen"] = max(
            detections_by_conv[conv_id]["images_seen"],
            images_per_key.get(key, 0),
        )
```

In `run_detect`, build `images_per_key` from entries before calling parse:
```python
    images_per_key = {
        e["key"]: len(e["request"].get("images", []))
        for e in entries
    }
    detections_by_conv = parse_detection_results(raw, images_per_key=images_per_key)
```

Add the CLI flag:
```python
    parser.add_argument("--with-screenshots", action="store_true",
                        help="Include anchored screenshots in detection prompts. "
                             "Requires a vision-capable model.")
```

And pass through to `run_detect(..., with_screenshots=args.with_screenshots)`.

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add annotator/core/detect.py tests/test_detect_parse.py
git commit -m "feat(detect): add --with-screenshots flag"
```

---

## Task 11: Annotation integration — `--with-screenshots`

**Files:**
- Modify: `annotator/core/annotate.py`
- Modify: `tests/test_detect_parse.py` (or create a new `tests/test_annotate_build.py`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_annotate_build.py`:

```python
"""Tests for annotator.core.annotate entry building with screenshots."""
import pytest


class TestBuildAnalysisEntriesWithScreenshots:
    def _detections_for(self, conv_id):
        return {
            conv_id: {
                "detections": [
                    {"turn_start": 1, "turn_end": 1, "annotation_type": "scaffolding",
                     "brief_description": "moment at turn 1"},
                    {"turn_start": 3, "turn_end": 3, "annotation_type": "scaffolding",
                     "brief_description": "moment at turn 3"},
                ],
                "usage": {},
            }
        }

    def test_per_moment_image_filtering(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.annotate import build_analysis_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.annotate as a
        monkeypatch.setattr(
            a, "load_prompt",
            lambda v, t: "P {brief_description} X {excerpt} X {turn_start} X {turn_end}",
        )

        # Narrow context window so only one moment's excerpt includes turn 1
        entries = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=1, version="v4",
            with_screenshots=True,
        )
        # Only the 4.000.jpg image anchors to turn 1 (usable one).
        # Moment at turn 1 (window 0..2): image in scope.
        # Moment at turn 3 (window 2..4): image NOT in scope.
        e0 = entries[0]
        e1 = entries[1]
        assert e0["request"]["images"] == [
            "deidentified/screenshots/099bf759-abcd/4.000.jpg"
        ]
        assert "images" not in e1["request"]

    def test_no_images_when_flag_off(self, local_storage, monkeypatch):
        from annotator.core.storage import load_transcript
        from annotator.core.annotate import build_analysis_entries

        conv_id = "2024-t1_2024-s1_099bf759-abcd"
        conv = load_transcript(conv_id)

        import annotator.core.annotate as a
        monkeypatch.setattr(
            a, "load_prompt",
            lambda v, t: "P {brief_description} X {excerpt} X {turn_start} X {turn_end}",
        )

        entries = build_analysis_entries(
            self._detections_for(conv_id), {conv_id: conv},
            context_window=20, version="v4",
            with_screenshots=False,
        )
        for e in entries:
            assert "images" not in e["request"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_annotate_build.py -v`
Expected: FAIL — `with_screenshots` kwarg unknown.

- [ ] **Step 3: Extend `build_analysis_entries`**

In `annotator/core/annotate.py`, replace the signature and body of `build_analysis_entries`:

```python
def build_analysis_entries(detections_by_conv: dict, conversations_map: dict,
                           context_window: int, version: str,
                           dialogue_only: bool = False,
                           annotator_style: str | None = None,
                           with_screenshots: bool = False) -> list[dict]:
    """Build batch entries for analysis.

    When with_screenshots=True, attaches per-moment images whose anchor turn
    falls inside the excerpt window (excerpt_start <= anchor_turn <= excerpt_end,
    inclusive).
    """
    from .screenshots import load_anchored_screenshots

    prompt_cache = {}
    entries = []

    for conv_id, conv_data in detections_by_conv.items():
        conversation = conversations_map.get(conv_id)
        if not conversation:
            print(f"WARNING: No transcript found for {conv_id}, skipping")
            continue

        all_screenshots = (
            load_anchored_screenshots(conv_id, conversation["turns"])
            if with_screenshots else []
        )

        turns = conversation.get("turns", [])
        min_turn = turns[0]["turn_number"] if turns else 1
        max_turn = turns[-1]["turn_number"] if turns else 1

        for idx, det in enumerate(conv_data.get("detections", [])):
            ann_type = det.get("annotation_type", "scaffolding")
            if ann_type not in VALID_ANNOTATION_TYPES:
                ann_type = "scaffolding"

            turn_start = det.get("turn_start", 0)
            turn_end = det.get("turn_end", turn_start)
            brief_desc = det.get("brief_description", "")

            excerpt_start = max(min_turn, turn_start - context_window)
            excerpt_end = min(max_turn, turn_end + context_window)
            in_scope = [
                s for s in all_screenshots
                if excerpt_start <= s["anchor_turn"] <= excerpt_end
            ]
            image_paths = [s["storage_path"] for s in in_scope]

            if ann_type not in prompt_cache:
                prompt_cache[ann_type] = load_prompt(version, ann_type)

            excerpt = format_excerpt(
                conversation, turn_start, turn_end,
                context_before=context_window, context_after=context_window,
                dialogue_only=dialogue_only,
                screenshots=in_scope if in_scope else None,
            )

            prompt = prompt_cache[ann_type]
            prompt = prompt.replace("{annotator_style}", "")
            prompt = prompt.replace("{brief_description}", brief_desc)
            prompt = prompt.replace("{excerpt}", excerpt)
            prompt = prompt.replace("{turn_start}", str(turn_start))
            prompt = prompt.replace("{turn_end}", str(turn_end))

            key = f"{conv_id}__{ann_type}__{idx}"
            entries.append(build_batch_entry(
                key, prompt, images=image_paths or None,
            ))

    return entries
```

Add `from .utils import format_excerpt` if not already imported.

- [ ] **Step 4: Thread flag through `run_annotate` and CLI**

Replace the `run_annotate` signature:

```python
def run_annotate(version: str, model: str, mode: str, prompt_version: str,
                 targets: list[str], phase_cfg: dict,
                 dialogue_only: bool = False, context_window: int = 20,
                 gold: bool = False, annotator_style: str | None = None,
                 detections_by_conv: dict | None = None,
                 with_screenshots: bool = False) -> dict:
```

Inside `run_annotate`, after `client = ModelClient(model)`:

```python
    if with_screenshots:
        from .client import validate_vision_support
        validate_vision_support(model)
        print("Screenshots: enabled -- vision model validated, caching ON")
```

Pass `with_screenshots=with_screenshots` into `build_analysis_entries`.

Change the batch call to pass `enable_cache=with_screenshots`:

```python
    if mode == "batch":
        poll_interval = phase_cfg["poll_interval"]
        raw = run_batch(client, entries, display_name="annotate",
                        poll_interval=poll_interval,
                        thinking=phase_cfg.get("thinking", False),
                        thinking_budget=phase_cfg.get("thinking_budget", 0),
                        reasoning_effort=phase_cfg.get("reasoning_effort", ""),
                        enable_cache=with_screenshots)
```

After `results = parse_and_merge(raw, detections_by_conv)`, add roll-ups:

```python
    images_per_key = {
        e["key"]: len(e["request"].get("images", []))
        for e in entries
    }
    for conv_id, conv_result in results.items():
        for i, ann in enumerate(conv_result["annotations"]):
            ann_type = ann.get("annotation_type", "scaffolding")
            k = f"{conv_id}__{ann_type}__{i}"
            ann["images_seen"] = images_per_key.get(k, 0)
        conv_result["images_seen"] = sum(a.get("images_seen", 0) for a in conv_result["annotations"])

    total_images_sent = sum(images_per_key.values())
    convs_with_images = sum(1 for r in results.values() if r.get("images_seen", 0) > 0)
    annotations_with_images = sum(
        1 for r in results.values()
        for a in r["annotations"]
        if a.get("images_seen", 0) > 0
    )
```

Add these to the output dict:

```python
    output = {
        ...,
        "with_screenshots": with_screenshots,
        "convs_with_images": convs_with_images,
        "annotations_with_images": annotations_with_images,
        "total_images_sent": total_images_sent,
        ...
    }
```

Add the CLI flag:

```python
    parser.add_argument("--with-screenshots", action="store_true",
                        help="Include anchored screenshots from each moment's "
                             "context window. Requires a vision-capable model.")
```

Pass through to `run_annotate(..., with_screenshots=args.with_screenshots)`.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add annotator/core/annotate.py tests/test_annotate_build.py
git commit -m "feat(annotate): add --with-screenshots flag with per-moment filtering"
```

---

## Task 12: Smoke test procedure documentation

**Files:**
- Create: `docs/screenshot_enrichment_smoke.md`

- [ ] **Step 1: Write the smoke-test doc**

Create `docs/screenshot_enrichment_smoke.md`:

```markdown
# Screenshot Enrichment — Smoke Test

Verifies the end-to-end path for `--with-screenshots` against real S3 data
before running at scale.

## Prerequisites

- AWS credentials for the `kylel-alexisr-edu` bucket
- `.env` with `STORAGE_BACKEND=s3`, `S3_BUCKET=kylel-alexisr-edu`,
  `S3_PREFIX=""`, `STORAGE_SCREENSHOTS=deidentified/screenshots`, and a valid
  `ANTHROPIC_API_KEY` (or whichever profile you want)

## Target conversation

`099bf759-2426-549b-8dff-ad3f4be80db2` (verified it has screenshots on S3
as of the design date; pick a different conv if screenshots have moved).

## Detection smoke

```
python -m annotator.core.detect \
  --profile anthropic \
  --with-screenshots \
  --test 1 \
  --version smoke_screenshots
```

Expected:
- Console prints `Screenshots: enabled -- vision model validated`
- Output `detections.json` has `with_screenshots: true`, `convs_with_images >= 1`,
  `total_images_sent > 0`
- `detect_requests.jsonl` entries contain an `"images"` array

## Annotation smoke

After detection completes:

```
python -m annotator.core.annotate \
  --profile anthropic \
  --with-screenshots \
  --version smoke_screenshots
```

Expected:
- Console prints `Screenshots: enabled -- vision model validated, caching ON`
- `annotations.json` has `with_screenshots: true`, `convs_with_images >= 1`,
  `annotations_with_images > 0`

## Cleanup

Results sit under `results/annotator/smoke_screenshots/` — delete when done.
```

- [ ] **Step 2: Commit**

```bash
git add docs/screenshot_enrichment_smoke.md
git commit -m "docs: add screenshot enrichment smoke test procedure"
```

---

## Self-Review Results

**Spec coverage:** Every section of the spec maps to tasks —
- §1 Storage → Tasks 1, 3
- §2 Anchoring → Tasks 2, 4
- §3 Client multimodal → Tasks 5, 6, 7, 8
- §4 Marker convention → Task 9
- §5 Detection → Task 10
- §6 Annotation → Task 11
- §7 Stays the same → protected by default-off behavior; existing tests cover regression
- §8 Testing → every task has TDD tests; Task 12 covers manual smoke

**Placeholder scan:** No TBDs, no "implement appropriate error handling," no code-free steps.

**Type consistency:** `load_anchored_screenshots` returns `list[dict]` with keys `filename / timestamp_seconds / anchor_turn / storage_path` in Tasks 4, 10, 11. `build_batch_entry` takes `images: list[str] | None` in Tasks 8, 10, 11. `_extract_entry` returns a 5-tuple starting in Task 8 and is consumed as a 5-tuple in Task 8's runner updates.
