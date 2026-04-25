# Screenshot Enrichment Design

## Problem

Transcripts are text-only today. Interleaved `SCREEN_UPDATE` / `SCREEN_INTERACTION` / `BOARD_UPDATE` turns either carry text narration (Eedi, some Stepup) or are bare `[SCREEN UPDATE]` placeholders with no content (many Stepup). Real screenshots, captured by ffmpeg from session videos at LLM-chosen timestamps, already exist on S3 under `deidentified/screenshots/{uuid}/{timestamp_seconds}.jpg`, with a sibling `_metadata.json` recording de-identification verification and Eedi-IP flags.

The annotator pipeline (detection + annotation) cannot see these images. When a tutor says "look at this" or a student reacts to something visual, the pipeline is judging pedagogy with incomplete context.

Benchmark tutor, benchmark student, and labeller are explicitly out of scope for this iteration — they stay text-only. The design must make adding them later a caller-side change, not a client-layer rearchitecture.

## What Doesn't Need Fixing

- **Upstream capture pipeline.** Screenshots land on S3 via an existing pipeline. We are a consumer.
- **Verification / IP flagging.** `_metadata.json` is authoritative for `flagged` and `eedi_ip`. We read, we filter, we do not write.
- **Detection prompt files.** v4/v5 detection and annotation prompts don't mention screenshots and don't need to. The inline text marker format `[SCREEN @ turn N: image K]` is self-describing; iterate prompts later only if image-aware runs underperform.
- **Existing results.** `v5_gold/` and other runs stay valid. This change is additive.

## Design

### 1. Storage

**Binary files — existing upstream layout, unchanged:**

```
s3://kylel-alexisr-edu/deidentified/screenshots/{uuid}/{timestamp_seconds}.jpg
s3://kylel-alexisr-edu/deidentified/screenshots/{uuid}/_metadata.json
```

Filename format: `{seconds}.jpg`, e.g. `603.834.jpg`. `os.path.splitext` correctly splits at the last dot. Timestamp parsing raises on unparseable names rather than silently zero-ing.

**Config addition — one new path category:**

```yaml
storage:
  paths:
    screenshots: deidentified/screenshots
```

Plus `STORAGE_SCREENSHOTS` in `.env.example`.

**StorageBackend ABC — new binary + URI methods:**

```python
class StorageBackend(ABC):
    # existing read_json / write_json / list_files / exists / get_local_path unchanged
    @abstractmethod
    def read_bytes(self, rel_path: str) -> bytes: ...
    @abstractmethod
    def write_bytes(self, rel_path: str, data: bytes) -> None: ...
    @abstractmethod
    def get_presigned_url(self, rel_path: str, expires_seconds: int = 172800) -> str: ...
```

- `LocalBackend.get_presigned_url` returns `file://` URI for parity.
- `S3Backend.get_presigned_url` uses `boto3.generate_presigned_url`. 48h expiry — longer than the 24h batch completion window.

**ID translation:** full `conv_id` (`2024-tN_2024-sN_uuid`) is the pipeline's internal key; S3 screenshots key on UUID only. Translation happens inside the storage helpers, callers stay in `conv_id`.

### 2. Anchoring screenshots to turns

Deterministic, no LLM pass:
- Parse timestamp from filename (`float(os.path.splitext(name)[0])`)
- Anchor each screenshot to the **latest turn whose `start_seconds` ≤ screenshot timestamp**
- If before any turn, anchor to turn 1

**Turn-timestamp normalization:** add `start_seconds: float` to every turn dict at transcript load time.
- JSONL-sourced transcripts already carry `start_seconds` upstream; stop converting it to a string.
- Consolidated-JSON transcripts parse `"MM:SS-MM:SS"` once at load time.
The existing `timestamp` string field stays alongside, untouched.

**New module:** `annotator/core/screenshots.py`
```python
def timestamp_seconds_from_filename(filename: str) -> float
def anchor_screenshots(filenames: list[str], turns: list[dict]) -> list[dict]
def load_anchored_screenshots(conv_id: str, turns: list[dict]) -> list[dict]
```

`load_anchored_screenshots` composes `list_screenshots` + `load_screenshot_verification` + `anchor_screenshots`, filters `flagged: true` or `eedi_ip: true`, and returns entries sorted by `anchor_turn`:

```python
[{"filename": "603.834.jpg",
  "storage_path": "deidentified/screenshots/099bf759.../603.834.jpg",
  "timestamp_seconds": 603.834,
  "anchor_turn": 45}, ...]
```

**New storage helpers** (in `storage.py`):
```python
def list_screenshots(conv_id: str) -> list[str]
def load_screenshot_bytes(conv_id: str, filename: str) -> bytes
def get_screenshot_uri(conv_id: str, filename: str) -> str
def load_screenshot_verification(conv_id: str) -> dict
```

### 3. Generic multimodal capability on ModelClient

Minimal surface: one new optional kwarg on `generate()` and `build_batch_entry()`:

```python
def generate(
    self, prompt: str, images: list[str] | None = None,
    json_mode=True, max_tokens=0, timeout=120,
    thinking=False, thinking_budget=0, reasoning_effort="",
    enable_cache: bool = False,  # Anthropic ephemeral image caching
) -> ModelResponse
```

Images are storage paths relative to the backend. `ModelClient` resolves per provider:

| Backend | Gemini | OpenAI | Anthropic |
|---------|--------|--------|-----------|
| Local   | base64 inline | base64 data URL | base64 inline |
| S3      | base64 inline (S3 URIs unsupported) | pre-signed HTTPS URL | pre-signed HTTPS URL |

Callers never encode base64 or format provider blocks. The client does it.

**Batch:** all three runners (`_run_batch_gemini`, `_run_batch_openai`, `_run_batch_anthropic`) gain image handling where they currently build request bodies. The provider-neutral batch-entry format grows one optional field:

```python
{"key": ..., "request": {"contents": [{"parts": [{"text": ...}], "role": "user"},
                                       {"images": ["path/to/1.jpg", ...]}]}}
```

`build_batch_entry(key, prompt_text, images=None, ...)` sets the second entry when images are passed.

**Prompt caching (Anthropic only):** when `enable_cache=True`, wrap each image block with `cache_control: {"type": "ephemeral"}`. No-op for Gemini and OpenAI — they use implicit prefix-matching caching. Default off for detection (unique per conv), default on for annotation-with-screenshots (images repeat across overlapping excerpt windows in the same conv).

**Vision-capable model validation:** `validate_vision_support(model)` in `client.py`, called from detection and annotation CLIs when `--with-screenshots` is set. Fails fast at startup.

```python
VISION_CAPABLE_PREFIXES = (
    "claude-opus-4", "claude-sonnet-4",
    "gemini-2", "gemini-3",
    "gpt-4o", "gpt-4.1", "gpt-5", "o4",
)
```

**MIME inference:** one-line helper from extension. `.jpg`/`.jpeg` → `image/jpeg`, `.png` → `image/png`, `.webp` → `image/webp`. Raises on unknown.

### 4. In-prompt marker convention + interleaved content blocks

Rendered inline by `format_transcript` / `format_excerpt` when `screenshots=` is passed:

```
Turn 44. STUDENT: ok
Turn 45. TUTOR: Let me show you this one
  [SCREEN @ turn 45: image 1]
Turn 46. STUDENT: ...
Turn 52. TUTOR: now try this one
  [SCREEN @ turn 52: image 2]
```

Images are passed ordered by `anchor_turn` ascending; the index `K` in `image K` is the 1-based position in the images list.

**Content interleaving.** `ModelClient` does NOT pile all image blocks at the end of the content array. `_interleave_text_and_images` splits the prompt at each marker line and inserts the matching image block immediately after the marker. Anthropic / OpenAI / Gemini all receive content of the form:

```
[text up to and including marker 1] [image 1] [text up to marker 2] [image 2] ... [trailing text]
```

This gives the model spatial proximity between marker and image — it doesn't have to remember "image 5 was at turn 87" because image 5 *lives* at turn 87 in the content stream. The marker text remains visible so the explicit "turn N" label is still right next to the image.

Edge cases:
- Marker referencing an out-of-range image index → marker stays as text, no image inserted
- Image not referenced by any marker → appended at the end (no silent drops)
- Future enrichment of marker (e.g., `[SCREEN @ turn 45, t=603.8s: image 1]`) — regex permits arbitrary metadata between `SCREEN` and `image N]`

### 5. Detection integration (`annotator/core/detect.py`)

- `format_transcript(conv, dialogue_only=False, screenshots=None)` — new `screenshots` param.
- `build_detection_entries(..., with_screenshots: bool = False)` — when True, loads anchored screenshots per conv, attaches to every entry.
- CLI: `--with-screenshots` flag, default off. Calls `validate_vision_support(model)` at startup when set.
- Output JSON gains `with_screenshots: bool`, `convs_with_images: int`, `total_images_sent: int`; each conv's entry gains `images_seen: int`.

### 6. Annotation integration (`annotator/core/annotate.py`)

- `format_excerpt(..., screenshots=None)` — same convention as detection, markers only rendered for anchors inside the excerpt range.
- `build_analysis_entries(..., with_screenshots: bool = False)` — loads all conv screenshots once, filters per moment to `excerpt_start ≤ anchor_turn ≤ excerpt_end` (inclusive both ends).
- CLI: `--with-screenshots` flag, default off.
- `enable_cache=True` passed to batch runners when `--with-screenshots` is set — exploits repeated images across overlapping windows.
- Output JSON gains `with_screenshots`, `convs_with_images`, `annotations_with_images`, `total_images_sent`; each annotation gains `images_seen`.

### 7. What stays exactly the same

- Benchmark tutor, student, labeller: text-only. Capability is available on the client; they simply don't pass `images=`.
- Detection / annotation without the flag: byte-for-byte identical request bodies and outputs to today.
- All existing prompt files.
- `view.py` and other viewer tooling.
- `data/transcripts/` JSON schema on disk.

### 8. Testing

- **`tests/test_screenshots.py`** (new): timestamp parsing, anchor boundaries, flagged filter, ID translation.
- **`tests/test_storage.py`** (extended): `read_bytes` / `write_bytes` local + moto S3; `get_presigned_url`.
- **`tests/test_client.py`** (extended): `build_batch_entry` with images, MIME inference, `validate_vision_support`.
- **Integration** (no API calls): fixture conv with 2 screenshots, one flagged — assert detection entries include 1 image path and the formatted prompt contains `[SCREEN @ turn N]`; assert annotation entries filter correctly by excerpt window.
- **Manual smoke:** `099bf759-...` conv on S3, `--with-screenshots --test 1` end-to-end for detection and annotation.

## Out of Scope

- Video → screenshot capture pipeline (upstream owns this).
- LLM-driven anchoring (filename timestamps make it deterministic).
- `fetch_screenshot` tool-call mode on `ModelClient` (deferred — inline-image-only).
- Benchmark tutor / student / labeller multimodal adoption.
- Viewer-side image display.
- Prompt rewrites.
- Video access, video APIs, ffmpeg, de-identification.

## Rollout

- **Phase 1 — ship default-off.** `--with-screenshots` opt-in flag. Existing runs unchanged.
- **Phase 2 — evaluation fork.** Run detection + annotation with `--with-screenshots` on conversations that have screenshots on S3. Save under a new version (e.g., `v5_gold_images/`). Compare recall / kappa vs. baseline, stratified by whether a conv has images.
- **Phase 3 (conditional).** If phase 2 shows measurable improvement, consider flipping default on. Not part of this spec.

No migration. New output-JSON keys are additive; old consumers ignore them.

## Cost

- Detection: ~5–30 images × ~100 convs × 1.6k tokens/image (Claude) ≈ ~10M image tokens per run on Claude, ~$15 at Opus rates. Gemini: ~$1.
- Annotation: ~900 moments × 3 styles × ~2–8 images × 1.6k tokens ≈ ~20M image tokens on Claude per full re-run, ~$330. Prompt caching brings this down substantially because images repeat across overlapping excerpt windows. Gemini: ~$3.
- Per-run costs are the main motivation for wiring prompt caching on from day one for annotation.
