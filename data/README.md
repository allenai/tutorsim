# Data Directory

Private data — gitignored. Do not commit files from this directory.

For transparency, code that reproduces the numbers in this README.md can be found in data/stats.ipynb. 

---

## transcripts/step_up.jsonl

Deidentified tutoring session transcripts from StepUp. Each line is one session.

**21,445 tutoring sessions across 4 batches** (`2025-10-16`, `2026-02-13`, `2026-03-24`, `2026-04-27`).

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `transcript_id` | string (UUID) | Unique identifier for this session |
| `source` | string | Always `"step_up"` |
| `batch` | string (date) | Ingestion batch (YYYY-MM-DD) |
| `has_video` | bool | Whether video was available during transcription |
| `turns` | array | Ordered list of conversation turns (see below) |
| `session` | object | Session metadata (see below) |
| `eedi_content_link` | array | Links to Eedi math content shown during session (often empty) |
| `enrichments` | array | Non-speech events interleaved with turns (see below) |

### `turns` items

| Field | Type | Description |
|---|---|---|
| `turn_number` | int | 1-indexed turn position |
| `start_seconds` | int | Turn start time in seconds from session start |
| `end_seconds` | int | Turn end time in seconds |
| `role` | string | `"Tutor"` or `"Student"` |
| `text` | string | Transcribed speech |

**Conversation length:** mean 349 turns, min 5, max 1,789 turns per session.

**Turn length (whitespace tokens):** mean 10.8, min 0, max 3,256 tokens per turn (across 7,492,218 turns total).

### `session` fields

| Field | Type | Description |
|---|---|---|
| `transcription_by` | string | Model/service used to transcribe (e.g. `"SUT (Gemini 2.5 Pro)"`) |
| `primary_language` | string | Session language (may be empty) |
| `prior_session_count` | int | Number of prior sessions this tutor–student pair had |
| `session_id` | string (UUID) | Platform session ID |
| `tutor_id` | string (UUID) | Deidentified tutor ID |
| `student_id` | string (UUID) | Deidentified student ID |

### `enrichments` items

Non-speech events (pauses, screen interactions, etc.) that occurred between or before turns.

| Field | Type | Description |
|---|---|---|
| `type` | string | Event category (e.g. `"pause"`, `"screen_interaction"`) |
| `start_seconds` | int | Event start time |
| `end_seconds` | int | Event end time |
| `before_turn` | int | This event occurred before this turn number |
| `label` | string | Short human-readable label |
| `content` | string | Description of what happened |

---

## teacher_annotations/step_up_annotations.jsonl

Human annotations on Step Up tutoring sessions, produced through a structured annotation interface. Each line is one **annotator session** — one annotator's work on one tutoring session, identified by `(annotator_id, transcript_id)`. Each annotator session contains a list of **key moments** (`turn_annotations`): spans `(turn_number_start, turn_number_end)` the annotator flagged, each with SAR fields.

**1,354 annotator sessions** — all from batch `2025-10-16`. Three annotation types: `caption` (79), `scaffolding` (474), `rapport` (801).

**Note:** 0.5% of rapport and scaffolding key moments have `null` for both `turn_number_start` and `turn_number_end`. These have substantive SAR content but were not anchored to specific turns. `build_ground_truth.py` skips these.

### Top-level fields (all types)

| Field | Type | Description |
|---|---|---|
| `transcript_id` | string (UUID) | Links to a record in `step_up.jsonl` |
| `source` | string | Always `"step_up"` |
| `batch` | string (date) | Annotation batch |
| `annotation_type` | string | `"caption"`, `"scaffolding"`, or `"rapport"` |
| `interface_version` | string | Annotation interface version (may be `"untracked"`) |
| `annotator_id` | string | Deidentified annotator identifier |
| `turn_annotations` | array | Segment-level annotations (see per-type detail below) |

### `caption` annotations

Free-text commentary on transcript segments. Also includes overall session-level annotations.

**`turn_annotations` items:**

| Field | Type | Description |
|---|---|---|
| `turn_number_start` | int | First turn of the annotated segment |
| `turn_number_end` | int | Last turn of the annotated segment |
| `annotation_timestamp` | string (ISO 8601) | When this annotation was submitted |
| `text` | string | Free-text commentary on the segment |

**Additional top-level field — `overall_annotations`** (object):

| Field | Description |
|---|---|
| `tutor_effectiveness` | Overall assessment of the tutor's effectiveness in the session |
| `student_learning_indicators` | Observed signals of whether the student learned |
| `student_behavior` | Notable student behaviors the annotator attended to |
| `student_description` | Description of the student's engagement and learning style |

### `scaffolding` and `rapport` annotations

Structured SAR (Situation–Action–Result) annotations on specific moments in the session.

**`turn_annotations` items:**

| Field | Type | Description |
|---|---|---|
| `turn_number_start` | int | First turn of the annotated moment |
| `turn_number_end` | int | Last turn of the annotated moment |
| `annotation_timestamp` | string (ISO 8601) | When this annotation was submitted |
| `situation` | string | What was happening in the session at this moment |
| `action` | string | What the tutor did (the pedagogical move) |
| `result` | string | How effective the move was and what the outcome was |
| `cut_turn` | int | *(some records)* Turn number the annotator chose as the benchmark cut point |
| `cut_turn_reason` | string | *(some records)* Annotator's rationale for the cut point (often empty) |
| `moment_id` | string | *(some records)* Identifier linking cut point to its parent moment. Each `moment_id` maps to exactly one `(transcript_id, turn_number_start, turn_number_end)`, but a given coordinate triple can have multiple `moment_id`s (when two annotators independently annotated the same span). |
| `annotator_id` | string | *(some records)* Annotator identifier at the turn level (duplicates top-level field) |

**Cut point coverage:**

| | scaffolding | rapport |
|---|---|---|
| Annotator sessions `(annotator_id, transcript_id)` with any cut_turn | 295 / 474 | 692 / 801 |
| Key moments `(annotator_id, transcript_id, turn_start, turn_end)` with cut_turn | 2,000 / 3,121 | 5,034 / 5,653 |
| Unique spans `(transcript_id, turn_start, turn_end)` with cut_turn | 1,023 / 1,536 (66.6%) | 928 / 928 (100%) |

Caption annotator sessions never have cut points.

**SAR field lengths (whitespace tokens, across all 8,774 scaffolding + rapport key moments):**

| Field | Mean | Min | Max |
|---|---|---|---|
| `situation` | 24.1 | 1 | 256 |
| `action` | 20.2 | 1 | 342 |
| `result` | 32.7 | 1 | 308 |

---

## ground_truth_hybrid/

One JSON file per conversation. Built from `step_up_annotations.jsonl` by `build_ground_truth.py` using the hybrid labeller. Moments labelled `unclear` are excluded. **207 conversations, 8,724 key moments** (3,106 scaffolding, 5,618 rapport).

### Key moment fields

| Field | Type | Description |
|---|---|---|
| `turn_start` | int | First turn of the moment |
| `turn_end` | int | Last turn of the moment |
| `annotation_type` | string | `"scaffolding"` or `"rapport"` |
| `annotator_id` | string | Deidentified annotator identifier |
| `situation` | string | SAR situation text |
| `action` | string | SAR action text |
| `result` | string | SAR result text |
| `strategy_label` | string | `"effective"`, `"partial"`, or `"ineffective"` |
| `action_decomposed` | list[str] | Atomic tutor action facets extracted by decompose.py |
| `result_decomposed` | list[str] | Atomic student indicator facets extracted by decompose.py |

### Decomposed facet prevalence (scaffolding moments only)

**Tutor actions** (`action_decomposed`): 88.2% of scaffolding moments have at least one action facet; avg **1.72 facets/moment** (max 12).

**Student indicators** (`result_decomposed`): 55.3% of scaffolding moments have at least one result facet; avg **0.83 facets/moment** (max 6).

| Split | Moments | Actions ≥1 | Avg actions | Indicators ≥1 | Avg indicators |
|---|---|---|---|---|---|
| All | 3,106 | 88.2% | 1.72 | 55.3% | 0.83 |
| Train | 1,628 | 90.0% | 1.77 | 57.5% | 0.88 |
| Test | 1,478 | 86.3% | 1.66 | 52.9% | 0.78 |

---

## split.json

Train/test split of the 207 conversations in `ground_truth_hybrid/`, generated by `split_ground_truth.py`. The split is anchored to `original_train_test.json` (which records which transcripts were seen during v4 prompt iteration vs. held out), with 2 IDs excluded because they had no ground truth file and 17 new ground truth conversations (added in later batches) shuffled equally into each set (seed=42).

### Train (102 conversations, 4,615 moments)

| | scaffolding (1,628) | rapport (2,987) |
|---|---|---|
| effective | 653 (40.1%) | 1,451 (48.6%) |
| partial | 299 (18.4%) | 693 (23.2%) |
| ineffective | 671 (41.2%) | 789 (26.4%) |

Span coverage (unique locations vs. annotated moments per location):

| | scaffolding | rapport |
|---|---|---|
| Unique spans `(conv_id, turn_start, turn_end)` | 866 | 468 |
| Annotated moments (one per annotator/span) | 1,628 (1.9/span) | 2,987 (6.4/span) |
| Annotated moments with a cross-annotator span match | 1,207 / 1,628 (74.1%) | 2,987 / 2,987 (100%) |

### Test (105 conversations, 4,109 moments)

| | scaffolding (1,478) | rapport (2,631) |
|---|---|---|
| effective | 575 (38.9%) | 1,227 (46.6%) |
| partial | 278 (18.8%) | 627 (23.8%) |
| ineffective | 606 (41.0%) | 749 (28.5%) |

Span coverage (unique locations vs. annotated moments per location):

| | scaffolding | rapport |
|---|---|---|
| Unique spans `(conv_id, turn_start, turn_end)` | 670 | 459 |
| Annotated moments (one per annotator/span) | 1,478 (2.2/span) | 2,631 (5.7/span) |
| Annotated moments with a cross-annotator span match | 1,154 / 1,478 (78.1%) | 2,631 / 2,631 (100%) |
