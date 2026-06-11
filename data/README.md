# Data Directory

Private data — gitignored. Do not commit files from this directory.

For transparency, code that reproduces the numbers in this README.md can be found in data/stats.ipynb. 

---

## transcripts/step_up.jsonl

Deidentified tutoring session transcripts from StepUp. Each line is one session.

**31,449 tutoring sessions across 5 batches** (`2025-10-16`, `2026-02-13`, `2026-03-24`, `2026-04-27`, `2026-05-29`).

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

**Conversation length:** mean 325 turns, min 3, max 1,789 turns per session.

**Turn length (whitespace tokens):** mean 11.5, min 0, max 4,058 tokens per turn (across 10,232,059 turns total).

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

**1,564 annotator sessions** — all from batch `2025-10-16`. Three annotation types: `caption` (79), `scaffolding` (684), `rapport` (801).

**Note:** 0.5% of rapport and 0.8% of scaffolding key moments have `null` for both `turn_number_start` and `turn_number_end`. These have substantive SAR content but were not anchored to specific turns. `build_ground_truth.py` skips these.

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
| `moment_id` | string | *(some key moments)* Canonical identifier for this moment, assigned by the annotation interface. Only present on a subset of key moments: 3,877 / 5,508 scaffolding (70.4%) and 4,696 / 5,653 rapport (83.1%). Each `moment_id` maps to exactly one `(transcript_id, turn_number_start, turn_number_end)`, but a given coordinate triple can have multiple `moment_id`s (when two annotators independently identified the same span). |
| `annotator_id` | string | *(some records)* Annotator identifier at the turn level (duplicates top-level field) |

**Cut point coverage:**

| | scaffolding | rapport |
|---|---|---|
| Annotator sessions `(annotator_id, transcript_id)` with any cut_turn | 506 / 684 | 692 / 801 |
| Key moments `(annotator_id, transcript_id, turn_start, turn_end)` with cut_turn | 4,388 / 5,508 | 5,034 / 5,653 |
| Unique spans `(transcript_id, turn_start, turn_end)` with cut_turn | 1,536 / 1,536 (100.0%) | 928 / 928 (100%) |

Caption annotator sessions never have cut points.

**SAR field lengths (whitespace tokens, across all 11,161 scaffolding + rapport key moments):**

| Field | Mean | Min | Max |
|---|---|---|---|
| `situation` | 24.4 | 1 | 256 |
| `action` | 19.6 | 1 | 342 |
| `result` | 29.3 | 1 | 308 |

---

## ground_truth_hybrid/

One JSON file per conversation. Built from `step_up_annotations.jsonl` by `build_ground_truth.py` using the hybrid labeller. Moments labelled `unclear` are excluded. **207 conversations, 10,403 key moments** (5,096 scaffolding, 5,307 rapport).

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

**Tutor actions** (`action_decomposed`): 91.6% of scaffolding moments have at least one action facet; avg **1.84 facets/moment** (max 12).

**Student indicators** (`result_decomposed`): 60.0% of scaffolding moments have at least one result facet; avg **0.92 facets/moment** (max 6).

| Split | Moments | Actions ≥1 | Avg actions | Indicators ≥1 | Avg indicators |
|---|---|---|---|---|---|
| All | 5,096 | 91.6% | 1.84 | 60.0% | 0.92 |
| Train | 2,696 | 92.0% | 1.85 | 61.1% | 0.95 |
| Test | 2,400 | 91.2% | 1.83 | 58.9% | 0.88 |

---

## split.json

Train/test split of the 207 conversations in `ground_truth_hybrid/`, generated by `split_ground_truth.py`. The split is anchored to `original_train_test.json` (which records which transcripts were seen during v4 prompt iteration vs. held out), with 2 IDs excluded because they had no ground truth file and 17 new ground truth conversations (added in later batches) shuffled equally into each set (seed=42).

### Train (102 conversations, 5,543 moments)

| | scaffolding (2,696) | rapport (2,847) |
|---|---|---|
| effective | 1,180 (43.8%) | 1,433 (50.3%) |
| partial | 525 (19.5%) | 670 (23.5%) |
| ineffective | 988 (36.6%) | 691 (24.3%) |

Span coverage (unique locations vs. annotated moments per location):

| | scaffolding | rapport |
|---|---|---|
| Unique spans `(conv_id, turn_start, turn_end)` | 866 | 468 |
| Annotated moments (one per annotator/span) | 2,696 (3.1/span) | 2,847 (6.1/span) |
| Annotated moments with a cross-annotator span match | 2,664 / 2,696 (98.8%) | 2,840 / 2,847 (99.8%) |

### Test (105 conversations, 4,860 moments)

| | scaffolding (2,400) | rapport (2,460) |
|---|---|---|
| effective | 1,047 (43.6%) | 1,214 (49.3%) |
| partial | 439 (18.3%) | 606 (24.6%) |
| ineffective | 909 (37.9%) | 612 (24.9%) |

Span coverage (unique locations vs. annotated moments per location):

| | scaffolding | rapport |
|---|---|---|
| Unique spans `(conv_id, turn_start, turn_end)` | 670 | 459 |
| Annotated moments (one per annotator/span) | 2,400 (3.6/span) | 2,460 (5.4/span) |
| Annotated moments with a cross-annotator span match | 2,375 / 2,400 (99.0%) | 2,460 / 2,460 (100%) |
