# Modal Cut Selection -- Design

*Status: spec / 2026-06-09*

## Goal

Replace the current "one scenario per human moment record" with "one scenario per moment cluster," where the cut point is the modal `cut_turn` across all annotators who annotated that moment. ~993 scenarios instead of the current 2,975 (~3x dedup).

## Why

Under the existing `extract_human_scenarios`, every key-moment record becomes its own benchmark scenario. Inspecting the data: 65% of scenarios share `(conv_id, cut_turn)` with at least one other -- different annotators (or the same annotator multiple times) wrote separate moment records for what is the same teachable moment. The benchmark runs the same `(transcript_prefix, cut)` setup multiple times, paying ~3x compute for what is mostly redundant evaluation.

By clustering moments per `(conv_id, turn_start, turn_end)` and selecting one representative cut per cluster (the modal teacher vote), we get one scenario per moment with a teacher-consensus cut point.

## Selection algorithm

Replaces the per-moment loop inside `extract_human_scenarios`. Operates on the same input source (`load_all_ground_truth_files()`) and same outer filters (`conv_id` not in `EXAMPLE_CONV_IDS`, transcript present in `transcripts`, etc.).

1. **Filter** moments to `situation_label_agg in {"scaffolding", "rigor"}` (unchanged from current behavior).
2. **Group** the kept moments by `(conv_id, turn_start, turn_end)`. Each group is a cluster representing one teachable moment.
3. **Per cluster, collect cut votes** by iterating member moments and keeping each member's `cut_turn` when **all** hold:
   - `cut_turn` is present (the key exists and is not `None`),
   - `cut_turn >= turn_start` (cut is not before the moment begins),
   - `cut_turn <= turn_end` (cut is not after the moment ends).
   Each surviving member contributes exactly one vote, **including same-annotator duplicates** -- an annotator who filed 3 records all picking `cut=12` contributes 3 votes for `12`. (Rationale: a strict reading of the pseudocode counts records; if we prefer one-vote-per-annotator later, dedupe by `(annotator_id, cut_turn)` before voting.)
4. **If the cluster has zero votes**, drop it (no usable cut).
5. **Pick the modal cut_turn**; on ties, pick the smallest.
6. **Role-adjust** the chosen cut against the transcript's turn list:
   - If `turns[cut_turn].role == "STUDENT"`: cut stays as-is. Prefix includes that student turn; the AI tutor's first generated turn responds to it.
   - If `turns[cut_turn].role == "TUTOR"`: `cut_turn -= 1`. Prefix excludes the human-tutor turn; the AI tutor's first generated turn replaces it.
   - If the lookup fails (cut_turn not present in turns -- shouldn't happen for valid cuts, but guard against weird data): drop the cluster.
   - After adjustment, additionally enforce `cut_turn >= 1` and `cut_turn <= max(turn_number)`; drop the cluster if adjustment pushes it out of range (e.g. TUTOR adjustment with `cut_turn = 1`).
7. **Pick a representative member** for the `detection` payload. Preference order:
   - Members whose own `cut_turn` equals the modal (pre-adjustment) cut.
   - Among those, the smallest `annotator_id` (lexicographic) for determinism.
   - Use that member's `situation`, `action`, `result`, `annotator_id`, `moment_id`.
8. **Build the `Scenario`** with `transcript_prefix` formatted via the existing `_format_prefix(conversation, cut_turn_adjusted)`.

## `Scenario` shape changes

Only `scenario_id` format and three new `detection` fields. No type changes.

- `scenario_id = f"{conv_id}__hum_{turn_start}_{turn_end}"` -- range-based, stable across re-runs and reclustering. Replaces the current `__hum_{moment_idx}` suffix.
- `detection` dict additions (existing keys preserved):
  - `chosen_cut_turn`: the modal cut before role adjustment (so we can audit vs the role-adjusted `cut_turn` saved on the `Scenario`).
  - `cut_votes`: `{int_cut: int_count, ...}` for traceability.
  - `cluster_size`: total moment-record count in the cluster (informational).

## Code shape

Stays inside `benchmark/core/scenarios.py`. `extract_human_scenarios(transcripts)` keeps its signature but its body is rewritten to follow the algorithm above. Internal helpers may be added:

- `_collect_clusters(gt_files) -> dict[(conv_id, ts, te), list[moment]]`
- `_pick_modal_cut(votes: list[int]) -> int | None` (returns smallest on tie)
- `_role_adjust_cut(cut_turn, conversation) -> int | None` (None if adjustment fails / out of range)
- `_pick_representative_member(cluster, chosen_cut) -> moment_dict`

These helpers stay file-private (underscore-prefixed) so the public surface of `scenarios.py` is unchanged.

## Tests

In `tests/test_benchmark_human_scenarios.py`:

- Update fixture and assertions: the existing `_gt_files()` fixture already produces singleton clusters, so most tests stay valid -- only `scenario_id` assertions change from `__hum_0` to the new range form (`__hum_5_7`, `__hum_8_9`, etc.).
- **New test: modal vote selection** -- three members of a cluster vote `(4, 4, 5)`, modal is 4.
- **New test: tie-break to smallest** -- two members vote `(4, 5)`, smallest wins → 4.
- **New test: same-annotator duplicates inflate votes** -- annotator A files three records voting 4, annotator B files one record voting 5; result is 4.
- **New test: cut-out-of-range votes are dropped** -- member with `cut_turn < turn_start` and member with `cut_turn > turn_end` don't count; cluster falls back to whatever valid votes remain (or drops if none).
- **New test: empty cluster after filter** -- all members have no `cut_turn` or out-of-range cuts → cluster dropped.
- **New test: role adjustment when cut lands on STUDENT** -- chosen cut is a student turn, `Scenario.cut_turn` unchanged.
- **New test: role adjustment when cut lands on TUTOR** -- chosen cut is a tutor turn, `Scenario.cut_turn = chosen - 1`.
- **New test: representative member selection** -- multiple members voted for the modal cut; the one with smallest annotator_id is used in `detection`.
- **New test: detection enrichment** -- output `detection` includes `chosen_cut_turn`, `cut_votes`, `cluster_size`.

## Migration

- Old benchmark runs (e.g. `dyn_smoke_2026_06_08`) used moment-index scenario IDs. Saved results stay readable but won't match new scenario IDs on re-run. No automatic migration -- intentional; the new runs are a fresh benchmark snapshot.
- After this lands, the `extract_human_scenarios` smoke from prior plans should be re-run; expected count drops from ~2,975 to ~993 scenarios.

## Out of scope

- Re-running prior result sets under the new selector.
- Aggregating multiple cut votes into multiple scenarios for "noise" measurement -- explicitly trading variance for cost.
- Changing the `[END]` / `max_turns` exchange behavior just shipped.

## Risk / open questions

- **Vote-counting interpretation**: counts each record; same-annotator duplicates inflate votes. If teammates argue for one-vote-per-annotator, the fix is local (dedup `(annotator_id, cut_turn)` before mode). Easy to flip.
- **Role-adjustment edge cases**: if a tutor cut lands at `cut_turn=1`, adjustment gives `0` and we drop the cluster. Unlikely but accounted for. The role-lookup also assumes `turns[]` is contiguously numbered starting from 1; current data matches this, but the helper guards against missing turns by dropping.
- **scenario_id stability if a moment's range changes**: very unlikely (annotators don't typically edit historical ranges), but if it ever happens, the saved exchange for the old range becomes orphaned. Same risk profile as the prior moment-index approach.
