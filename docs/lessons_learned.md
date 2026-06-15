# Lessons Learned

Running log of hard-won lessons from building and maintaining this project.

---

## 2026-06-14: Empty `{"key": []}` wrapper silently became two bogus decomposition facets

**What happened:** Evaluating decompose_overscaffold.md, the prompt looked terrible (21% precision, predicted over-scaffolding almost everywhere). The cause wasn't the prompt — it was `_coerce_facets` in `annotator/core/decompose.py`. opus-4-8 (with json_mode) wraps the requested array as `{"spans": [...]}`. For an *empty* result it returns `{"spans": []}`. The old code only took list values when the flattened result was non-empty (`if list_facets:`), so an empty list value fell through to the "cram object keys+values into facets" fallback and returned `['spans', '[]']` — a 2-element list. Downstream, `len(facets) > 0` flipped an empty result into a (wrong) non-empty one.

**Why it matters:** This affects the real ground-truth pipeline, not just the eval — any action/result/over-scaffold decomposition the model returns as `{"facets": []}` / `{"spans": []}` would gain two junk facets. Corrected over-scaffold eval: 58% precision / 100% recall / F1 74% / 85% accuracy.

**Fix:** Distinguish "has any list value" from "flattened result is non-empty". If any value is a list, return the concatenation (possibly empty); only cram when there is no list value at all. Regression tests in `tests/test_decompose_parse.py`.

---

## 2026-03-24: Never gitignore results without preserving them first

**What happened:** During the repo restructuring, `results/` was added to `.gitignore` and the old `synthetic_annotator/results/` directory was deleted as part of collapsing `synthetic_annotator/` into `annotator/`. The results were copied to the new `results/annotator/` location first, but a `git checkout HEAD -- .` command (used to recover from a failed `history/` restore) reverted tracked files and could have wiped results if they'd been in tracked paths.

**Why it matters:** Results contain token usage data from every pipeline and benchmark run -- input_tokens, output_tokens, total_tokens per API call. This is the only record of cumulative spend across the project. Losing results means losing the ability to track total cost.

**Rule:** Before any destructive git operation (`git checkout -- .`, `git reset --hard`, `git clean`), verify that `results/` exists at its expected location and is not at risk. Results are gitignored and exist only on disk -- there is no backup in git history for the new location.

**Where results live:**
- `results/annotator/{version}/` -- detections.json, annotations.json (with token usage per conversation), eval_*.json
- `results/benchmark/{version}/` -- exchanges, annotations, scores, leaderboard (with token usage per scenario)
- `history/` -- archived iteration results from earlier prompt versions (also gitignored, restored from git commit 6ace505)

---

## 2026-03-24: Windows MAX_PATH breaks git operations on deeply nested files

**What happened:** `git checkout 6ace505 -- synthetic_annotator/history/` failed with hundreds of "Filename too long" errors. The history directory contains JSONL files with paths like `synthetic_annotator/history/archive/data/annotations/rapport_annotations/stepup/Forbes/2025-t27975_2024-s8239_3b070869-730b-49a5-a6a6-c22380166d93/rapport_phase_1_incremental.jsonl` -- well over 260 characters when combined with the repo's already-long base path on OneDrive.

**Fix:** `git config core.longpaths true` in the repo. This tells git to use the Windows long-path API (\\?\) which supports up to 32,767 characters.

**Rule:** Always set `core.longpaths true` on Windows repos that may contain deeply nested files, especially when the repo lives under OneDrive (which adds ~80 characters to every path).

---

## 2026-03-24: git checkout HEAD -- . is a sledgehammer

**What happened:** After the failed history restore left partially-extracted files in the staging area, `git checkout HEAD -- .` was used to clean up. This correctly removed the partial files, but also reverted ALL other tracked file edits (benchmark imports, .gitignore, CLAUDE.md) that hadn't been committed yet.

**Rule:** Never use `git checkout HEAD -- .` (or `git restore .`) when you have uncommitted edits to tracked files that you want to keep. Instead:
- Use `git checkout HEAD -- <specific-path>` to target only the files you want to revert
- Or commit your work-in-progress first, then clean up, then amend if needed

---

## Prompt Iteration: v3 to v4 Transition (Lessons from 19+ batch runs)

The v3-to-v4 rewrite was the single biggest quality improvement in the project. v3 was built by accumulating advisor-suggested patches over multiple rounds. v4 was a ground-up rewrite that incorporated what we learned from those failures. These lessons apply to any future prompt iteration work.

Full details are in [annotator/iteration/ITERATION_INSTRUCTIONS.md](../annotator/iteration/ITERATION_INSTRUCTIONS.md) (the "Lessons Learned" section). The iteration logs are in `history/claude_annotation_iteration/annotation_iteration.md` and `history/claude_key_moment_iteration/key_moment_iteration.md`.

### 1. Internal consistency beats metric chasing

v3 was full of contradictory nudges: "be balanced" alongside "Critical Check: What Could Be Better?" and "Before rating, ask yourself..." Each one targeted a specific metric weakness but together they made the prompt incoherent. The model couldn't follow conflicting instructions, so it defaulted to noise.

**Rule:** Before adding any instruction, check if it contradicts existing instructions. If it does, rewrite the section -- don't append a nudge.

### 2. Separation of concerns between pipeline passes

Each pass has one job:
- **Pass 1 (Detector)**: Find turn ranges. Cast a wide net. Don't judge.
- **Pass 2 (Annotator)**: Analyze the moment (situation/action/result). Don't classify.
- **Pass 3 (Labeller)**: Read the full S/A/R analysis and classify (effective/partial/ineffective). Don't re-analyze.

v3 violated this: the annotator prompt included full effectiveness criteria and examples with labels, making the labeller a rubber stamp. When we removed the overlap and let each pass do its own job, metrics improved.

**What went wrong when we tried the opposite:** We tried removing ALL effectiveness language from the annotator (v3 of the annotation iteration) and having the labeller independently classify from balanced text. Kappa dropped 3pp -- when every annotation mentions both strengths and weaknesses, the labeller can't distinguish true "partial" from "effective-with-minor-notes." The labeller needs the annotator to take a stance, just not to produce the final label.

### 3. Prompts need real definitions, not just instructions

v3 said "identify scaffolding-related pedagogical events" without defining what scaffolding means in the context of this study. The model used its general training knowledge, which didn't match the research constructs.

v4 added: research context (what we're studying and why), construct definitions (what scaffolding/rigor/rapport mean here), strategy taxonomies (named strategies the annotator can reference), and the core tradeoff the tutor faces.

### 4. Check the full pipeline, not just the prompt you're iterating

The biggest v4 win came from fixing the labeller -- it was only reading the `result` field, not the full annotation. This had nothing to do with Pass 1 or Pass 2 prompts. We also found a bug where the in-memory pipeline ran Pass 3 successfully but never saved labels to disk -- eval was reading stale data.

**Rule:** Before starting any iteration cycle, run the full pipeline end-to-end and spot-check every pass. Read actual outputs in `annotations.json`, check label distributions, verify the labeller reads enough context.

### 5. The advisor is good at diagnosis, bad at prescription

The LLM advisor identifies error patterns well (it reads 40+ error examples with full transcript context). But its proposed prompt edits are often directional nudges that create the consistency problems from lesson 1.

**Updated rule:** Use the advisor to understand *what's going wrong* (error patterns), but write the fixes yourself. Don't blindly apply all proposed changes.

### 6. Examples are more powerful than instructions

v3's examples included `"effectiveness": "effective"` in the JSON output, so the model produced effectiveness labels regardless of what the instructions said. The examples set the tone, voice, level of detail, and output format more strongly than any instruction paragraph.

**Rule:** When changing prompt behavior, update the examples to match. The model follows examples over instructions when they conflict.

### 7. Changing one pass's output breaks downstream passes

When we removed effectiveness labels from Pass 2's output, the labeller (designed for evaluative text like "Effective. The tutor correctly...") suddenly received analytical text with mixed strengths/weaknesses. It defaulted to "partial" for everything.

**Rule:** When you change a pass's output format or style, re-check that downstream passes still work with the new input.

### 8. Don't patch -- rewrite after 3 rounds

After 3 rounds of advisor-driven patches, the prompt becomes incoherent. Stop patching. Read it end-to-end, then rewrite from scratch incorporating what you learned. This is exactly what the v3-to-v4 transition was.

### 9. Detection has a hard ceiling that prompt iteration can't break

We ran 12 detection iterations (v1-v11). Every content change either regressed or was within the +/-1pp variance band. The only marginal gain came from raising count targets (v10: +1.7pp recall at -5.5pp precision cost).

Why content changes fail:
- Any evaluative language makes the model more selective (even "especially when X")
- New sections create priority signals that displace existing correct detections
- The v1 prompt's simplicity (11 neutral bullets) is its strength

40% of remaining misses are theoretically fixable micro-moments, but every attempt to add them causes regression because the model reallocates "detection attention" from proven patterns to new criteria.

**Conclusion:** Detection ceiling is model-limited, not prompt-limited. Improvements will come from better models or multi-pass detection, not prompt iteration.

### 10. Batch-to-batch variance is large -- always measure it

- Detection: +/-1pp overall, +/-3pp per type
- Annotation: +/-7pp kappa (!)

We confirmed this with variance checks (rerunning identical prompts). A +4pp annotation gain could be noise. After finding a promising change, always re-run to confirm before declaring victory.

### 11. Go back to the source material

When a prompt has drifted through iteration, don't patch further. Go back to the original research instructions and rewrite from scratch. The original research framing ("We are studying how tutors decide to push for rigor versus introduce scaffolds...") was clear and well-scoped. v3 had drifted away from it. Going back improved the prompts more than any advisor cycle.

---

## 2026-05-31: Setting max_tokens too low breaks Anthropic thinking calls

**What happened:** `build_ground_truth.py` passed `max_tokens=32` to `build_batch_entry` for the label classification step (a short yes/no output). When the label phase has `thinking: true` in `config.yaml`, the batch API rejected every request with: `max_tokens must be greater than thinking.budget_tokens`.

**Why it happened:** `thinking_budget` defaults to 16384. `max_tokens` must be strictly greater than `budget_tokens` — the model needs room for both the thinking trace and the actual output. A caller-specified `max_tokens` that made sense for a tiny text output is still invalid when thinking is on.

**Fix:** Removed the `max_tokens=32` override so the call uses `build_batch_entry`'s default (65536), consistent with `label.py` and `situate.py`. Also added a guard in `_run_batch_anthropic` and `_generate_anthropic` in `client.py` that bumps `max_tokens` to `budget_tokens + 64` when thinking is enabled, as a safety net for any future caller that sets a low explicit limit.

**Rule:** Never set a low explicit `max_tokens` on a batch entry without accounting for the thinking budget. Prefer omitting `max_tokens` and letting the pipeline default handle it.

---

## 2026-05-30: situation_label_agg: must exclude both=no_mention annotators before majority vote

**What happened:** First implementation of `compute_situation_label_agg` remapped `no_mention` → `no` for all annotators, then took a majority vote. This inflated "neither" counts and turned clusters into "mixed" ties. The notebook's expected counts (both=14, scaffolding=589, rigor=338, neither=157) were not reproduced.

**Why it happened:** The notebook's `build_binary_conf_counts` has an explicit step: *exclude annotators whose (scaf, rigor) tuple is both `no_mention`* before collecting votes. An annotator who left both slots as `no_mention` gave no signal — including them as a `(no, no)` vote biases the result toward "neither" and creates false ties.

**Fix:** Three-step process matching the notebook exactly:
1. Normalize `unclear`/`None` → `no_mention` (mirrors notebook's `_sit` helper)
2. Skip annotators where both slots are `no_mention`
3. Remap remaining `no_mention` → `no`, then majority-vote

---

## 2026-04-28: Benchmark screenshots — wired, but data-pairing gap blocks validation

**Original gap:** The annotator pipeline supports `--with-screenshots` (delivered 2026-04-24) but the benchmark — which reuses the annotator under the hood — never threaded the flag. Detection ran without images; tutor/student exchanges were text-only; annotation didn't see the screen even though the equivalent annotator-standalone run did.

**Why the flag was a no-op:** `build_analysis_entries` and `build_detection_entries` were keying screenshot lookup on `conv_id`. The benchmark bridge remaps `conv_id -> scenario_id` to namespace bulk batch keys, so the lookup silently returned `[]`.

**Fix (committed 2026-04-28):** Decoupled screenshot loading from screenshot use. Both functions now accept an optional `screenshots_by_conv` dict — if provided, the function uses it directly instead of looking up by conv_id. The bridge loads screenshots using the original `scenario.conv_id` and passes a dict keyed on `scenario_id` so the function's iteration key still matches. Phase 1 exchange similarly accepts `images=` (sync) and `images_by_scenario=` (batch). Vision validation runs once at run start when `--with-screenshots` is on. Default off — existing text-only runs are byte-for-byte unchanged.

**Caveat — data-pairing gap (not yet resolved):** Smoke-tested but **never exercised end-to-end with real images**. The S3 bucket has screenshots for only 3 conv UUIDs (`099bf759`, `202f38ab`, `9c6f61b1`) under `deidentified/screenshots/`, and *none* of those UUIDs appear anywhere else in the bucket — not in `deidentified/step_up.jsonl` (250 deidentified transcripts, all `has_video: False`), not in `transcripts/text transcripts.zip` (109 transcripts), not in our local 212 transcripts. Until someone deidentifies-and-publishes transcripts for those video sessions (or deidentifies more screenshots for already-published transcripts), every benchmark run with `--with-screenshots` degrades to text-only because the loader honestly returns `[]` for every conv. This isn't an S3 access problem (creds work fine via boto3 default chain) — it's a data-completeness problem owned by whoever runs the deidentification pipeline.

**How to verify when data lands:** any conv that's in both the local transcript set and S3's `deidentified/screenshots/` will produce non-zero `total_images_sent` in `detections.json`. That's the cleanest single-field signal that real images flowed.

**Latent gap noted but not fixed:** Benchmark annotation shards don't track `images_attached`/`images_seen` because `annotator_bridge.execute_and_parse_bulk` calls `parse_and_merge` directly, skipping the `_stamp_and_shard` step in `annotator/core/annotate.py:343-356` that adds those fields. Worth a follow-up so annotation-side image flow is visible per-shard, not only inferable from API request logs.

---

## 2026-06-10: Opus 4.8 (and 4.7 / Fable 5) reject `thinking.type=enabled`

**What happened:** Switching the `anthropic` profile from `claude-opus-4-6` to `claude-opus-4-8` made every call 400 with: `"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort"`. The Anthropic request builders hardcoded `thinking: {type: enabled, budget_tokens: N}`.

**Why:** Opus 4.7, Opus 4.8, and Fable 5 removed manual extended thinking. They require adaptive thinking (`thinking: {type: adaptive}`) with depth controlled by `output_config: {effort: low|medium|high|xhigh|max}`. Opus 4.6 and earlier still accept `budget_tokens`.

**Fix:** Added `_build_anthropic_thinking(model, ...)` in `annotator/core/client.py` — model-aware: adaptive-only models (`opus-4-7`, `opus-4-8`, `fable-5`) get `{thinking: adaptive, output_config: {effort}}`; older models keep `{thinking: enabled, budget_tokens}` + the max_tokens floor. Both sync (`_generate_anthropic`, via `extra_body`) and batch (`_run_batch_anthropic`, merged into params) paths use it. `reasoning_effort` is threaded from config (defaults to `high`). The pinned SDK (0.75.0) has no typed `output_config`/adaptive params, so the sync path sends them via `extra_body`; batch keeps them as extra dict keys (they survive `maybe_transform` serialization). `extra_body` produces byte-identical wire JSON to the old typed `thinking` kwarg, so opus-4-6 runs are unchanged.

**Config:** `reasoning_effort: high` added to the `anthropic` profile (opus-4-8). `thinking_budget` is now ignored by opus-4-8 but kept for the `anthropic46` profile.

---

## 2026-06-10: OpenAI json_object mode can't emit a top-level array — decompose dropped facets

**What happened:** `openai` profile runs logged `annotator.core.decompose | Could not parse result decomposition ... '{ "The student participates." : "The student identifies...", ... }'` and stored empty `result_decomposed` lists. The model returned a JSON *object*, but `_parse_decomposed` only accepted a list, so every facet was dropped (counted as an error).

**Why:** The decomposer prompts ask for a bare JSON array, but the OpenAI path sets `response_format={"type": "json_object"}` (`client.py` sync ~414, batch ~849), which forbids a top-level array. The model is forced to wrap the facets in an object — either under a key (`{"facets": [...]}`) or, with no list to hand, crammed across the object's keys *and* values (`{"facet a": "facet b", ...}`). Gemini (only `response_mime_type`) and Anthropic (soft system rule) honor the array, so this is OpenAI-only.

**Fix:** Added `_coerce_facets()` in `annotator/core/decompose.py` and routed `_parse_decomposed` through it. It accepts a bare array, a `{...: [list]}` wrapper (flattens list values), or the crammed shape (interleaves keys+values). Parser-level fix only — no prompt/client changes, so Gemini/Anthropic outputs are unchanged. Covered by `tests/test_decompose_parse.py`. Tradeoff: a single-key wrapper with a *string* value (e.g. `{"facets": "one facet"}`) would yield `["facets", "one facet"]`, but OpenAI doesn't emit that shape in practice.

---

## 2026-06-10: eval scorecard filename omitted `--profile` — profiles clobbered each other

**What happened:** `annotator.eval.eval` selected profile-specific *input* files (detections/annotations/structure all build a `_{profile}` suffix), but the *output* filename was `eval_{mode}{style}{split}.json` — no profile. So `--profile anthropic` then `--profile openai` (same mode/style/split) loaded different inputs but wrote the same `eval_{mode}.json`, silently clobbering the first. `load_eval_json`/`--compare` and `advisor.load_eval_metrics` also read the unsuffixed name, so a profile-suffixed scorecard wouldn't be found.

**Fix:** Added `eval_output_filename(mode, profile, style, split)` (single source of truth, suffix order profile→style→split, train stays unsuffixed). Save site uses it; `load_eval_json` now takes profile/style/split and tries suffixed → unsuffixed `eval_{mode}.json` → legacy `eval.json`. `--compare` threads `--profile`. `advisor.load_eval_metrics` now delegates to `load_eval_json` (replacing its own ad-hoc `eval_mode_key = f"annotations{style}"` hack, which ignored profile). Output dict now records profile/style/split for self-identification. Covered by `tests/test_eval_metrics.py`.
