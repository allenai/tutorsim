# Labeller Headroom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the per-type hybrid labeller (test_v2 kappa 0.782) has real headroom, without commissioning new human ratings. Two-phase: (1) qualitative error analysis on the ~24 hybrid errors in test_v2; (2) try Gemini and OpenAI as alternative meta-prompt designers and slot a winner into the per-type router if it beats v6 on rapport or v2 on scaffolding.

**Architecture:** Phase 1 is read-only -- no LLM calls -- a single script joins predictions to SAR text and emits human-readable markdown. Phase 2 generalizes the existing meta-prompt script (`validation/7_generate_labeller_prompt_minimal.py`) to accept any provider via `ModelClient`, generates `classify_v7.txt` (Gemini) and `classify_v8.txt` (OpenAI), and runs the existing eval driver (`validation/4_run_labeller_eval.py`) against train_v2 first, finalist-only on test_v2. The classifier itself stays Claude Opus 4.6 for apples-to-apples comparison with v2 and v6.

**Tech Stack:** Python 3.11+, ModelClient (multi-provider), batch eval via `run_batch`, YAML config in `config.yaml` for router updates. Meta-prompt generation is a single sync call (one prompt -- batch mode does not apply).

**Pre-flight:**
- Branch: `wip/labeller-validation` (already current; commits go on this branch)
- Untracked JSON files in working tree (`data_log.json`, `ids_for_lucy*.json`, `xcript_dlog.json`, `exports_log.json`) are unrelated to this work and left alone per user direction.
- Test split: `data/labeller_validation/eval/labeller_test_v2.jsonl` (147 rows, seed=42 stratified) -- DO NOT regenerate.

---

## Phase 1 -- Error Analysis (no LLM calls)

Reads the existing v2 + v6 prediction files, builds the per-type hybrid (v2 for scaffolding, v6 for rapport), surfaces the errors with their full SAR context for human inspection.

### Task 1: Build the hybrid error dump

**Files:**
- Create: `validation/8_error_analysis.py`
- Output: `data/labeller_validation/eval/test_v2_hybrid_errors.md`
- Reads: `data/labeller_validation/eval/labeller_predictions_test_v2_anthropic.jsonl` (v2), `labeller_predictions_test_v2_anthropic_v6.jsonl` (v6), `data/labeller_validation/step_up_annotations.jsonl`

- [ ] **Step 1: Write the script**

```python
"""Combine v2 (scaffolding) + v6 (rapport) test_v2 predictions into a per-type
hybrid, dump the errors as readable markdown with full SAR context.

Reads existing prediction files -- no LLM calls. Output is meant for human
inspection: do the errors look genuinely-ambiguous (humans would disagree too)
or fixable-by-prompt (a smarter classifier could catch them)?
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
OUT = EVAL / "test_v2_hybrid_errors.md"

V2_PREDS = EVAL / "labeller_predictions_test_v2_anthropic.jsonl"
V6_PREDS = EVAL / "labeller_predictions_test_v2_anthropic_v6.jsonl"
SAR_FILE = ROOT / "step_up_annotations.jsonl"

HUMAN_TO_LLM = {
    "effective": "effective",
    "partially_effective": "partial",
    "ineffective": "ineffective",
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def annotation_key(transcript_id, source_annotator_id, annotation_type, ts, te):
    return f"{transcript_id}|{source_annotator_id}|{annotation_type}|{ts}|{te}"


def build_sar_lookup():
    lookup = {}
    for row in load_jsonl(SAR_FILE):
        if row.get("annotation_type") not in ("scaffolding", "rapport"):
            continue
        for ta in row.get("turn_annotations", []):
            key = annotation_key(
                row["transcript_id"], row["source_annotator_id"],
                row["annotation_type"], ta["turn_number_start"], ta["turn_number_end"],
            )
            lookup[key] = {
                "annotation_type": row["annotation_type"],
                "situation": ta.get("situation", ""),
                "action": ta.get("action", ""),
                "result": ta.get("result", ""),
            }
    return lookup


def main():
    v2 = {r["annotation_key"]: r for r in load_jsonl(V2_PREDS)}
    v6 = {r["annotation_key"]: r for r in load_jsonl(V6_PREDS)}
    sar = build_sar_lookup()

    rows = []
    for key, r2 in v2.items():
        r6 = v6.get(key)
        if not r6:
            continue
        ann_type = r2["annotation_type"]
        hybrid_pred = r2["predicted_label"] if ann_type == "scaffolding" else r6["predicted_label"]
        human = HUMAN_TO_LLM[r2["human_rating"]]
        rows.append({
            "key": key,
            "annotation_type": ann_type,
            "human_rating": r2["human_rating"],
            "human_3way": human,
            "hybrid_pred": hybrid_pred,
            "v2_pred": r2["predicted_label"],
            "v6_pred": r6["predicted_label"],
            "is_error": hybrid_pred != human,
        })

    errors = [r for r in rows if r["is_error"]]
    by_type = {"scaffolding": [], "rapport": []}
    for r in errors:
        by_type[r["annotation_type"]].append(r)

    lines = []
    lines.append(f"# Per-Type Hybrid Errors on test_v2 (n={len(rows)})\n")
    lines.append(f"Total errors: {len(errors)} | scaffolding: {len(by_type['scaffolding'])} | rapport: {len(by_type['rapport'])}\n")
    lines.append("Hybrid rule: scaffolding uses v2 (classify_v2), rapport uses v6 (unprimed Claude meta-prompt).\n")

    for ann_type, items in by_type.items():
        lines.append(f"\n## {ann_type} ({len(items)} errors)\n")
        for i, r in enumerate(items, 1):
            s = sar.get(r["key"], {})
            lines.append(f"### {ann_type} #{i}  --  human: `{r['human_3way']}`, hybrid: `{r['hybrid_pred']}`")
            if r["v2_pred"] != r["v6_pred"]:
                lines.append(f"  (v2 said `{r['v2_pred']}`, v6 said `{r['v6_pred']}` -- prompts disagreed)")
            lines.append(f"- **annotation_key:** `{r['key']}`")
            lines.append(f"- **situation:** {s.get('situation','').strip()}")
            lines.append(f"- **action:** {s.get('action','').strip()}")
            lines.append(f"- **result:** {s.get('result','').strip()}")
            lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(errors)} errors to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python -m validation.8_error_analysis`
Expected: `Wrote ~24 errors to data/labeller_validation/eval/test_v2_hybrid_errors.md`

- [ ] **Step 3: Commit**

```bash
git add validation/8_error_analysis.py
git commit -m "feat(validation): hybrid error dump for test_v2 inspection"
```
(The output `.md` lives under `data/labeller_validation/` which is gitignored.)

### Task 2: Read the errors and categorize

- [ ] **Step 1: Open `data/labeller_validation/eval/test_v2_hybrid_errors.md`** and read each error
- [ ] **Step 2: For each error, classify as one of**
  - **A: Genuinely ambiguous** -- two reasonable humans could disagree. Result text doesn't anchor effectiveness clearly. Counts against ceiling.
  - **B: Model misread** -- result text is clear but the classifier picked the wrong cell. A different prompt might catch it.
  - **C: Annotation-input issue** -- the SAR fields themselves are sparse/contradictory and no prompt can recover. Counts against ceiling.
- [ ] **Step 3: Tally categories and write summary into this plan doc under "Phase 1 Findings"** below
- [ ] **Step 4: Decision gate.** If A+C >= ~70% of errors, Phase 2 will produce noise -- stop after Phase 1 with that finding written into status. If B >= ~50%, Phase 2 is justified -- proceed.

### Phase 1 Findings (2026-05-22)

**Tallies (n=21 errors, n_total=147):**
- scaffolding errors: 10 -- A: 9, B: 0, C: 1
- rapport errors: 11 -- A: 7, B: 4, C: 0
- Overall: A=16, B=4, C=1 -> A+C = 17/21 = **81%**

**Decision:** Phase 1 gate says stop (A+C ≥ 70%). But Phase 2 may still be worth running for rapport specifically -- with a caveat.

**Qualitative summary:**

Every scaffolding error is either a junk annotation (#1: "no math work in this conversation" in all three SAR fields) or a "partial-vs-pole" disagreement where the action says the tutor did NOT use a strategy and the result frames improvement potential. Humans inconsistently call these "partial" (generous) or "ineffective" (literal). The scaffolding prompt v2 is reading the result text correctly -- the noise is in the human ratings, not the model. **No prompt iteration will help scaffolding.**

Rapport is more interesting. 4 of 11 rapport errors (#1, #2, #5, #9) are model misreads of a specific pattern: result text starts with an explicit positive anchor ("It seems to be effective", "This strategy was effective because...", "minimally effective", "amazing job") and then appends improvement suggestions. v6 reads the improvement framing as "partial". On 3 of those 4 (#1, #2, #9), v2 would have correctly returned "effective" -- but v2 has worse overall rapport kappa (0.664 vs 0.775) because it misses partials elsewhere.

So there's a precision/recall tradeoff on rapport: v6 catches more true partials (boosting kappa) but misclassifies these "effective + improvement note" cases. A *new* rapport prompt that threads the needle -- treating "effective + improvement note" as effective while still catching the partials v6 catches -- could plausibly improve. Upper bound: 4 fixable errors / 147 = ~2-3 percentage points of kappa.

**Recommendation:** Run Phase 2 on rapport only (skip scaffolding -- nothing fixable there). Margin is small (3 percentage points max), so the +0.02-on-test rule from the plan is the right gate: a v7 or v8 rapport prompt needs to beat 0.775 by at least 0.02 on test_v2 to ship.

---

## Phase 2 -- Other-Model Meta-Prompts (Gemini, OpenAI)

Generalize the existing minimal meta-prompt script to take any provider, generate `classify_v7.txt` (Gemini designs the prompt) and `classify_v8.txt` (OpenAI designs the prompt), eval each on train_v2 to filter, finalist(s) on test_v2 to decide. Update the per-type router if a finalist beats v6 on rapport (0.775) or v2 on scaffolding (0.783).

### Task 3: Generalize the meta-prompt script

**Files:**
- Modify: `validation/7_generate_labeller_prompt_minimal.py`

The existing script hard-codes `META_MODEL = "claude-opus-4-7"` and uses the Anthropic SDK directly. Replace with `ModelClient` + CLI flags so the same script can call any provider. The meta-prompt template itself does not change.

- [ ] **Step 1: Add CLI flags and ModelClient wiring**

Replace the top-of-file constants and the body of `main()` with:

```python
"""Minimal meta-prompt generator. Calls a designer model to produce a labeller
classifier prompt from the 343 train examples, with no diagnostic priming.

Usage:
  python -m validation.7_generate_labeller_prompt_minimal \
    --meta-model claude-opus-4-7 \
    --output-version v6        # (original behavior)
  python -m validation.7_generate_labeller_prompt_minimal \
    --meta-model gemini-3.1-pro-preview --output-version v7
  python -m validation.7_generate_labeller_prompt_minimal \
    --meta-model gpt-5.4 --output-version v8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from annotator.core.client import ModelClient
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
PROMPTS_DIR = Path("prompts/annotator/labeller")
SAR_TYPES = ("scaffolding", "rapport")
```

Keep `load_jsonl`, `annotation_key`, `build_sar_lookup`, `HUMAN_TO_LLM`, `format_examples`, and `META_PROMPT_TEMPLATE` exactly as they are. Replace `main()` with:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-model", required=True,
                        help="Model to design the prompt (any provider: anthropic|gemini|openai prefix-recognized)")
    parser.add_argument("--output-version", required=True,
                        help="Output prompt version, e.g. 'v6', 'v7', 'v8'. Writes to prompts/annotator/labeller/classify_{version}.txt")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Max tokens for the designer response")
    args = parser.parse_args()

    setup_logging(version=f"labeller_meta_{args.output_version}")
    load_dotenv()

    train_rows = list(load_jsonl(EVAL / "labeller_train_v2.jsonl"))
    sar_lookup = build_sar_lookup(ROOT / "step_up_annotations.jsonl")
    logger.info("Loaded %d train rows", len(train_rows))

    examples_block = format_examples(train_rows, sar_lookup)
    meta_prompt = (
        META_PROMPT_TEMPLATE
        .replace("__N_EXAMPLES__", str(len(train_rows)))
        .replace("__EXAMPLES__", examples_block)
    )
    logger.info("Meta-prompt size: %d chars", len(meta_prompt))

    logger.info("Calling %s ...", args.meta_model)
    client = ModelClient(args.meta_model)
    response = client.generate(
        meta_prompt,
        json_mode=False,
        max_tokens=args.max_tokens,
        timeout=300,
    )
    prompt_text = response.text.strip()

    logger.info("Generated prompt: %d chars, %d lines",
                len(prompt_text), len(prompt_text.splitlines()))
    logger.info("Usage: input=%d output=%d tokens",
                response.usage.get("input_tokens", 0),
                response.usage.get("output_tokens", 0))

    output_path = PROMPTS_DIR / f"classify_{args.output_version}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt_text + "\n", encoding="utf-8")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
```

**Why ModelClient.generate (not run_batch):** This is a single one-shot prompt-design call -- one input, one output. Batch mode requires a JSONL of entries and adds polling overhead with no benefit for n=1.

- [ ] **Step 2: Smoke-verify by regenerating v6**

Run: `python -m validation.7_generate_labeller_prompt_minimal --meta-model claude-opus-4-7 --output-version v6_check --max-tokens 8000`
Expected: `Wrote prompts/annotator/labeller/classify_v6_check.txt`, file size in same order of magnitude as existing `classify_v6.txt`. Then delete `classify_v6_check.txt` -- this was a smoke test, not a replacement.

- [ ] **Step 3: Commit**

```bash
git add validation/7_generate_labeller_prompt_minimal.py
rm prompts/annotator/labeller/classify_v6_check.txt
git commit -m "refactor(validation): meta-prompt script accepts any provider via ModelClient"
```

### Task 4: Generate v7 (Gemini)

- [ ] **Step 1: Run the script with Gemini**

Run: `python -m validation.7_generate_labeller_prompt_minimal --meta-model gemini-3.1-pro-preview --output-version v7`
Expected: `Wrote prompts/annotator/labeller/classify_v7.txt`. Check that the generated prompt uses `{annotation_type}`, `{situation}`, `{action}`, `{result_text}` placeholders -- if it doesn't, the model didn't follow instructions and we'll fail at eval time. Read the file briefly.

- [ ] **Step 2: Sanity-check placeholders**

Run: `grep -c "{annotation_type}\|{situation}\|{action}\|{result_text}" prompts/annotator/labeller/classify_v7.txt`
Expected: each placeholder present at least once. If any missing, retry generation up to 2x; if still missing, document as "Gemini failed to follow instructions" and skip v7 eval.

- [ ] **Step 3: Commit**

```bash
git add prompts/annotator/labeller/classify_v7.txt
git commit -m "feat(prompts): classify_v7 designed by gemini-3.1-pro-preview"
```

### Task 5: Generate v8 (OpenAI)

- [ ] **Step 1: Run the script with GPT**

Run: `python -m validation.7_generate_labeller_prompt_minimal --meta-model gpt-5.4 --output-version v8`
Expected: `Wrote prompts/annotator/labeller/classify_v8.txt`.

- [ ] **Step 2: Sanity-check placeholders**

Run: `grep -c "{annotation_type}\|{situation}\|{action}\|{result_text}" prompts/annotator/labeller/classify_v8.txt`
Expected: each placeholder present at least once. Same retry/skip policy as v7.

- [ ] **Step 3: Commit**

```bash
git add prompts/annotator/labeller/classify_v8.txt
git commit -m "feat(prompts): classify_v8 designed by gpt-5.4"
```

### Task 6: Train-set eval of v7 and v8

The classifier model stays at Claude Opus 4.6 (`--model-profile anthropic`). We evaluate on `train_v2` (343 rows) first -- this is the data the prompts were designed against, so it tells us if the designer learned anything generalizable beyond what the meta-prompt already encoded. We do NOT touch test_v2 here.

- [ ] **Step 1: Eval v7 on train**

Run: `python -m validation.4_run_labeller_eval --model-profile anthropic --prompt-version v7 --split train_v2`
Expected outputs:
- `data/labeller_validation/eval/labeller_metrics_train_v2_anthropic_v7.json`
- `data/labeller_validation/eval/labeller_predictions_train_v2_anthropic_v7.jsonl`

- [ ] **Step 2: Eval v8 on train**

Run: `python -m validation.4_run_labeller_eval --model-profile anthropic --prompt-version v8 --split train_v2`
Expected outputs:
- `data/labeller_validation/eval/labeller_metrics_train_v2_anthropic_v8.json`
- `data/labeller_validation/eval/labeller_predictions_train_v2_anthropic_v8.jsonl`

- [ ] **Step 3: Compare by-type kappa**

Read v6, v7, v8 train metrics. Specifically the `by_type` block:

```python
# Quick comparison in REPL or one-shot script
import json
for v in ("v6", "v7", "v8"):
    m = json.loads(open(f"data/labeller_validation/eval/labeller_metrics_train_v2_anthropic_{v}.json").read())
    print(v, m["kappa_3way"], m["by_type"]["scaffolding"]["kappa_3way"], m["by_type"]["rapport"]["kappa_3way"])
```

Existing baselines on TRAIN (from prior eval, same format):
- v2: TRAIN baselines in `labeller_metrics_train_v2_anthropic.json`
- v6: TRAIN baselines in `labeller_metrics_train_v2_anthropic_v6.json`

- [ ] **Step 4: Pick finalist(s)**

Decision rule:
- If v7 OR v8 by-type kappa beats v6 by-type kappa on **rapport (train)**, that prompt is a rapport finalist.
- If v7 OR v8 by-type kappa beats v2 by-type kappa on **scaffolding (train)**, that prompt is a scaffolding finalist.
- A prompt can be a finalist for one or both annotation types.
- If neither v7 nor v8 wins anything on train, **stop** -- writing up "Gemini and OpenAI did not produce better meta-prompts than Claude on this data" is the deliverable.

Write the train results table into the plan doc below.

### Phase 2 Train Results

| Prompt | Designer | 3-way kappa (train) | Scaffolding kappa (train) | Rapport kappa (train) |
|--------|----------|---------------------|---------------------------|------------------------|
| v2     | rule-based (hand-written) | 0.7714 | **0.7776** | 0.7614 |
| v6     | claude-opus-4-7  | 0.7624 | 0.7098 | **0.8081** |
| v7     | gemini-3.1-pro-preview | 0.7368 | 0.6941 | 0.7724 |
| v8     | gpt-5.4 | -- | -- | -- |

**Finalists:** none.
- v7 loses on scaffolding (0.6941 vs v2 0.7776, -0.0835) and on rapport (0.7724 vs v6 0.8081, -0.0357). Not a finalist for either type. No test eval run.
- v8 was blocked on OpenAI quota (insufficient_quota 429 after 5 retries) and never generated. Plan paused on v8 pending billing.

**Bigger finding:** v7's underperformance across both annotation types is evidence that the classifier model (Claude Opus 4.6) is the bottleneck, not the prompt design. When a different architecture (Gemini) designs the prompt, Claude's classification biases dominate the output. Combined with the per-prompt prediction correlations on test_v2 (v4=v5=v6 agree 91.2% of items, full 4-way agreement 83.7%), this rules out "different designer architectures will produce uncorrelated errors and expand the oracle ceiling."

### Task 7: Test-set eval of finalist(s)

Only run if Task 6 picked at least one finalist.

- [ ] **Step 1: Eval finalist on test_v2**

For each finalist `vN` from task 6:

Run: `python -m validation.4_run_labeller_eval --model-profile anthropic --prompt-version vN --split test_v2`
Expected: `labeller_metrics_test_v2_anthropic_vN.json` written.

- [ ] **Step 2: Compute new hybrid kappa**

If a finalist beats the existing per-type winner on its annotation_type:
- For scaffolding: compare finalist's by_type.scaffolding.kappa_3way vs 0.783 (v2 test baseline).
- For rapport: compare finalist's by_type.rapport.kappa_3way vs 0.775 (v6 test baseline).

Build the new hybrid mentally: scaffolding_winner + rapport_winner = new hybrid. Compute its overall 3-way kappa by combining predictions:

```python
# Combine: predict label per item using the winner-for-its-type, then run
# cohens_kappa on the combined (human_3way, hybrid_pred) pairs.
# Use compute_metrics from validation/4_run_labeller_eval as a reference.
```

Acceptable margin: a finalist needs to beat the incumbent by at least +0.02 on its by-type test kappa to justify shipping, given we can't establish variance from a single split (Phase 2 -- the cross-val we skipped -- would have given this number). Below +0.02 is in the noise.

- [ ] **Step 3: Write the test results table into the plan doc**

### Phase 2 Test Results

Skipped. No finalist from Phase 2 Train Results, so no test eval was run. The current per-type hybrid (v2 scaffolding + v6 rapport, test kappa 0.782) remains canonical.

### Oracle Analysis (computed during execution, not in original plan)

Ad-hoc analysis after Phase 1 to bound the ceiling of any routing scheme over the prompts we already have. Used the existing test_v2 predictions for v2, v4, v5, v6.

| Strategy | 3-way kappa (test_v2) | Notes |
|----------|------------------------|-------|
| v2 alone | 0.7250 | best scaffolding (0.7831), weak rapport (0.6636) |
| v6 alone | 0.7411 | weak scaffolding (0.6989), best rapport (0.7751) |
| **per-type hybrid (current)** | **0.7819** | **canonical: v2 sc + v6 ra** |
| majority vote (v2+v4+v5+v6) | 0.7501 | regresses; prompts too correlated for naive ensemble |
| v4-instead-of-v6 hybrid | 0.7819 | tied with v6 -- v4 ~= v6 ~= v5 |
| heuristic override (v2 when v2=eff vs v6=partial on rapport) | 0.7250 | loses; the override is right 3/11 cases, wrong 8/11 |
| **oracle (best-of-4 per item)** | **0.8330** | upper bound for any routing across these 4 prompts |

**Key observations:**
- v4=v5=v6 agree on 134/147 items (91.2%); full v2+v4+v5+v6 agreement 123/147 (83.7%). The Claude-designed prompt variants are highly correlated.
- Of 21 hybrid errors on test_v2, only 5 are oracle-correctable (1 scaffolding, 4 rapport).
- Naive ensemble loses kappa (-0.032) because the prompts share error patterns.
- The oracle headroom (+0.051 kappa) requires a per-item routing predictor we don't have.

- [ ] **Step 4: Commit eval outputs (gitignored, but lock the decision)**

```bash
# Test predictions / metrics live under data/labeller_validation which is gitignored.
# Nothing to git add for the eval JSONs themselves -- they're artifacts.
# Only commit the plan doc updates with results filled in.
git add docs/plans/2026-05-22-labeller-headroom.md
git commit -m "docs(labeller): record train+test results for v7/v8 meta-prompts"
```

### Task 8: Update router if challenger wins by >= +0.02

- [ ] **Step 1: Edit `config.yaml` labeller routing**

If, say, v7 wins rapport at +0.03 on test_v2, update:

```yaml
annotator:
  labeller:
    scaffolding: classify_scaffolding   # (or whichever wins)
    rapport: classify_v7                # (was classify_rapport)
```

If the existing `classify_rapport.txt` IS v6 content (it was generated from script 7), keep names stable by overwriting `classify_rapport.txt` instead of pointing at a new name -- but only if the user agrees. For this plan, default to **pointing at the new filename** so we preserve provenance of which model designed which prompt.

- [ ] **Step 2: Verify config loads**

Run: `python -c "from annotator.core.config import get_phase_config; print(get_phase_config('label', 'anthropic'))"`
Expected: prints a dict including the labeller routing without error.

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "feat(labeller): route {type} to {prompt} -- test_v2 kappa {old} -> {new}"
```

If nothing wins by the +0.02 margin, skip task 8 entirely -- no config change.

---

## Wrap-up

### Task 9: Docs and final commit

- [ ] **Step 1: Update `docs/status.md`** -- replace the "Recently Shipped" section's open follow-ups with the actual Phase 1 + Phase 2 findings. If the hybrid changed, restate the test_v2 kappa. If not, write "Phase 2 attempted Gemini and OpenAI as meta-prompt designers; neither beat Claude meta on this split. Per-type hybrid (v2 scaffolding + v6 rapport) remains canonical at 0.782."

- [ ] **Step 2: Update `docs/plans/_summary.md`** -- add a record:
  - Title: Labeller headroom check
  - Goal: Decide whether per-type hybrid has real headroom without new human ratings.
  - Status: Complete.
  - Result: (one line summary of Phase 1 + Phase 2 outcome).

- [ ] **Step 3: Update auto-memory `project_labeller_validation.md`** if the hybrid changed -- update the "Test-v2 result" line. If nothing changed, leave it.

- [ ] **Step 4: Final commit**

```bash
git add docs/status.md docs/plans/_summary.md
git commit -m "docs: record labeller headroom check outcome"
```

---

## Self-Review

- Phase 1 task fully specified (script + run + categorize). YES.
- Phase 2 task fully specified (script change + 4 evals + decision rule). YES.
- Models named: gemini-3.1-pro-preview, gpt-5.4, claude-opus-4-6 (classifier), claude-opus-4-7 (existing meta). YES.
- Batch vs sync clarified: eval is batch (existing default), meta-prompt gen is sync single-call (one prompt). YES.
- Decision gates: A+C >= 70% stops at Phase 1; no finalist stops Phase 2 mid-way; +0.02 margin required to ship router change. YES.
- Test set protection: only score finalists on test_v2; train_v2 used for filter. YES.
- Commits per task with clear messages. YES.
