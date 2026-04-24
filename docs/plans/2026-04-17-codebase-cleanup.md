# Codebase Cleanup & CLI Simplification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix verified bugs, eliminate hardcoded constants, and simplify CLI invocation so common operations need fewer flags.

**Architecture:** All fixes weave into existing patterns. CLI simplification adds an `annotator:` section to config.yaml for defaults that currently require CLI flags. Auto-generates `--version` when not provided. No new abstractions, no new files.

**Tech Stack:** Python, YAML config, argparse

---

## Verified Issues Being Fixed

| # | Issue | Severity |
|---|-------|----------|
| 1 | `benchmark/eval/view.py:315` — mode compared to `'key_moment'` instead of `'detected'` | Bug |
| 2 | `benchmark/run.py:85` — redundant `config["resolved_models"]` assignment | Cleanup |
| 3 | `benchmark/run.py:105` — detect_mode sourced from annotator config, not detect config | Bug |
| 4 | JUNK_TEXTS duplicated in 3 places | DRY violation |
| 5 | ARCHETYPE_ANNOTATORS hardcoded in function body | Config concern |
| 6 | `benchmark/eval/view.py:430` — dead `strategy_label` fallback | Cleanup |
| 7 | CLI requires too many flags for common operations | Ergonomics |

## Verified NOT Issues (No Action)

- **Cohen's kappa formula** (`eval/eval.py:87-106`): Correct. Standard weighted kappa with disagreement weights. `pe / (n*n)` is the right normalization for expected proportions.
- **`get_backend()` vs `_get_backend()`** in storage.py: Standard Python public/private convention. Not a naming collision.
- **Style agreement cartesian product** (`benchmark/eval/eval.py:109-115`): Works correctly. The pipeline guarantees one annotation per type per style per scenario.
- **`aggregate.py` label overwrite**: Same — pipeline guarantees one annotation per type.
- **Thread safety of module globals**: CLI tool, not a service. Irrelevant.
- **IoU set operations**: Turn ranges are 1-50. Premature optimization.
- **Binary labeling mode**: Implemented and functional. Just not commonly used.

---

### Task 1: Fix benchmark/eval/view.py mode bug

**Files:**
- Modify: `benchmark/eval/view.py:315`

- [ ] **Step 1: Fix the mode comparison string**

In `benchmark/eval/view.py`, line 315, change `'key_moment'` to `'detected'`:

```python
# Before:
const modeClass = s.mode === 'key_moment' ? 'key_moment' : 'random';

# After:
const modeClass = s.mode === 'detected' ? 'detected' : 'random';
```

Also update the CSS class name to match. Search for any `.key_moment` CSS selector in the same file and rename it to `.detected`.

- [ ] **Step 2: Remove dead strategy_label fallback**

Line 430, clean up the label resolution:

```python
# Before:
const label = ann.effectiveness || ann.strategy_label || 'unclear';

# After:
const label = ann.effectiveness || 'unclear';
```

`strategy_label` is never set on AI annotation objects. This fallback is dead code.

- [ ] **Step 3: Commit**

```bash
git add benchmark/eval/view.py
git commit -m "fix: benchmark viewer mode class and dead strategy_label fallback"
```

---

### Task 2: Fix benchmark/run.py bugs

**Files:**
- Modify: `benchmark/run.py:85,93,105`

- [ ] **Step 1: Remove redundant resolved_models assignment**

Line 85 assigns `config["resolved_models"] = resolved_models`, then line 93 does the same thing (after adding detector). Since `resolved_models` is the same dict object mutated in place between the two lines, the first assignment is redundant. Remove line 85.

```python
# Before (lines 84-93):
    resolved_models["labeler"] = get_phase_config("label", ann_profile)["model"]
    config["resolved_models"] = resolved_models
    config["run_version"] = version

    # Resolve detect model
    detect_cfg_section = config.get("detect", {})
    detect_profile = detect_cfg_section.get("profile", "anthropic")
    detect_prompt_version = detect_cfg_section.get("prompt_version", "v5")
    resolved_models["detector"] = get_phase_config("detect", detect_profile)["model"]
    config["resolved_models"] = resolved_models

# After:
    resolved_models["labeler"] = get_phase_config("label", ann_profile)["model"]
    config["run_version"] = version

    # Resolve detect model
    detect_cfg_section = config.get("detect", {})
    detect_profile = detect_cfg_section.get("profile", "anthropic")
    detect_prompt_version = detect_cfg_section.get("prompt_version", "v5")
    resolved_models["detector"] = get_phase_config("detect", detect_profile)["model"]
    config["resolved_models"] = resolved_models
```

- [ ] **Step 2: Fix detect_mode sourcing**

Line 105 currently reads detect_mode from the annotator config section:

```python
# Before:
detect_mode = config.get("annotator", {}).get("mode", "batch")

# After:
detect_mode = detect_phase_cfg.get("mode", "batch")
```

`detect_phase_cfg` is already resolved on line 103 from the correct detect profile. This is the right source for detection mode.

- [ ] **Step 3: Add comment explaining student model resolution**

Line 81 uses `get_phase_config("tutor", student_profile)` to resolve the student model. This is correct (the "tutor" phase has the base model for any profile, and there's no separate "student" phase), but reads like a bug. Add a clarifying comment:

```python
# Student uses the base model from its profile (no separate "student" phase in config)
resolved_models["student"] = get_phase_config("tutor", student_profile)["model"]
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/run.py
git commit -m "fix: detect_mode source, redundant assignment, student model comment"
```

---

### Task 3: Centralize JUNK_TEXTS

**Files:**
- Modify: `annotator/core/label.py:41` (keep as canonical source)
- Modify: `data/build_ground_truth.py:40` (import instead of duplicate)
- Modify: `data/extract_ground_truth.py:60-62` (import instead of inline check)

- [ ] **Step 1: Import JUNK_TEXTS in build_ground_truth.py**

In `data/build_ground_truth.py`, replace the local definition:

```python
# Before (line 40):
JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}

# After:
from annotator.core.label import JUNK_TEXTS
```

Verify this import works: `build_ground_truth.py` already imports from `annotator.core.client` and `annotator.core.config`, so this import path is established.

- [ ] **Step 2: Import JUNK_TEXTS in extract_ground_truth.py**

In `data/extract_ground_truth.py`, replace the inline check:

```python
# Before (lines ~60-62):
if not stripped or stripped in ("n/a", "test", "sdf", "this is a test annotation"):
    labels.append("unclear")

# After:
# At top of file, add:
from annotator.core.label import JUNK_TEXTS

# At the check:
if stripped in JUNK_TEXTS:
    labels.append("unclear")
```

Note: The empty string check is already handled — `JUNK_TEXTS` includes `""`, and `stripped` will be `""` if the original was empty/whitespace.

- [ ] **Step 3: Verify no other duplicates exist**

```bash
grep -rn "this is a test annotation" --include="*.py"
```

Should only find `annotator/core/label.py`.

- [ ] **Step 4: Commit**

```bash
git add annotator/core/label.py data/build_ground_truth.py data/extract_ground_truth.py
git commit -m "cleanup: centralize JUNK_TEXTS in label.py, remove duplicates"
```

---

### Task 4: Move ARCHETYPE_ANNOTATORS to config.yaml

**Files:**
- Modify: `config.yaml` (add archetype mapping)
- Modify: `annotator/core/config.py` (add accessor)
- Modify: `annotator/core/utils.py:93-97` (read from config instead of hardcoding)

The archetype-to-annotator mapping is data-derived and referenced in multiple places (utils.py, classify_annotators.py produces it, eval uses it via load_ground_truth). Moving it to config.yaml makes it visible and editable.

- [ ] **Step 1: Add archetype_annotators to config.yaml**

Add after the `profile:` line (top-level, since it's used by annotator pipeline):

```yaml
# Annotator archetype assignments (from classify_annotators.py analysis)
archetype_annotators:
  generous:
    - Gerber
    - Jones
    - Shields
    - Stobbe
    - Trujillo
  balanced:
    - Forbes
    - Mann
    - Padgett
  demanding:
    - Flick
```

- [ ] **Step 2: Add config accessor**

In `annotator/core/config.py`, add a function:

```python
def get_archetype_annotators(archetype: str | None = None) -> dict[str, set[str]] | set[str] | None:
    """Get annotator IDs by archetype from config.

    Args:
        archetype: If given, return the set of annotator IDs for that archetype.
                   If None, return the full mapping {archetype: set(annotator_ids)}.

    Returns:
        Full mapping dict, or a set of annotator IDs, or None if archetype not found.
    """
    config = load_config()
    raw = config.get("archetype_annotators", {})
    mapping = {k: set(v) for k, v in raw.items()}
    if archetype is None:
        return mapping
    return mapping.get(archetype)
```

- [ ] **Step 3: Update utils.py to use config**

In `annotator/core/utils.py`, replace the hardcoded dict inside `load_ground_truth()`:

```python
# Before (lines 93-98):
    ARCHETYPE_ANNOTATORS = {
        "generous": {"Gerber", "Jones", "Shields", "Stobbe", "Trujillo"},
        "balanced": {"Forbes", "Mann", "Padgett"},
        "demanding": {"Flick"},
    }
    filter_ids = ARCHETYPE_ANNOTATORS.get(annotator_style) if annotator_style else None

# After:
    from .config import get_archetype_annotators
    filter_ids = get_archetype_annotators(annotator_style) if annotator_style else None
```

- [ ] **Step 4: Verify the pipeline still works**

```bash
python -m annotator.eval.eval --version v5_gold --style balanced --mode annotations
```

Compare output to known values (balanced 3-way kappa should be ~0.4574).

- [ ] **Step 5: Commit**

```bash
git add config.yaml annotator/core/config.py annotator/core/utils.py
git commit -m "cleanup: move archetype_annotators mapping to config.yaml"
```

---

### Task 5: Add annotator defaults to config.yaml + simplify CLI

**Files:**
- Modify: `config.yaml` (add annotator section)
- Modify: `annotator/core/config.py` (add annotator defaults accessor)
- Modify: `annotator/run.py` (read defaults from config, auto-generate version)
- Modify: `annotator/core/detect.py` (read defaults from config)
- Modify: `annotator/core/annotate.py` (read defaults from config)
- Modify: `annotator/core/label.py` (read defaults from config)
- Modify: `annotator/eval/eval.py` (read defaults from config)

The goal: `python -m annotator` should work with zero flags for the most common use case. Every default lives in config.yaml so it's visible and editable.

- [ ] **Step 1: Add annotator defaults section to config.yaml**

Add after `profile:` and `archetype_annotators:`:

```yaml
# Annotator pipeline defaults (override with CLI flags)
annotator:
  prompt_version: v4           # prompt version for detection + annotation
  style: null                  # null = no style. Set to generous/balanced/demanding
  version: null                # null = auto-generate as {profile}_{date}
```

- [ ] **Step 2: Add defaults accessor to config.py**

```python
def get_annotator_defaults() -> dict:
    """Get annotator pipeline defaults from config."""
    config = load_config()
    return config.get("annotator", {})
```

- [ ] **Step 3: Update annotator/run.py to use config defaults**

Make `--version` optional. When omitted, auto-generate from `{profile}_{YYYY-MM-DD}`. Read `--style` and `--prompt-version` defaults from config.

```python
# In main(), after parser setup:

# Change --version from required=True to:
parser.add_argument("--version", default=None,
                    help="Results version directory (default: auto-generated from profile + date)")

# After parse_args:
from .core.config import get_annotator_defaults
import datetime

defaults = get_annotator_defaults()

# Resolve profile first (needed for version generation)
profile = args.profile or load_config().get("profile", "anthropic")

# Auto-generate version if not specified
if args.version:
    version = args.version
elif defaults.get("version"):
    version = defaults["version"]
else:
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    version = f"{profile}_{date_str}"
    print(f"  Auto-generated version: {version}")

# Resolve style from config if not on CLI
style = args.style or defaults.get("style")

# Resolve prompt_version: CLI > config > version
prompt_version = args.prompt_version or defaults.get("prompt_version") or version
```

Then thread `version`, `style`, and `prompt_version` through the rest of the function (replacing `args.version`, `args.style`, `args.prompt_version or args.version`).

- [ ] **Step 4: Update detect.py main() similarly**

Same pattern: make `--version` optional, read style/prompt-version defaults from config.

```python
def main():
    from .config import get_annotator_defaults
    import datetime

    defaults = get_annotator_defaults()
    phase_cfg = get_phase_config("detect")

    parser = argparse.ArgumentParser(description="Pass 1: Key moment detection")
    parser.add_argument("--version", default=None,
                        help="Results version (default: auto-generated)")
    # ... other args unchanged ...

    args = parser.parse_args()

    if args.profile:
        phase_cfg = get_phase_config("detect", args.profile)

    profile = args.profile or load_config().get("profile", "anthropic")
    version = args.version or defaults.get("version") or f"{profile}_{datetime.date.today():%Y-%m-%d}"
    style = args.style or defaults.get("style")
    prompt_version = args.prompt_version or defaults.get("prompt_version") or version
    # ... rest unchanged, using local vars instead of args.* ...
```

- [ ] **Step 5: Update annotate.py main() similarly**

Same pattern as detect.py.

- [ ] **Step 6: Update label.py main() similarly**

Same pattern. Note: label.py doesn't need prompt_version.

- [ ] **Step 7: Update eval/eval.py to read version from config**

The eval module already has `--version` as optional (only required in non-compare mode). Add config fallback:

```python
# In main(), after parse_args:
if not args.version and not args.compare:
    from annotator.core.config import get_annotator_defaults
    defaults = get_annotator_defaults()
    version = defaults.get("version")
    if not version:
        parser.error("--version is required (unless using --compare, or set in config.yaml annotator.version)")
else:
    version = args.version
```

- [ ] **Step 8: Verify zero-flag invocation**

With `style: balanced` and `prompt_version: v4` set in config.yaml:

```bash
# Should auto-generate version as anthropic_2026-04-17 and use balanced style
python -m annotator --test 1 --mode sync
```

Verify it:
1. Prints the auto-generated version name
2. Uses balanced style prompts
3. Uses v4 prompt version
4. Runs successfully on 1 transcript

- [ ] **Step 9: Commit**

```bash
git add config.yaml annotator/core/config.py annotator/run.py annotator/core/detect.py annotator/core/annotate.py annotator/core/label.py annotator/eval/eval.py
git commit -m "feat: annotator CLI defaults from config.yaml, auto-generate version"
```

---

### Task 6: Simplify benchmark CLI similarly

**Files:**
- Modify: `config.yaml` (add benchmark version default)
- Modify: `benchmark/run.py` (auto-generate version when omitted)

- [ ] **Step 1: Add version default to benchmark config**

In `config.yaml`, under `benchmark:`, add:

```yaml
benchmark:
  version: null  # null = auto-generate as {tutor_profile}_{date}
  scenarios:
    # ... existing ...
```

- [ ] **Step 2: Update benchmark/run.py**

Make `--version` optional with auto-generation:

```python
parser.add_argument("--version", default=None,
                    help="Benchmark version (default: auto-generated from tutor profile + date)")

# After parse_args:
config = load_config(overrides)

if args.version:
    version = args.version
else:
    bm_version = config.get("version")
    if bm_version:
        version = bm_version
    else:
        import datetime
        tutor_profile = config.get("tutor_profiles", ["anthropic"])[0]
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        version = f"{tutor_profile}_{date_str}"
        print(f"  Auto-generated version: {version}")

run_benchmark(version, config)
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml benchmark/run.py
git commit -m "feat: benchmark CLI auto-generate version, version default in config"
```

---

## Summary: What running looks like after these changes

**Before:**
```bash
python -m annotator --version v5_gold --profile anthropic --style balanced --prompt-version v4
python -m annotator.eval.eval --version v5_gold --style balanced
python -m benchmark --version claude-opus-4-6_2026-04-17
```

**After (with config defaults set):**
```bash
python -m annotator
python -m annotator.eval.eval
python -m benchmark
```

All flags still work as overrides. Zero breaking changes.
