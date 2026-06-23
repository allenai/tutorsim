# v5 Prompt Summary

## What v5 Is

v5 = v4 + cut point detection. All v4 prompt content is preserved verbatim. The only additions are to the Pass 1 (detection) prompts.

## What Changed from v4

### Detection (p1): Added `suggested_cut_turn` field

Both `p1/scaffolding.md` and `p1/rapport.md` now ask the detector to identify a `suggested_cut_turn` for each detection -- the specific STUDENT turn where cutting the transcript creates a meaningful evaluation point for a synthetic tutor. The tutor must always be the next speaker after the cut.

- New output field: `suggested_cut_turn` (integer, required)
- New prompt section: "Cut Point Guidance" defining what makes a good cut point
- Hard constraint: must be a STUDENT turn within [turn_start - 2, turn_end]

### Annotation (p2): Removed

v5 p2 prompts were originally identical to v4 p2. They have been removed because the canonical annotation prompts now live in `profiles/{balanced,generous,demanding}/p2/`, which were iterated independently and are significantly more evolved. When running with `--style`, the pipeline uses profile prompts directly. The v4/p2 base prompts serve as the fallback for non-styled runs.

### File extension change

v5 prompts use `.md` extension. `load_prompt` functions updated to try `.md` first, then `.txt`.

### Code changes

- `detect.py`: Updated `load_prompt` for .md support. Validates and always populates `suggested_cut_turn` in `parse_detection_results`.
- `annotate.py`: Updated `load_prompt` for .md support.
- `scenarios.py`: Uses `suggested_cut_turn` from moment data when available.