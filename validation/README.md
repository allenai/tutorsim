# Validation Notebooks

Reproducible evidence that the synthetic annotation pipeline produces reasonable key moment detections and effectiveness labels. Designed for supplementary material or methods section references.

## Notebooks

1. **`1_detection_validation.ipynb`** -- Key moment detection evidence. Shows that the LLM detector identifies pedagogically meaningful moments at rates consistent with human annotation patterns. Includes IoU sensitivity analysis, per-conversation distributions, overlap visualizations, and error taxonomy.

2. **`2_annotation_validation.ipynb`** -- Annotation and labeling evidence. Shows that the LLM annotator's effectiveness judgments meet or exceed human inter-annotator agreement. Includes per-archetype analysis, confusion matrices, label distributions, and qualitative examples.

## Prerequisites

These notebooks analyze existing pipeline results on disk. 

**Required data** (all gitignored):
- `data/transcripts/` -- Tutoring transcript JSON files
- `data/ground_truth/` -- Human expert annotations
- `results/annotator/` -- Pipeline output (at minimum `v4/detections.json` and `v4/annotations.json`)
- `results/annotator/annotator_profiles.json` -- Annotator archetype classifications

**Python dependencies**: `jupyter`, `matplotlib`, `seaborn`, `pandas`, `numpy` (in addition to the project's own `annotator` package).

## Running

```bash
cd validation/
jupyter notebook
```

Or from the repo root:

```bash
jupyter notebook validation/1_detection_validation.ipynb
```

The notebooks add the repo root to `sys.path` so that `annotator.*` imports work regardless of working directory.

## Output

Key figures are saved to `validation/figures/` as PNG (300 DPI) and PDF for publication use. The `figures/` directory is created automatically on first run.
