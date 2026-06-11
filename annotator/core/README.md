# annotator/core

Pipeline scripts for LLM-based annotation. Each script is a self-contained pass that reads from the previous pass's output and writes its own result file.

## Pipeline order

```
detect.py → annotate.py → label.py
                        → situate.py
                        → decompose.py → embed.py
                                       → structure.py
```

`label.py`, `situate.py`, and `decompose.py` all read from `annotate.py` output and can be run in any order relative to each other. `embed.py` and `structure.py` both require `decompose.py` to have run first.

---

### 1. `detect.py` — Key moment detection

Reads full transcripts and detects candidate key moments (turn ranges + brief descriptions). Output is saved as `detections_{target}.json`.

```
python -m annotator.core.detect --version v1
python -m annotator.core.detect --version v1 --target rapport
python -m annotator.core.detect --version v1 --split test
```

---

### 2. `annotate.py` — SAR annotation

Reads detected moments and writes detailed Situation / Action / Result (SAR) annotations for each. Can also run in `--gold` mode to annotate gold truth moments. Output is saved as `annotations_{target}.json`.

```
python -m annotator.core.annotate --version v1
python -m annotator.core.annotate --version v1 --gold
python -m annotator.core.annotate --version v1 --split test
```

---

### 3a. `label.py` — Effectiveness labelling

Reads SAR annotations and classifies each moment as `effective`, `partial`, or `ineffective`. Routes by annotation type (scaffolding vs. rapport) when configured. Output is saved as `labels_{target}.json`.

```
python -m annotator.core.label --version v1
python -m annotator.core.label --version v1 --gold
python -m annotator.core.label --version v1 --split test
```

---

### 3b. `situate.py` — Situation classification (scaffolding only)

Reads scaffolding SAR annotations and classifies each situation as relevant to scaffolding and/or rigor. Output is saved as `situation_labels_{target}.json`.

```
python -m annotator.core.situate --version v1
python -m annotator.core.situate --version v1 --gold
python -m annotator.core.situate --version v1 --split test
```

---

### 3c. `decompose.py` — Facet decomposition

Reads SAR annotations and breaks each `action` and `result` field into lists of short, atomic, standalone statements (`action_decomposed`, `result_decomposed`). Output is saved as `decomposed_{target}.json`.

```
python -m annotator.core.decompose --version v1
python -m annotator.core.decompose --version v1 --gold
python -m annotator.core.decompose --version v1 --split test
```

---

### 4a. `structure.py` — Facet classification

Reads decomposed facets and classifies each:
- `action_decomposed` facets as `scaffolding` / `rigor` / `neither` / `both`, using `prompts/annotator/action_labeller/classify_action.md`
- `result_decomposed` facets as a single mutually-exclusive student-outcome verdict — `pos` (statements trend toward demonstrated understanding/realization) or `neg` (misconceptions/misunderstandings predominantly remain), using `prompts/annotator/student_result_classifier/classify_student_result.md`

Adds `action_label` (str) and `result_label` (str) to each annotation. Annotations with no facets in a field are skipped and given the documented default (`"neither"` for actions, `"no_evidence"` for results — no statements means there's nothing to classify). Output is saved as `structure_labels_{target}.json`.

```
python -m annotator.core.structure --version v1
python -m annotator.core.structure --version v1 --gold
python -m annotator.core.structure --version v1 --split test
```

---

### 4b. `embed.py` — Facet embedding

Reads decomposed facets and encodes them into 384-dim dense vectors using `sentence-transformers/all-MiniLM-L6-v2`. Can also embed ground truth facets directly. Output is saved as `embedded_{target}.json` (or `data/embeddings_{labeller}.json` in ground truth mode).

```
# Embed decompose.py output
python -m annotator.core.embed --version v1
python -m annotator.core.embed --version v1 --gold
python -m annotator.core.embed --version v1 --split test

# Embed ground truth directly
python -m annotator.core.embed --ground-truth
python -m annotator.core.embed --ground-truth --labeller hybrid
```

---

## Support modules

| Module | Purpose |
|---|---|
| `client.py` | Provider-agnostic model client (Anthropic, Gemini, OpenAI) with batch API, retry logic, and usage tracking |
| `config.py` | Loads `pipeline/config.yaml`, resolves the active profile, and provides per-phase config to callers |
| `storage.py` | Read/write layer for local and S3 backends — all file I/O goes through here |
| `utils.py` | Shared utilities: IoU, transcript formatting, excerpt extraction |
| `screenshots.py` | Screenshot anchoring and loading helpers |
