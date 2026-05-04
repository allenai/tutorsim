# Follow-up Tasks

Open items surfaced during other work. Each entry: what, why, and where to start.

## `Evaluator._load_jsonlines` duplicates `io_utils.load_jsonl`

**What:** [tutor_bench/evaluator.py:153](../tutor_bench/evaluator.py#L153) defines `_load_jsonlines(filepath: str)`, a private JSONL loader that predates the shared `load_jsonl` in [tutor_bench/toolkit/io_utils.py](../tutor_bench/toolkit/io_utils.py).

**Why it matters:** Two loaders means two places to fix bugs and two places where storage-backend support (now `STORAGE_ROOT` / `UPath` aware) has to be re-implemented. The evaluator's loader does not benefit from S3 routing.

**Where to start:** Replace the body (or call sites) of `_load_jsonlines` with `tutor_bench.toolkit.io_utils.load_jsonl`. Verify [tests/](../tests/) coverage exercises the evaluator's load path before/after.
