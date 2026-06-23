# Open Tasks

Small follow-up items that aren't worth a full plan but shouldn't be forgotten.

---

## Fix 5 pre-existing test failures (unrelated to current work)

**Found:** 2026-06-07, while running the full suite to validate `structure.py` changes on `scaffolding_anno`. Confirmed pre-existing (not caused by this branch's work) via `git stash && pytest && git stash pop` -- they fail identically on clean HEAD.

- `tests/test_config.py::test_phase_overrides_merge` -- assertion `50 == 20`, looks like a stale expected config value that drifted from `pipeline/config.yaml`.
- `tests/test_storage.py::TestS3Backend::*` (4 tests) -- `ModuleNotFoundError: No module named 'boto3'` / `'moto'`. Missing test dependencies in the `syn_stu` conda env (`pip install boto3 moto` should resolve, then re-check whether the tests themselves still pass).

**Next step:** Install `boto3`/`moto` in `syn_stu`, re-run `TestS3Backend`, and check whether `test_phase_overrides_merge`'s expected value (`20`) should be updated to match current `config.yaml`, or whether the config itself regressed.
