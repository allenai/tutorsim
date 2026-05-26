# Screenshot Enrichment — Smoke Test

Verifies the end-to-end path for `--with-screenshots` against real S3 data
before running at scale.

## Prerequisites

- AWS credentials for the `kylel-alexisr-edu` bucket
- `.env` with `STORAGE_BACKEND=s3`, `S3_BUCKET=kylel-alexisr-edu`,
  `S3_PREFIX=""`, `STORAGE_SCREENSHOTS=deidentified/screenshots`, and a valid
  `ANTHROPIC_API_KEY` (or whichever profile you want)

## Target conversation

`099bf759-2426-549b-8dff-ad3f4be80db2` (verified it has screenshots on S3
as of the design date; pick a different conv if screenshots have moved).

## Detection smoke

```
python -m annotator.core.detect \
  --profile anthropic \
  --with-screenshots \
  --test 1 \
  --version smoke_screenshots
```

Expected:
- Console prints `Screenshots: enabled -- vision model validated`
- Output `detections.json` has `with_screenshots: true`, `convs_with_images >= 1`,
  `total_images_sent > 0`
- `detect_requests.jsonl` entries contain an `"images"` array

## Annotation smoke

After detection completes:

```
python -m annotator.core.annotate \
  --profile anthropic \
  --with-screenshots \
  --version smoke_screenshots
```

Expected:
- Console prints `Screenshots: enabled -- vision model validated, caching ON`
- `annotations.json` has `with_screenshots: true`, `convs_with_images >= 1`,
  `annotations_with_images > 0`

## Cleanup

Results sit under `results/annotator/smoke_screenshots/` — delete when done.
