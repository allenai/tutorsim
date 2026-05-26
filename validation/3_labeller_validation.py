"""Labeller validation analysis.

Reads the per-reviewer JSONL files at data/labeller_validation/{reviewer}.jsonl
and reports:
  1. Per-reviewer rating distributions.
  2. Cross-reviewer overlap on annotation_key + inter-rater agreement.
  3. A v2 stratified train/test split that includes nathan.

Run: python -m validation.3_labeller_validation
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path("data/labeller_validation")
EVAL = ROOT / "eval"
REVIEWERS = ["dani", "nathan", "query", "rebecca"]
DONE_RATINGS = {"effective", "partially_effective", "ineffective"}
SEED = 42
TEST_RATIO = 0.3


def load_reviewer(name: str) -> list[dict]:
    path = ROOT / f"{name}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_rating_per_key(rows: list[dict]) -> dict[str, dict]:
    """If a reviewer rated the same key twice, keep the most recent row."""
    rows_sorted = sorted(rows, key=lambda r: r.get("submitted_at", ""))
    out: dict[str, dict] = {}
    for row in rows_sorted:
        out[row["annotation_key"]] = row
    return out


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float | None, int]:
    n = len(pairs)
    if n == 0:
        return None, 0
    classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in classes)
    if pe == 1.0:
        return 1.0, n
    return (po - pe) / (1 - pe), n


def per_reviewer_summary() -> dict:
    summary = {}
    for name in REVIEWERS:
        rows = load_reviewer(name)
        deduped = last_rating_per_key(rows)
        ratings = Counter(r["rating"] for r in deduped.values())
        strategies = Counter(
            r["annotation_type"] for r in deduped.values() if r.get("annotation_type")
        )
        done = sum(v for k, v in ratings.items() if k in DONE_RATINGS)
        summary[name] = {
            "raw_rows": len(rows),
            "unique_keys": len(deduped),
            "ratings": dict(ratings),
            "done_ratings": done,
            "strategies": dict(strategies),
        }
    return summary


def overlap_and_ceiling() -> dict:
    """Build per-reviewer {annotation_key: rating} maps (DONE_RATINGS only),
    intersect across pairs, and compute Cohen's kappa per pair plus an overall
    'pooled' kappa on all pairwise overlaps."""
    by_reviewer: dict[str, dict[str, str]] = {}
    by_reviewer_with_type: dict[str, dict[str, str]] = {}
    for name in REVIEWERS:
        rows = load_reviewer(name)
        deduped = last_rating_per_key(rows)
        by_reviewer[name] = {
            k: r["rating"] for k, r in deduped.items() if r["rating"] in DONE_RATINGS
        }
        by_reviewer_with_type[name] = {
            k: (r["rating"], r["annotation_type"])
            for k, r in deduped.items()
            if r["rating"] in DONE_RATINGS
        }

    pairwise = {}
    pooled_pairs: list[tuple[str, str]] = []
    pooled_pairs_by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for a, b in combinations(REVIEWERS, 2):
        shared = set(by_reviewer[a]) & set(by_reviewer[b])
        if not shared:
            pairwise[f"{a}__{b}"] = {"n": 0, "kappa": None, "agreement": None}
            continue
        pairs = [(by_reviewer[a][k], by_reviewer[b][k]) for k in shared]
        kappa, n = cohens_kappa(pairs)
        agree = sum(1 for x, y in pairs if x == y) / n if n else None
        pairwise[f"{a}__{b}"] = {"n": n, "kappa": kappa, "agreement": agree}
        pooled_pairs.extend(pairs)
        for k in shared:
            ann_type = by_reviewer_with_type[a][k][1]
            pooled_pairs_by_type[ann_type].append(
                (by_reviewer[a][k], by_reviewer[b][k])
            )

    pooled_kappa, pooled_n = cohens_kappa(pooled_pairs)
    by_type = {}
    for t, pairs in pooled_pairs_by_type.items():
        k, n = cohens_kappa(pairs)
        by_type[t] = {"n": n, "kappa": k}

    return {
        "pairwise": pairwise,
        "pooled": {"n": pooled_n, "kappa": pooled_kappa},
        "pooled_by_type": by_type,
    }


def build_v2_split() -> dict:
    """Stratified by (annotation_type, human_rating). Last-write-wins per (reviewer, key).
    No deduplication across reviewers -- each reviewer's rating is its own row."""
    all_rows = []
    for name in REVIEWERS:
        rows = load_reviewer(name)
        deduped = last_rating_per_key(rows)
        for r in deduped.values():
            if r["rating"] not in DONE_RATINGS:
                continue
            all_rows.append(
                {
                    "annotation_key": r["annotation_key"],
                    "annotation_type": r["annotation_type"],
                    "human_rating": r["rating"],
                    "reviewer": r["reviewer"],
                    "source_annotator_id": r["source_annotator_id"],
                    "split_version": "v2",
                    "transcript_id": r["transcript_id"],
                    "turn_number_start": r["turn_number_start"],
                    "turn_number_end": r["turn_number_end"],
                }
            )

    # Stratify by (annotation_type, human_rating)
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        by_stratum[(row["annotation_type"], row["human_rating"])].append(row)

    rng = random.Random(SEED)
    train, test = [], []
    for key, rows in sorted(by_stratum.items()):
        rng.shuffle(rows)
        n_test = max(1, round(len(rows) * TEST_RATIO)) if len(rows) > 1 else 0
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])

    # Stable sort by annotation_key for reproducibility
    train.sort(key=lambda r: r["annotation_key"])
    test.sort(key=lambda r: r["annotation_key"])

    EVAL.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: list[dict]) -> str:
        content = "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
        path.write_text(content, encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    train_sha = write_jsonl(EVAL / "labeller_train_v2.jsonl", train)
    test_sha = write_jsonl(EVAL / "labeller_test_v2.jsonl", test)

    snapshot = {
        name: len([r for r in load_reviewer(name) if r["rating"] in DONE_RATINGS])
        for name in REVIEWERS
    }

    meta = {
        "version": "v2",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "method": "stratified-by-annotation_type-and-human_rating",
        "test_ratio": TEST_RATIO,
        "input_ratings_snapshot": snapshot,
        "train_count": len(train),
        "test_count": len(test),
        "train_sha256": train_sha,
        "test_sha256": test_sha,
        "excluded_reviewers": ["test", "testlucy"],
    }
    (EVAL / "labeller_split_meta_v2.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    train_dist = Counter((r["annotation_type"], r["human_rating"]) for r in train)
    test_dist = Counter((r["annotation_type"], r["human_rating"]) for r in test)

    return {
        "meta": meta,
        "train_distribution": {f"{t}|{r}": c for (t, r), c in sorted(train_dist.items())},
        "test_distribution": {f"{t}|{r}": c for (t, r), c in sorted(test_dist.items())},
    }


def main() -> None:
    print("=" * 70)
    print("LABELLER VALIDATION ANALYSIS")
    print("=" * 70)

    print("\n--- Per-reviewer summary ---")
    summary = per_reviewer_summary()
    for name, info in summary.items():
        print(f"\n  {name}")
        print(f"    raw_rows         : {info['raw_rows']}")
        print(f"    unique keys      : {info['unique_keys']}")
        print(f"    done ratings     : {info['done_ratings']}")
        print(f"    ratings          : {info['ratings']}")
        print(f"    strategies       : {info['strategies']}")

    print("\n--- Cross-reviewer overlap + human ceiling ---")
    overlap = overlap_and_ceiling()
    print("\n  Pairwise (done ratings only):")
    for pair, info in overlap["pairwise"].items():
        if info["n"] == 0:
            print(f"    {pair:30s} n=0 (no overlap)")
        else:
            print(
                f"    {pair:30s} n={info['n']:4d}  "
                f"agreement={info['agreement']:.3f}  kappa={info['kappa']:.3f}"
            )
    pooled = overlap["pooled"]
    if pooled["n"]:
        print(
            f"\n  Pooled kappa (all overlapping pairs): "
            f"n={pooled['n']}, kappa={pooled['kappa']:.3f}"
        )
        for t, info in overlap["pooled_by_type"].items():
            print(f"    by type {t:12s}: n={info['n']:4d}  kappa={info['kappa']:.3f}")
    else:
        print(
            "\n  No overlapping annotation_keys across any reviewer pair -- "
            "cannot compute a human ceiling from this data."
        )

    print("\n--- v2 train/test split (Nathan included) ---")
    split = build_v2_split()
    print(json.dumps(split["meta"], indent=2))
    print("\n  train distribution by (annotation_type, human_rating):")
    for k, c in split["train_distribution"].items():
        print(f"    {k:40s} {c}")
    print("\n  test distribution by (annotation_type, human_rating):")
    for k, c in split["test_distribution"].items():
        print(f"    {k:40s} {c}")


if __name__ == "__main__":
    main()
