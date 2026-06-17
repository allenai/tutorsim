"""Aggregate benchmark scores across multiple run dirs into a leaderboard.

Scans results/benchmark/ for run dirs matching --pattern, reads each run's
scores/{profile}.json, and emits a side-by-side comparison as both a
Markdown table (for Slack/PR) and a CSV (for Excel/pivot).

Usage:
    # All v10 runs on 2026-06-16
    python scripts/leaderboard.py --pattern "*_v10_*_20260616"

    # All runs for a specific tutor mode across all dates
    python scripts/leaderboard.py --pattern "*_scaffolding_rigor_tutor_*"

    # Write to a specific output file
    python scripts/leaderboard.py --pattern "*_v10_*" --out leaderboard.md

The output preserves enough columns that you can recover any derived metric
without re-running annotation (n_clean_yes, n_overscaffold, n_total per axis).
"""
import argparse
import csv
import fnmatch
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _parse_run_dir_name(name: str) -> dict:
    """Recover (tutor_model, prompt_version, tutor_mode, student_mode, date)
    from a run dir like
    'claude-opus-4-8_v10_scaffolding_rigor_tutor_oracle_student_20260616'.

    Naming scheme is
    '{tutor_model}_{prompt_version}_{tutor_mode}_tutor_{student_mode}_student_{date}'.
    Splits on '_tutor_' (giving the tutor_mode boundary) and '_student_'
    (giving the student_mode boundary). Robust to model IDs with dashes.
    """
    out = {"name": name, "tutor_model": "?", "prompt_version": "?",
           "tutor_mode": "?", "student_mode": "?", "date": "?"}
    if "_tutor_" not in name or "_student_" not in name:
        return out

    head, tail = name.split("_tutor_", 1)
    # head = '{tutor_model}_{prompt_version}_{tutor_mode}'
    # tail = '{student_mode}_student_{date}'
    student_mode, date = tail.split("_student_", 1)
    out["student_mode"] = student_mode
    out["date"] = date

    head_parts = head.split("_")
    # First v\d+ token in head = prompt_version. Anything before = tutor_model,
    # anything after = tutor_mode.
    pv_idx = None
    for i, p in enumerate(head_parts):
        if p.startswith("v") and p[1:].isdigit():
            pv_idx = i
            break
    if pv_idx is not None:
        out["tutor_model"] = "_".join(head_parts[:pv_idx])
        out["prompt_version"] = head_parts[pv_idx]
        out["tutor_mode"] = "_".join(head_parts[pv_idx + 1:]) or "?"
    else:
        out["tutor_model"] = head

    return out


def _load_scores(run_dir: Path, profile: str) -> dict | None:
    scores_path = run_dir / "scores" / f"{profile}.json"
    if not scores_path.exists():
        return None
    with open(scores_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, places=3):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{places}f}"
    return str(v)


def _row(meta: dict, scores: dict) -> dict:
    cal_s = scores.get("scaffold_calibrated", {}) or {}
    cal_r = scores.get("rigor_calibrated", {}) or {}
    did_s = scores.get("scaffolding_did", {}) or {}
    did_r = scores.get("rigor_did", {}) or {}
    over = scores.get("overscaffold", {}) or {}
    timings = scores.get("timings", {}) or {}
    tokens = scores.get("tokens", {}) or {}
    tk_total = (tokens.get("total") or {}).get("total_tokens")
    tk_tutor = (tokens.get("tutor") or {}).get("total_tokens")
    tk_student = (tokens.get("student") or {}).get("total_tokens")
    tk_ann = (tokens.get("annotation") or {}).get("total_tokens")
    latency = scores.get("latency", {}) or {}
    tutor_lat = latency.get("tutor") or {}
    student_lat = latency.get("student") or {}
    return {
        "tutor_model": meta["tutor_model"],
        "tutor_mode": meta["tutor_mode"],
        "prompt_version": meta["prompt_version"],
        "student_mode": meta["student_mode"],
        "date": meta["date"],
        "n_total": scores.get("n_scenarios", 0),
        # Headline calibrated metrics
        "scaffold_calibrated": cal_s.get("score"),
        "rigor_calibrated": cal_r.get("score"),
        # Components for recoverability
        "scaffold_clean_yes": cal_s.get("n_clean_yes"),
        "scaffold_overscaffold": cal_s.get("n_overscaffold"),
        "scaffold_moments": cal_s.get("n_total"),
        "rigor_clean_yes": cal_r.get("n_clean_yes"),
        "rigor_moments": cal_r.get("n_total"),
        # Diagnostic: legacy did-rates + over-scaffold rate + outcome+
        "did_scaffold_rate": did_s.get("rate"),
        "did_rigor_rate": did_r.get("rate"),
        "overscaffold_rate": over.get("rate"),
        "outcome_pos_rate": scores.get("outcome_pos_rate"),
        # Latency + tokens
        "phase1_seconds": timings.get("phase1_exchange_seconds"),
        "phase2_seconds": timings.get("phase2_annotate_seconds"),
        "total_seconds": timings.get("total_seconds"),
        "tokens_total": tk_total,
        "tokens_tutor": tk_tutor,
        "tokens_student": tk_student,
        "tokens_annotation": tk_ann,
        # Per-call latency (only meaningful in sync mode)
        "tutor_lat_mean": tutor_lat.get("mean_seconds"),
        "tutor_lat_p50": tutor_lat.get("p50_seconds"),
        "tutor_lat_p95": tutor_lat.get("p95_seconds"),
        "student_lat_mean": student_lat.get("mean_seconds"),
        "student_lat_p50": student_lat.get("p50_seconds"),
        "student_lat_p95": student_lat.get("p95_seconds"),
        "run_dir": meta["name"],
    }


def _markdown_table(rows: list[dict]) -> str:
    cols = [
        ("tutor_model", "tutor_model"),
        ("tutor_mode", "mode"),
        ("n_total", "n"),
        ("scaffold_calibrated", "scaffold_cal (higher better)"),
        ("rigor_calibrated", "rigor_cal (higher better)"),
        ("overscaffold_rate", "over_rate (lower better)"),
        ("outcome_pos_rate", "outcome_pos (higher better)"),
        ("did_scaffold_rate", "did_scaf"),
        ("did_rigor_rate", "did_rig"),
        ("tutor_lat_p50", "tutor_p50_sec (lower better)"),
        ("tutor_lat_p95", "tutor_p95_sec (lower better)"),
        ("tokens_total", "tokens_total"),
    ]
    header = "| " + " | ".join(label for _, label in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            elif isinstance(v, int):
                cells.append(str(v))
            elif v is None:
                cells.append("-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="*",
                   help="Glob pattern matched against run dir name under "
                        "results/benchmark/. Default '*' matches everything.")
    p.add_argument("--benchmark-root", default="results/benchmark",
                   help="Directory to scan for run dirs.")
    p.add_argument("--profile", default="anthropic",
                   help="Scores file to read inside each run dir "
                        "(scores/{profile}.json). Default: anthropic.")
    p.add_argument("--sort-by", default="scaffold_calibrated",
                   help="Column to sort the leaderboard by (descending).")
    p.add_argument("--out-md", default=None,
                   help="Optional: write markdown table to this path.")
    p.add_argument("--out-csv", default=None,
                   help="Optional: write CSV to this path.")
    args = p.parse_args()

    root = Path(args.benchmark_root)
    if not root.exists():
        raise SystemExit(f"No such directory: {root}")

    rows = []
    skipped = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not fnmatch.fnmatch(child.name, args.pattern):
            continue
        if child.name.startswith("_"):
            continue   # skip _viewers_* aggregator dirs etc.
        scores = _load_scores(child, args.profile)
        if scores is None:
            skipped.append(child.name)
            continue
        # Skip empty/failed runs (n_scenarios == 0) so stale dirs don't
        # pollute the leaderboard.
        if not scores.get("n_scenarios"):
            skipped.append(child.name + " (n=0)")
            continue
        meta = _parse_run_dir_name(child.name)
        rows.append(_row(meta, scores))

    # Sort descending; None goes last.
    def _key(r):
        v = r.get(args.sort_by)
        return (v is None, -(v if isinstance(v, (int, float)) else 0))
    rows.sort(key=_key)

    md = _markdown_table(rows)
    print(md)
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
        logger.info("Wrote %s", args.out_md)
    if args.out_csv:
        _write_csv(rows, Path(args.out_csv))
        logger.info("Wrote %s", args.out_csv)

    if skipped:
        logger.info("Skipped %d run dirs (no scores/%s.json): %s",
                    len(skipped), args.profile,
                    ", ".join(skipped[:5]) + (" …" if len(skipped) > 5 else ""))


if __name__ == "__main__":
    main()
