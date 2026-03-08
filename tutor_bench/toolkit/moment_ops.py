"""Timestamp matching helpers for moment-set comparisons."""

from __future__ import annotations

from typing import Any


def unmatched_timestamps(source: list[float], reference: list[float], tolerance_seconds: float) -> list[float]:
    out: list[float] = []
    for ts in source:
        if not any(abs(ts - r) <= tolerance_seconds for r in reference):
            out.append(ts)
    return out


def pairwise_overlap(a: list[float], b: list[float], tolerance_seconds: float) -> dict[str, Any]:
    def covered(src: list[float], other: list[float]) -> int:
        return sum(1 for x in src if any(abs(x - y) <= tolerance_seconds for y in other))

    n_a = len(a)
    n_b = len(b)
    cov_a = covered(a, b)
    cov_b = covered(b, a)
    return {
        "n_a": n_a,
        "n_b": n_b,
        "covered_a": cov_a,
        "covered_b": cov_b,
        "recall_a_to_b": (cov_a / n_a) if n_a else 0.0,
        "recall_b_to_a": (cov_b / n_b) if n_b else 0.0,
    }
