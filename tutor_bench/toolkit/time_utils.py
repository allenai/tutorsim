"""Timestamp and tolerance helpers shared across data workflows."""

from __future__ import annotations

from collections.abc import Iterable


def ts_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm timestamp to seconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def nearest_within(target: float, candidates: Iterable[float], tolerance_seconds: float) -> float | None:
    """Return nearest candidate within tolerance, else None."""
    best: tuple[float, float] | None = None
    for c in candidates:
        d = abs(float(c) - target)
        if best is None or d < best[0]:
            best = (d, float(c))
    if best is None or best[0] > tolerance_seconds:
        return None
    return best[1]
