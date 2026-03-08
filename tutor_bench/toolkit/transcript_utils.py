"""Gold transcript parsing and context extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tutor_bench.toolkit.time_utils import fmt_ts, ts_to_seconds

LINE_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(\w+):\s*(.*)")


@dataclass
class Segment:
    start: float
    end: float
    role: str
    text: str


def parse_gold_transcript(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        segments.append(
            Segment(
                start=ts_to_seconds(match.group(1)),
                end=ts_to_seconds(match.group(2)),
                role=match.group(3),
                text=match.group(4).strip(),
            )
        )
    return segments


def transcript_lines(segments: list[Segment]) -> list[str]:
    return [f"[{fmt_ts(s.start)} - {fmt_ts(s.end)}] {s.role}: {s.text}" for s in segments]


def context_window(segments: list[Segment], center_t: float, pre_s: float, post_s: float) -> str:
    lo = center_t - pre_s
    hi = center_t + post_s
    lines: list[str] = []
    for seg in segments:
        if seg.end >= lo and seg.start <= hi:
            lines.append(f"[{fmt_ts(seg.start)} - {fmt_ts(seg.end)}] {seg.role}: {seg.text}")
    return "\n".join(lines)


def context_prefix(segments: list[Segment], center_t: float, pre_s: float) -> str:
    lo = center_t - pre_s
    lines: list[str] = []
    for seg in segments:
        if seg.end >= lo and seg.start <= center_t:
            lines.append(f"[{fmt_ts(seg.start)} - {fmt_ts(seg.end)}] {seg.role}: {seg.text}")
    return "\n".join(lines)
