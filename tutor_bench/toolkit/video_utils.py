"""Video probing and extraction helpers built on ffprobe/ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_video_duration(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"ffprobe duration failed: {proc.stderr[:300]}")
    return float(proc.stdout.strip())


def extract_clip(video: Path, clip_start: float, clip_duration: float, out_mp4: Path) -> None:
    fast_seek = max(0.0, clip_start - 30.0)
    precise_seek = clip_start - fast_seek
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{fast_seek:.3f}",
        "-i",
        str(video),
        "-ss",
        f"{precise_seek:.3f}",
        "-t",
        f"{clip_duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg clip extraction failed: {proc.stderr[:400]}")


def extract_frame(video_or_clip: Path, at_seconds: float, out_jpg: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, at_seconds):.3f}",
        "-i",
        str(video_or_clip),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_jpg),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr[:400]}")
