from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tutor_bench.toolkit.video_utils import extract_clip, extract_frame, get_video_duration


def test_get_video_duration_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    video.write_text("", encoding="utf-8")

    def fake_run(cmd, capture_output, text, timeout):  # type: ignore[no-untyped-def]
        assert "ffprobe" in cmd[0]
        return SimpleNamespace(returncode=0, stdout="12.345\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert abs(get_video_duration(video) - 12.345) < 1e-9


def test_get_video_duration_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    video.write_text("", encoding="utf-8")

    def fake_run(cmd, capture_output, text, timeout):  # type: ignore[no-untyped-def]
        return SimpleNamespace(returncode=1, stdout="", stderr="bad probe")

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(RuntimeError):
        get_video_duration(video)


def test_extract_clip_and_frame_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text, timeout):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    video = tmp_path / "in.mp4"
    clip = tmp_path / "clip.mp4"
    frame = tmp_path / "frame.jpg"
    video.write_text("", encoding="utf-8")

    extract_clip(video, clip_start=50.0, clip_duration=10.0, out_mp4=clip)
    extract_frame(clip, at_seconds=2.5, out_jpg=frame)

    assert len(calls) == 2
    assert calls[0][0] == "ffmpeg"
    assert calls[1][0] == "ffmpeg"
    assert str(clip) in calls[0]
    assert str(frame) in calls[1]
