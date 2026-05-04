"""File I/O helpers for shared experiment artifacts.

Paths are resolved through `resolve_path`, which honors the `STORAGE_ROOT`
environment variable. When set, relative paths are joined onto the root, which
may be a local filesystem path or a remote URL such as ``s3://bucket/prefix``.
Absolute local paths and explicit URLs (anything with a scheme) bypass the
root and are used as-is.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from upath import UPath

PathLike = str | os.PathLike[str]


def _has_scheme(value: str) -> bool:
    head, sep, _ = value.partition("://")
    return bool(sep) and head.isalpha()


def resolve_path(path: PathLike) -> UPath:
    """Return a `UPath` for `path`, joined onto `STORAGE_ROOT` if applicable.

    - If `path` is an absolute local path or carries a URL scheme, it is
      returned unchanged.
    - Otherwise, if `STORAGE_ROOT` is set, the result is `STORAGE_ROOT / path`.
    - With no root and a relative path, the path is returned as-is (cwd-relative).
    """
    raw = os.fspath(path)
    if _has_scheme(raw) or Path(raw).is_absolute():
        return UPath(raw)
    root = os.environ.get("STORAGE_ROOT")
    if not root:
        return UPath(raw)
    return UPath(root) / raw


def read_stems_file(path: PathLike, max_stems: int = 0) -> list[str]:
    p = resolve_path(path)
    stems = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            stems.append(s)
    if max_stems > 0:
        stems = stems[:max_stems]
    return stems


def save_jsonl(path: PathLike, rows: list[dict[str, Any]]) -> None:
    p = resolve_path(path)
    parent = p.parent
    if parent != p:
        parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_jsonl(path: PathLike) -> list[dict[str, Any]]:
    p = resolve_path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows
