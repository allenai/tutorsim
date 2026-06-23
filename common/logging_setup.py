"""Process-wide logging configuration for both pipelines.

Call `setup_logging()` once at process start to attach a console handler,
then again as `setup_logging(version)` once a run version is resolved to
attach a per-run file handler at `logs/{version}/run.log`.

Env vars:
    LOG_LEVEL      DEBUG | INFO | WARNING | ERROR (default: INFO)
    LOG_FILE       set to "0" to disable the per-run file handler
    LOG_REPO_ROOT  override repo-root resolution (test/CI use)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONSOLE_SENTINEL = "_logging_setup_console"
_FILE_SENTINEL = "_logging_setup_file"


def _repo_root() -> Path:
    override = os.environ.get("LOG_REPO_ROOT", "")
    return Path(override) if override else _REPO_ROOT_DEFAULT


def _resolve_level() -> int:
    name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(name)
    return level if isinstance(level, int) else logging.INFO


def _has_handler(root: logging.Logger, sentinel: str) -> bool:
    return any(getattr(h, sentinel, False) for h in root.handlers)


def setup_logging(version: str | None = None) -> None:
    """Configure root logger. Idempotent across calls.

    First call attaches a console handler (stderr) and sets the level
    from LOG_LEVEL. Subsequent calls leave the console handler alone.
    Passing `version` attaches a file handler at logs/{version}/run.log
    unless LOG_FILE=0 is set.
    """
    root = logging.getLogger()
    root.setLevel(_resolve_level())

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if not _has_handler(root, _CONSOLE_SENTINEL):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        setattr(console, _CONSOLE_SENTINEL, True)
        root.addHandler(console)

    if version is None:
        return
    if os.environ.get("LOG_FILE", "1") == "0":
        return
    if _has_handler(root, _FILE_SENTINEL):
        return

    log_dir = _repo_root() / "logs" / version
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    # Line buffering (encoding-aware) keeps `tail -f` responsive during long runs.
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    setattr(file_handler, _FILE_SENTINEL, True)
    root.addHandler(file_handler)
