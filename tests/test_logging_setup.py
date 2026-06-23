"""Tests for common.logging_setup — idempotency, two-phase init, env-var gating."""
import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Reset the root logger between tests so handlers don't leak."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers.clear()
    yield
    root.handlers.clear()
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def _our_console_handlers():
    """Filter to handlers added by setup_logging (ignores pytest's caplog)."""
    from common.logging_setup import _CONSOLE_SENTINEL
    return [h for h in logging.getLogger().handlers
            if getattr(h, _CONSOLE_SENTINEL, False)]


def _our_file_handlers():
    from common.logging_setup import _FILE_SENTINEL
    return [h for h in logging.getLogger().handlers
            if getattr(h, _FILE_SENTINEL, False)]


@pytest.fixture
def in_tmp_repo(tmp_path, monkeypatch):
    """Run setup_logging with the repo root pointed at a temp dir.

    The file handler writes under <repo_root>/logs, so we override the
    resolution by setting the env var the module reads.
    """
    monkeypatch.setenv("LOG_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("LOG_FILE", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    return tmp_path


class TestConsoleHandler:
    def test_first_call_adds_console_handler(self, in_tmp_repo):
        from common.logging_setup import setup_logging
        setup_logging()
        assert len(_our_console_handlers()) == 1

    def test_idempotent_no_duplicate_console_handler(self, in_tmp_repo):
        from common.logging_setup import setup_logging
        setup_logging()
        setup_logging()
        setup_logging()
        assert len(_our_console_handlers()) == 1


class TestFileHandler:
    def test_no_file_handler_without_version(self, in_tmp_repo):
        from common.logging_setup import setup_logging
        setup_logging()
        assert _our_file_handlers() == []

    def test_file_handler_attached_when_version_given(self, in_tmp_repo):
        from common.logging_setup import setup_logging
        setup_logging(version="testrun")
        handlers = _our_file_handlers()
        assert len(handlers) == 1
        log_path = Path(handlers[0].baseFilename)
        assert log_path == in_tmp_repo / "logs" / "testrun" / "run.log"
        assert log_path.parent.exists()

    def test_two_phase_init_upgrades_to_file_handler(self, in_tmp_repo):
        from common.logging_setup import setup_logging
        setup_logging()
        setup_logging(version="phase2")
        assert len(_our_file_handlers()) == 1

    def test_log_file_disabled_via_env(self, in_tmp_repo, monkeypatch):
        monkeypatch.setenv("LOG_FILE", "0")
        from common.logging_setup import setup_logging
        setup_logging(version="disabled")
        assert _our_file_handlers() == []

    def test_file_handler_writes_log_record(self, in_tmp_repo):
        from common.logging_setup import setup_logging
        setup_logging(version="writes")
        logging.getLogger("test.module").warning("hello world")
        for h in logging.getLogger().handlers:
            h.flush()
        log_path = in_tmp_repo / "logs" / "writes" / "run.log"
        contents = log_path.read_text(encoding="utf-8")
        assert "hello world" in contents
        assert "test.module" in contents
        assert "WARNING" in contents


class TestLevel:
    def test_default_level_is_info(self, in_tmp_repo):
        from common.logging_setup import setup_logging
        setup_logging()
        assert logging.getLogger().level == logging.INFO

    def test_log_level_env_debug(self, in_tmp_repo, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        from common.logging_setup import setup_logging
        setup_logging()
        assert logging.getLogger().level == logging.DEBUG

    def test_log_level_env_warning(self, in_tmp_repo, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        from common.logging_setup import setup_logging
        setup_logging()
        assert logging.getLogger().level == logging.WARNING

    def test_log_level_invalid_falls_back_to_info(self, in_tmp_repo, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "BANANA")
        from common.logging_setup import setup_logging
        setup_logging()
        assert logging.getLogger().level == logging.INFO
