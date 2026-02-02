"""Unit tests for core functionality."""

from pathlib import Path

import pytest


class TestCoreComponents:
    """Test core components of tutor-bench."""

    def test_import(self):
        """Test that the package can be imported."""
        import tutor_bench

        assert tutor_bench is not None

    def test_version(self):
        """Test version information."""
        from tutor_bench.version import VERSION, __version__

        assert __version__ == VERSION
        assert isinstance(__version__, str)

    def test_project_structure(self):
        """Test that expected directories exist."""
        project_root = Path(__file__).parent.parent
        expected_dirs = [
            "tutor_bench",
            "tests",
            "docs",
            "scripts",
        ]
        for dir_name in expected_dirs:
            assert (project_root / dir_name).exists(), f"Directory {dir_name} not found"

    @pytest.mark.parametrize(
        "config_file",
        [
            "pyproject.toml",
            "README.md",
            ".gitignore",
        ],
    )
    def test_config_files_exist(self, config_file):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent
        assert (project_root / config_file).exists(), f"Config file {config_file} not found"
