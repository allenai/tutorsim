# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup with modern Python packaging standards
- Comprehensive testing framework with pytest
- Documentation with Sphinx and MyST parser
- CI/CD workflows with GitHub Actions
- Code quality tools: black, isort, ruff, mypy
- Makefile for common development tasks
- Project structure following best practices from olmocr

### Changed
- Migrated from setuptools-only to modern pyproject.toml configuration
- Updated Python requirement to 3.11+
- Replaced flake8 with ruff for faster linting

### Removed
- Legacy configuration files (setup.py, requirements.txt approach)
- Separate pytest.ini, mypy.ini, .flake8 files (consolidated into pyproject.toml)

## [0.1.0] - 2024-01-01

### Added
- Initial project template

[Unreleased]: https://github.com/allenai/tutor-bench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/allenai/tutor-bench/releases/tag/v0.1.0
