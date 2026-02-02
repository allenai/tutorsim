# tutor-bench Documentation

Welcome to **tutor-bench**, a benchmarking and evaluation framework for AI tutoring systems.

## Overview

tutor-bench provides a comprehensive framework for:
- Evaluating AI tutoring models and systems
- Running benchmarks on educational tasks
- Analyzing model performance across different educational domains
- Training and fine-tuning models for tutoring applications

## Quick Start

```bash
# Install tutor-bench
pip install tutor-bench

# Or install from source with development dependencies
pip install -e ".[dev]"
```

## Features

- **Modular Architecture**: Easily extensible framework for adding new benchmarks and evaluation metrics
- **Comprehensive Testing**: Built-in test suites for validation and benchmarking
- **Multiple Model Support**: Works with various transformer-based models
- **Rich Evaluation Metrics**: Educational domain-specific metrics and analysis tools
- **Async Processing**: Efficient parallel processing for large-scale evaluations
- **Type Safety**: Full type hints and mypy validation

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

usage
configuration
benchmarks
evaluation
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/modules
```

```{toctree}
:maxdepth: 2
:caption: Development

contributing
changelog
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
