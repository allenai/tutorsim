# tutor-bench

[![CI](https://github.com/allenai/tutor-bench/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/tutor-bench/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/tutor-bench/badge/?version=latest)](https://tutor-bench.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/tutor-bench.svg)](https://badge.fury.io/py/tutor-bench)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

A comprehensive benchmarking and evaluation framework for AI tutoring systems.

## Features

- 🚀 **High Performance**: Async/parallel processing for efficient evaluation
- 📊 **Rich Metrics**: Educational domain-specific evaluation metrics
- 🔧 **Modular Design**: Easily extensible architecture for custom benchmarks
- 🔍 **Type Safe**: Full type hints with mypy validation
- 📝 **Well Documented**: Comprehensive documentation with Sphinx
- ✅ **Thoroughly Tested**: Extensive test coverage with pytest

## Benchmark Workflow

The tutor-bench benchmark follows a three-stage pipeline:

1. **Input Stage**: Users provide transcripts as JSONLines files following the OpenAI chat API format
2. **Annotation Stage**: An annotation script processes input files and produces separate JSONLines files containing annotations (both span-level and transcript-level)
3. **Evaluation Stage**: An evaluation script reads both the original input files and annotation files to produce metrics

### Input Format

Transcripts should be in JSONLines format (.jsonl) where each line is a JSON object representing a conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful tutor."},
    {"role": "user", "content": "Can you explain photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is the process..."}
  ],
  "metadata": {
    "session_id": "abc123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Annotation Format

Annotations are stored separately in JSONLines format:

```json
{
  "session_id": "abc123",
  "annotations": {
    "span_level": [
      {
        "message_index": 2,
        "start": 0,
        "end": 15,
        "label": "definition",
        "confidence": 0.95
      }
    ],
    "transcript_level": {
      "quality_score": 0.87,
      "pedagogical_effectiveness": "high",
      "concepts_covered": ["photosynthesis", "chlorophyll", "light reactions"]
    }
  }
}
```

### Basic API

```python
from tutor_bench import Annotator, Evaluator

# Annotate transcripts
annotator = Annotator(model="gpt-4")
annotations = annotator.process_transcripts("path/to/transcripts.jsonl")
annotations.save("path/to/annotations.jsonl")

# Evaluate with annotations
evaluator = Evaluator()
metrics = evaluator.evaluate(
    transcripts="path/to/transcripts.jsonl",
    annotations="path/to/annotations.jsonl"
)
print(metrics.summary())
```

## Quick Start

### Installation

```bash
# Install from PyPI (when available)
pip install tutor-bench

# Or install from source
git clone https://github.com/allenai/tutor-bench.git
cd tutor-bench
pip install -e .
```

### Basic Usage

```python
import tutor_bench

# Initialize a benchmark
benchmark = tutor_bench.Benchmark()

# Load your model
model = tutor_bench.load_model("model_name")

# Run evaluation
results = benchmark.evaluate(model)

# View results
print(results.summary())
```

### Running the Demo Pipeline

Try out the complete workflow with mock data:

```bash
# Run the entire demo pipeline
python scripts/run_demo.py

# Or run each step individually:

# 1. Generate sample transcripts
python scripts/generate_sample_transcripts.py

# 2. Annotate transcripts
python scripts/annotate.py data/sample_transcripts/mixed_transcripts.jsonl

# 3. Evaluate with annotations
python scripts/evaluate.py \
    data/sample_transcripts/mixed_transcripts.jsonl \
    data/sample_transcripts/mixed_transcripts_annotations.jsonl
```

### Command-Line Tools

The benchmark provides command-line scripts for each stage:

```bash
# Annotation script
python scripts/annotate.py --help

# Evaluation script
python scripts/evaluate.py --help
```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Make (for running development commands)
- Git

### Setup Development Environment

1. Clone the repository:
```bash
git clone https://github.com/allenai/tutor-bench.git
cd tutor-bench
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
make install-dev
```

### Development Workflow

```bash
# Run all checks (format, lint, test)
make run-checks

# Format code
make format

# Run tests
make test

# Run tests with coverage
make test-cov

# Build documentation
make docs

# Serve documentation locally
make serve-docs

# Clean build artifacts
make clean
```

## Project Structure

```
tutor-bench/
├── tutor_bench/         # Main package code
│   ├── configs/         # Configuration files
│   ├── data/           # Data processing utilities
│   ├── evaluation/     # Evaluation metrics and runners
│   ├── models/         # Model implementations
│   └── prompts/        # Prompt templates and utilities
├── tests/              # Test suite
│   ├── fixtures/       # Test fixtures and data
│   └── unit_*.py      # Unit tests
├── docs/               # Documentation
│   └── source/        # Sphinx documentation source
├── scripts/           # Utility scripts
└── .github/           # GitHub Actions workflows
```

## Documentation

Full documentation is available at [https://tutor-bench.readthedocs.io](https://tutor-bench.readthedocs.io)

To build documentation locally:

```bash
make docs
make serve-docs  # View at http://localhost:8000
```

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run fast tests only
make test-fast

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/unit_test_core.py -v
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and checks (`make run-checks`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project follows best practices from the [Allen Institute for AI](https://allenai.org/) and is inspired by the excellent engineering in projects like [olmocr](https://github.com/allenai/olmocr).

## Contact

- **Author**: Kyle Lo
- **Email**: kylel@allenai.org
- **Issues**: [GitHub Issues](https://github.com/allenai/tutor-bench/issues)
