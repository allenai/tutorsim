# Installation

## Requirements

- Python 3.11 or higher
- PyTorch 2.0 or higher
- CUDA 11.8+ (optional, for GPU support)

## Install from PyPI

```bash
pip install tutor-bench
```

## Install from Source

### Basic Installation

```bash
git clone https://github.com/allenai/tutor-bench.git
cd tutor-bench
pip install -e .
```

### Development Installation

For development, install with all optional dependencies:

```bash
git clone https://github.com/allenai/tutor-bench.git
cd tutor-bench
pip install -e ".[dev]"
```

This installs additional tools for:
- Testing (pytest, coverage)
- Code formatting (black, isort)
- Linting (ruff, mypy)
- Documentation (sphinx, furo)

### Optional Dependencies

Install specific feature sets:

```bash
# For evaluation and visualization tools
pip install -e ".[eval]"

# For model training capabilities
pip install -e ".[train]"

# For everything
pip install -e ".[all]"
```

## Verify Installation

```python
import tutor_bench
print(tutor_bench.version.__version__)
```

## GPU Support

To enable GPU acceleration, ensure you have:

1. NVIDIA GPU with CUDA capability
2. CUDA toolkit installed
3. PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:

```python
import torch
print(torch.cuda.is_available())
```
