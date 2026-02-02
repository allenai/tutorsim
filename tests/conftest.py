"""Pytest configuration and fixtures for tutor-bench tests."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
import torch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_data_dir(tmp_path) -> Path:
    """Create a temporary directory with sample data for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""

    class MockModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def predict(self, text: str) -> str:
            return f"Mock prediction for: {text}"

        def evaluate(self, data):
            return {"accuracy": 0.95, "loss": 0.05}

    return MockModel()


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def gpu_available():
    """Check if GPU is available for tests."""
    return torch.cuda.is_available()


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Add unit marker to all tests in test files starting with "unit_"
        if "unit_" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Add integration marker to all tests in test files starting with "integration_"
        if "integration_" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Skip GPU tests if no GPU is available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="GPU not available")
            item.add_marker(skip_gpu)
