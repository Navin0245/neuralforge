"""
Shared pytest fixtures for neuralforge tests.

Fixtures defined here are available to all tests
without explicit import.
"""

import pytest
import torch


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def sample_1d_input(batch_size):
    """Sample 1D input tensor: (batch, n, dv)."""
    return torch.randn(batch_size, 64, 32)


@pytest.fixture
def sample_2d_input(batch_size):
    """Sample 2D input tensor: (batch, s1, s2, dv)."""
    return torch.randn(batch_size, 32, 32, 16)
