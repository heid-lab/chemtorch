"""
Shared fixtures for routine tests.

This module contains fixtures that are used across multiple test files
in the routine test directory.
"""

import pytest
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection


class SimpleModel(nn.Module):
    """Simple model for testing purposes."""
    
    def __init__(self, input_size: int = 10, output_size: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Fixture providing a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_model_class():
    """Fixture providing the SimpleModel class for testing."""
    return SimpleModel


@pytest.fixture
def loss_function():
    """Fixture providing a loss function for testing."""
    return nn.MSELoss()


@pytest.fixture
def sample_batch():
    """Fixture providing a sample batch for testing."""
    inputs = torch.randn(4, 10)
    targets = torch.randn(4)
    return inputs, targets


@pytest.fixture
def metrics():
    """Fixture providing sample metrics for testing."""
    return {
        "train": MeanAbsoluteError(),
        "val": MetricCollection({"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}),
        "test": MeanSquaredError()
    }
