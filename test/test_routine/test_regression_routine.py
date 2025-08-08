
import pytest
import torch
from torch import nn
from unittest.mock import Mock
from torchmetrics import MeanAbsoluteError

from chemtorch.core.routine.regression_routine import RegressionRoutine


def test_forward_squeeze_edge_cases(simple_model):
    """Test edge cases in forward method squeezing logic."""
    
    # Test 0-D tensor (scalar) - should remain unchanged
    class ScalarModel(nn.Module):
        def forward(self, x):
            return torch.tensor(5.0)  # Scalar tensor

    routine = RegressionRoutine(model=ScalarModel())
    inputs = torch.randn(1, 10)
    
    output = routine.forward(inputs)
    assert output.shape == torch.Size([])  # Scalar shape
    
    # Test 1-D tensor - should remain unchanged
    class OneDModel(nn.Module):
        def forward(self, x):
            return torch.randn(4)  # Already 1-D

    routine = RegressionRoutine(model=OneDModel())
    output = routine.forward(inputs)
    assert output.shape == (4,)
    
    # Test tensor with multiple trailing dimensions of size 1
    class MultipleOnesModel(nn.Module):
        def forward(self, x):
            return torch.randn(4, 1, 1, 1)  # Shape: (4, 1, 1, 1)

    routine = RegressionRoutine(model=MultipleOnesModel())
    output = routine.forward(inputs)
    # Should only squeeze the last dimension: (4, 1, 1)
    assert output.shape == (4, 1, 1)

def test_forward_multidimensional_squeeze(simple_model):
    """Test that forward properly squeezes multi-dimensional outputs."""
    # Create model that outputs multi-dimensional tensor
    class MultiDimModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)  # Shape: (batch, 1)
    
    model = MultiDimModel()
    routine = RegressionRoutine(model=model)
    inputs = torch.randn(4, 10)
    
    output = routine.forward(inputs)
    
    assert output.shape == (4,)  # Should be squeezed from (4, 1)
    

def test_step_with_metrics(simple_model, loss_function, sample_batch):
    """Test that metrics are properly updated during steps."""
    metrics = {"train": MeanAbsoluteError()}
    routine = RegressionRoutine(
        model=simple_model,
        loss=loss_function,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        metrics=metrics
    )
    
    # Mock the log method to verify it's called
    routine.log = Mock()
    routine.log_dict = Mock()
    
    loss = routine.training_step(sample_batch)
    
    # Verify loss is logged
    routine.log.assert_called()
    
    # Verify metric is updated (check that it has been called)
    assert routine.train_metrics.compute() is not None
