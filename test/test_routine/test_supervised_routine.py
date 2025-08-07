"""
Comprehensive tests for SupervisedRoutine class.

This module contains all tests for the SupervisedRoutine Lightning module,
including core functionality, configuration, and integration tests.
"""

from unittest.mock import Mock
import pytest
import torch
import torch.nn as nn
import tempfile
import os
from functools import partial

from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from chemtorch.core.routine.supervised_routine import SupervisedRoutine


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


class TestSupervisedRoutineCore:
    """Test core Lightning functionality."""
    
    def test_initialization_minimal(self, simple_model):
        """Test minimal initialization for inference only."""
        routine = SupervisedRoutine(model=simple_model)
        
        assert routine.model is simple_model
        assert routine.loss is None
        assert routine.optimizer_factory is None
        assert routine.lr_scheduler_config is None
        assert routine.metrics is None
    
    def test_initialization_full(self, simple_model, loss_function, metrics):
        """Test full initialization for training."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
            metrics=metrics
        )
        
        assert routine.model is simple_model
        assert routine.loss is loss_function
        assert routine.optimizer_factory is not None
        assert routine.lr_scheduler_config is not None
        assert routine.metrics is not None
        assert len(routine.metrics) == 3
    
    def test_forward(self, simple_model, sample_batch):
        """Test forward pass."""
        routine = SupervisedRoutine(model=simple_model)
        inputs, _ = sample_batch
        
        output = routine.forward(inputs)
        
        assert output.shape == (4,)  # Should be squeezed
        assert isinstance(output, torch.Tensor)
    
    def test_forward_multidimensional_squeeze(self, simple_model):
        """Test that forward properly squeezes multi-dimensional outputs."""
        # Create model that outputs multi-dimensional tensor
        class MultiDimModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)  # Shape: (batch, 1)
        
        model = MultiDimModel()
        routine = SupervisedRoutine(model=model)
        inputs = torch.randn(4, 10)
        
        output = routine.forward(inputs)
        
        assert output.shape == (4,)  # Should be squeezed from (4, 1)
    
    def test_training_step(self, simple_model, loss_function, sample_batch):
        """Test training step."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        
        loss = routine.training_step(sample_batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_validation_step(self, simple_model, loss_function, sample_batch):
        """Test validation step."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        
        loss = routine.validation_step(sample_batch)
        
        assert isinstance(loss, torch.Tensor)
    
    def test_test_step(self, simple_model, loss_function, sample_batch):
        """Test test step."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        
        loss = routine.test_step(sample_batch)
        
        assert isinstance(loss, torch.Tensor)
    
    def test_step_with_metrics(self, simple_model, loss_function, sample_batch):
        """Test that metrics are properly updated during steps."""
        metrics = {"train": MeanAbsoluteError()}
        routine = SupervisedRoutine(
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


class TestSetupMethod:
    """Test the setup method for different stages."""
    
    def test_setup_training_without_loss(self, simple_model):
        """Test setup fails for training without loss."""
        routine = SupervisedRoutine(model=simple_model)
        
        with pytest.raises(ValueError, match="Loss function must be defined for training"):
            routine.setup(stage="fit")
    
    def test_setup_training_without_optimizer(self, simple_model, loss_function):
        """Test setup fails for training without optimizer."""
        routine = SupervisedRoutine(model=simple_model, loss=loss_function)
        
        with pytest.raises(ValueError, match="Optimizer must be defined for training"):
            routine.setup(stage="fit")
    
    def test_setup_validation_without_loss(self, simple_model):
        """Test setup fails for validation without loss."""
        routine = SupervisedRoutine(model=simple_model)
        
        with pytest.raises(ValueError, match="Loss function must be defined for training"):
            routine.setup(stage="validate")
    
    def test_setup_predict_no_requirements(self, simple_model):
        """Test setup succeeds for prediction without loss/optimizer."""
        routine = SupervisedRoutine(model=simple_model)
        
        # Should not raise any exception
        routine.setup(stage="predict")


class TestMetricsInitialization:
    """Test metrics initialization and handling."""
    
    def test_single_metric_initialization(self, simple_model, loss_function):
        """Test initialization with a single metric."""
        metric = MeanAbsoluteError()
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics=metric
        )
        
        assert len(routine.metrics) == 3
        assert "train" in routine.metrics
        assert "val" in routine.metrics
        assert "test" in routine.metrics
        
        # Check that metrics are properly prefixed
        assert hasattr(routine, "train_metrics")
        assert hasattr(routine, "val_metrics")
        assert hasattr(routine, "test_metrics")
    
    def test_metric_collection_initialization(self, simple_model, loss_function):
        """Test initialization with a MetricCollection."""
        metrics = MetricCollection({"mae": MeanAbsoluteError(), "mse": MeanSquaredError()})
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics=metrics
        )
        
        assert len(routine.metrics) == 3
        assert isinstance(routine.train_metrics, MetricCollection)
        assert isinstance(routine.val_metrics, MetricCollection)
        assert isinstance(routine.test_metrics, MetricCollection)
    
    def test_dict_metrics_initialization(self, simple_model, loss_function, metrics):
        """Test initialization with a dictionary of metrics."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics=metrics
        )
        
        assert len(routine.metrics) == 3
        assert hasattr(routine, "train_metrics")
        assert hasattr(routine, "val_metrics") 
        assert hasattr(routine, "test_metrics")
    
    def test_invalid_metrics_type(self, simple_model, loss_function):
        """Test that invalid metrics type raises TypeError."""
        with pytest.raises(TypeError, match="Metrics must be a torchmetrics.Metric"):
            SupervisedRoutine(
                model=simple_model,
                loss=loss_function,
                optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                metrics="invalid"
            )
    
    def test_invalid_metrics_dict_keys(self, simple_model, loss_function):
        """Test that invalid metric dict keys raise ValueError."""
        invalid_metrics = {"invalid_key": MeanAbsoluteError()}
        
        with pytest.raises(ValueError, match="Metric dictionary keys must be any of"):
            SupervisedRoutine(
                model=simple_model,
                loss=loss_function,
                optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                metrics=invalid_metrics
            )


class TestOptimizerConfiguration:
    """Test optimizer configuration."""
    
    def test_optimizer_only(self, simple_model, loss_function):
        """Test configuration with optimizer only."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        
        config = routine.configure_optimizers()
        
        assert isinstance(config, torch.optim.Adam)
        assert config.param_groups[0]['lr'] == 1e-3
    
    def test_none_optimizer(self, simple_model):
        """Test that None optimizer returns None."""
        routine = SupervisedRoutine(model=simple_model, optimizer=None)
        
        config = routine.configure_optimizers()
        assert config is None
    
    def test_optimizer_with_factory_scheduler(self, simple_model, loss_function):
        """Test optimizer with factory function scheduler."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        )
        
        config = routine.configure_optimizers()
        
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert isinstance(config["optimizer"], torch.optim.Adam)
        assert isinstance(config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR)
    
    def test_optimizer_with_lightning_scheduler_config(self, simple_model, loss_function):
        """Test optimizer with Lightning scheduler configuration."""
        lr_scheduler_config = {
            "scheduler": partial(torch.optim.lr_scheduler.StepLR, step_size=10),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            lr_scheduler=lr_scheduler_config
        )
        
        config = routine.configure_optimizers()
        
        assert isinstance(config, dict)
        lr_scheduler = config["lr_scheduler"]
        assert "scheduler" in lr_scheduler
        assert "interval" in lr_scheduler
        assert "frequency" in lr_scheduler
        assert "monitor" in lr_scheduler
        assert lr_scheduler["interval"] == "epoch"
        assert lr_scheduler["monitor"] == "val_loss"
    
    def test_invalid_scheduler_config(self, simple_model, loss_function):
        """Test invalid scheduler configuration raises error."""
        with pytest.raises(TypeError, match="LR scheduler must be callable or dict"):
            routine = SupervisedRoutine(
                model=simple_model,
                loss=loss_function,
                optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                lr_scheduler="invalid"
            )
            routine.configure_optimizers()
    
    def test_missing_scheduler_key(self, simple_model, loss_function):
        """Test missing scheduler key in config dict raises error."""
        lr_scheduler_config = {"interval": "epoch"}  # Missing 'scheduler' key
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            lr_scheduler=lr_scheduler_config
        )
        
        with pytest.raises(ValueError, match="LR scheduler config dictionary must contain 'scheduler' key"):
            routine.configure_optimizers()


class TestCheckpointHandling:
    """Test checkpoint loading functionality."""
    
    def test_load_pretrained_model_only(self, simple_model, loss_function):
        """Test loading pretrained model without resuming training."""
        # Create a temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {
                'model_state_dict': simple_model.state_dict()
            }
            torch.save(checkpoint, f.name)
            
            try:
                routine = SupervisedRoutine(
                    model=simple_model,
                    loss=loss_function,
                    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    ckpt_path=f.name,
                    resume_training=False  # Only load model, not optimizer/scheduler
                )
                
                # Mock the log method
                routine.log = Mock()
                
                routine.setup(stage="fit")
                
                # Verify log was called
                routine.log.assert_called_with("pretrained_model_loaded", True, prog_bar=True)
                
                # Should not have checkpoint state for optimizer/scheduler
                assert not hasattr(routine, '_checkpoint_state')
                
            finally:
                os.unlink(f.name)
    
    def test_nonexistent_checkpoint_path(self, simple_model, loss_function):
        """Test that nonexistent checkpoint path raises error."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            ckpt_path="/nonexistent/path.pt"
        )
        
        with pytest.raises(ValueError, match="Pretrained path does not exist"):
            routine.setup(stage="fit")
    
    def test_invalid_checkpoint_structure(self, simple_model, loss_function):
        """Test that invalid checkpoint structure raises error."""
        # Create checkpoint missing required keys
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {"invalid_key": "invalid_value"}
            torch.save(checkpoint, f.name)
            
            try:
                routine = SupervisedRoutine(
                    model=simple_model,
                    loss=loss_function,
                    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    ckpt_path=f.name
                )
                
                with pytest.raises(ValueError, match="Checkpoint is missing required keys"):
                    routine.setup(stage="fit")
                    
            finally:
                os.unlink(f.name)
    
    def test_load_checkpoint_for_inference(self, simple_model, sample_batch):
        """Test loading checkpoint for inference without training components."""
        # Create a checkpoint with just model state
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {
                'model_state_dict': simple_model.state_dict()
            }
            torch.save(checkpoint, f.name)
            
            try:
                # Create routine for inference only (no loss/optimizer)
                routine = SupervisedRoutine(
                    model=SimpleModel(),  # Fresh model with different weights
                    ckpt_path=f.name,
                    resume_training=False  # Not resuming training, just loading for inference
                )
                
                # Mock the log method
                routine.log = Mock()
                
                # Setup for prediction should work
                routine.setup(stage="predict")
                
                # Should be able to do inference
                inputs, _ = sample_batch
                output = routine.forward(inputs)
                
                assert output.shape == (4,)
                assert isinstance(output, torch.Tensor)
                
                # Verify model was loaded
                routine.log.assert_called_with("pretrained_model_loaded", True, prog_bar=True)
                
                # Should not have checkpoint state for optimizer/scheduler
                assert not hasattr(routine, '_checkpoint_state')
                
            finally:
                os.unlink(f.name)
    
    def test_load_checkpoint_missing_optimizer_state_with_resume(self, simple_model, loss_function):
        """Test warning when resuming training but checkpoint lacks optimizer state."""
        # Create checkpoint with model but missing optimizer state
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {
                'model_state_dict': simple_model.state_dict()
                # Missing optimizer_state_dict and lr_scheduler_state_dict
            }
            torch.save(checkpoint, f.name)
            
            try:
                routine = SupervisedRoutine(
                    model=simple_model,
                    loss=loss_function,
                    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
                    ckpt_path=f.name,
                    resume_training=True  # Want to resume but checkpoint doesn't have optimizer state
                )
                
                # Mock the log method
                routine.log = Mock()
                
                # Should issue warnings about missing optimizer/scheduler states
                with pytest.warns(UserWarning, match="checkpoint does not contain 'optimizer_state_dict'"):
                    routine.setup(stage="fit")
                
                # Model should still be loaded
                routine.log.assert_called_with("pretrained_model_loaded", True, prog_bar=True)
                
            finally:
                os.unlink(f.name)


class TestRealisticConfigurations:
    """Test realistic configuration examples that users might use."""
    
    def test_research_training_setup(self, simple_model, loss_function):
        """Test typical research training configuration."""
        lr_scheduler_config = {
            "scheduler": partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=200),
            "interval": "epoch",
            "frequency": 1
        }
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.AdamW(
                params, lr=1e-3, weight_decay=1e-2
            ),
            lr_scheduler=lr_scheduler_config,
            metrics=MetricCollection({"mae": MeanAbsoluteError(), "mse": MeanSquaredError()})
        )
        
        config = routine.configure_optimizers()
        
        assert isinstance(config["optimizer"], torch.optim.AdamW)
        assert isinstance(config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
        assert hasattr(routine, "train_metrics")
        assert isinstance(routine.train_metrics, MetricCollection)
    
    def test_fine_tuning_setup(self, simple_model, loss_function):
        """Test fine-tuning configuration."""
        lr_scheduler_config = {
            "scheduler": partial(torch.optim.lr_scheduler.ReduceLROnPlateau, factor=0.5, patience=5),
            "interval": "epoch",
            "monitor": "val_loss",
            "strict": True
        }
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=5e-5),
            lr_scheduler=lr_scheduler_config,
            metrics=MeanAbsoluteError()
        )
        
        config = routine.configure_optimizers()
        
        assert config["optimizer"].param_groups[0]['lr'] == 5e-5
        assert config["lr_scheduler"]["monitor"] == "val_loss"
        assert config["lr_scheduler"]["strict"] == True
    
    def test_inference_only_setup(self, simple_model, sample_batch):
        """Test inference-only configuration."""
        routine = SupervisedRoutine(model=simple_model)
        
        inputs, _ = sample_batch
        output = routine.forward(inputs)
        
        assert output.shape == (4,)
        assert routine.configure_optimizers() is None


class TestEdgeCasesAndHardBugs:
    """Test edge cases and potential hard-to-find bugs."""
    
    def test_forward_squeeze_edge_cases(self, simple_model):
        """Test edge cases in forward method squeezing logic."""
        
        # Test 0-D tensor (scalar) - should remain unchanged
        class ScalarModel(nn.Module):
            def forward(self, x):
                return torch.tensor(5.0)  # Scalar tensor
        
        routine = SupervisedRoutine(model=ScalarModel())
        inputs = torch.randn(1, 10)
        
        output = routine.forward(inputs)
        assert output.shape == torch.Size([])  # Scalar shape
        
        # Test 1-D tensor - should remain unchanged
        class OneDModel(nn.Module):
            def forward(self, x):
                return torch.randn(4)  # Already 1-D
        
        routine = SupervisedRoutine(model=OneDModel())
        output = routine.forward(inputs)
        assert output.shape == (4,)
        
        # Test tensor with multiple trailing dimensions of size 1
        class MultipleOnesModel(nn.Module):
            def forward(self, x):
                return torch.randn(4, 1, 1, 1)  # Shape: (4, 1, 1, 1)
        
        routine = SupervisedRoutine(model=MultipleOnesModel())
        output = routine.forward(inputs)
        # Should only squeeze the last dimension: (4, 1, 1)
        assert output.shape == (4, 1, 1)
    
    def test_metrics_state_isolation(self, simple_model, loss_function):
        """Test that metrics for different stages are properly isolated."""
        metric = MeanAbsoluteError()
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics=metric
        )
        
        # Update train metrics
        routine.train_metrics.update(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5]))
        train_result = routine.train_metrics.compute()
        
        # Val metrics should be unaffected
        val_result = routine.val_metrics.compute()
        
        # They should be different instances with different states
        assert train_result != val_result
        assert id(routine.train_metrics) != id(routine.val_metrics)
        assert id(routine.train_metrics) != id(routine.test_metrics)
        assert id(routine.val_metrics) != id(routine.test_metrics)
    
    def test_metric_prefix_collision_bug(self, simple_model, loss_function):
        """Test potential bug with metric prefix handling."""
        # Create a metric that already has a prefix
        metric = MeanAbsoluteError()
        metric.prefix = "existing_"
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics=metric
        )
        
        # Should override existing prefix
        assert routine.train_metrics.prefix == "train_"
        assert routine.val_metrics.prefix == "val_"
        assert routine.test_metrics.prefix == "test_"
    
    def test_checkpoint_resume_without_checkpoint_state(self, simple_model, loss_function):
        """Test edge case where resume_training=True but no checkpoint was provided."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
            resume_training=True  # But no checkpoint path provided
        )
        
        # Should issue a warning when no checkpoint path was provided
        with pytest.warns(UserWarning, match="resume_training=True but no checkpoint path was provided"):
            routine.setup(stage="fit")
        
        config = routine.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config
    
    def test_optimizer_parameter_mutation_bug(self, simple_model, loss_function):
        """Test potential bug where optimizer factory is called multiple times."""
        call_count = 0
        
        def counting_optimizer_factory(params):
            nonlocal call_count
            call_count += 1
            return torch.optim.Adam(params, lr=1e-3)
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=counting_optimizer_factory
        )
        
        # Should only call the factory once per configure_optimizers call
        config1 = routine.configure_optimizers()
        assert call_count == 1
        
        config2 = routine.configure_optimizers() 
        assert call_count == 2  # Called again for second configuration
    
    def test_scheduler_factory_parameter_validation(self, simple_model, loss_function):
        """Test that scheduler factory receives correct optimizer type."""
        received_optimizer = None
        
        def validating_scheduler_factory(optimizer):
            nonlocal received_optimizer
            received_optimizer = optimizer
            assert isinstance(optimizer, torch.optim.Adam)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            lr_scheduler=validating_scheduler_factory
        )
        
        config = routine.configure_optimizers()
        assert received_optimizer is not None
        assert isinstance(received_optimizer, torch.optim.Adam)
    
    def test_metrics_with_empty_dict(self, simple_model, loss_function):
        """Test edge case with empty metrics dictionary - FIXED BUG."""
        # Empty dict should behave the same as metrics=None (no metrics tracked)
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics={}  # Empty dict should be treated like None
        )
        
        # Should result in None metrics (no metrics tracking)
        assert routine.metrics is None
        
        # Should not have any metric attributes
        assert not hasattr(routine, "train_metrics")
        assert not hasattr(routine, "val_metrics")
        assert not hasattr(routine, "test_metrics")
    
    def test_checkpoint_state_dict_corruption(self, simple_model, loss_function):
        """Test handling of corrupted checkpoint state dictionaries."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Create checkpoint with wrong optimizer state dict structure
            checkpoint = {
                'model_state_dict': simple_model.state_dict(),
                'optimizer_state_dict': {"corrupted": "data"},  # Wrong structure
                'lr_scheduler_state_dict': {"also_corrupted": "data"}
            }
            torch.save(checkpoint, f.name)
            
            try:
                routine = SupervisedRoutine(
                    model=simple_model,
                    loss=loss_function,
                    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
                    ckpt_path=f.name,
                    resume_training=True
                )
                
                routine.setup(stage="fit")
                
                # The optimizer/scheduler loading might fail, but it should be handled gracefully
                # or raise a clear error, not cause silent corruption
                with pytest.raises((RuntimeError, KeyError, ValueError)):
                    routine.configure_optimizers()
                    
            finally:
                os.unlink(f.name)
    
    def test_metric_clone_deep_copy_behavior(self, simple_model, loss_function):
        """Test that metric cloning creates proper deep copies - BUG FIXED."""
        
        # Create a metric with some internal state
        original_metric = MeanAbsoluteError()
        
        # Add some data to create internal state
        original_metric.update(torch.tensor([1.0, 2.0]), torch.tensor([1.1, 2.1]))
        original_value = original_metric.compute()
        
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics=original_metric
        )
        
        # FIXED: The cloned metrics should start fresh, not inherit state
        train_value = routine.train_metrics.compute()
        
        # Cloned metrics should start with zero state (different from original)
        # Handle the case where reset metrics might return NaN
        if torch.isnan(train_value):
            # This is expected behavior for reset metrics with no data
            assert not torch.isnan(original_value)  # Original should have valid value
        else:
            assert train_value != original_value
        
        # Verify the fix: they should have different internal states
        assert not torch.equal(original_metric.sum_abs_error, routine.train_metrics.sum_abs_error)
        assert not torch.equal(original_metric.total, routine.train_metrics.total)
        
        # Cloned metrics should start at zero
        assert routine.train_metrics.sum_abs_error.item() == 0.0
        assert routine.train_metrics.total.item() == 0
        
        # Modifying original should not affect cloned metrics
        original_metric.update(torch.tensor([10.0]), torch.tensor([10.1]))
        new_original_value = original_metric.compute()
        train_value_after = routine.train_metrics.compute()
        
        # Handle NaN case in comparison
        if torch.isnan(train_value_after):
            # Reset metrics with no new data should be consistent
            assert torch.isnan(train_value) if not torch.isnan(train_value) else True
        else:
            assert train_value == train_value_after  # Cloned metric unchanged
        assert new_original_value != train_value_after
    
    def test_checkpoint_loading_with_model_architecture_changes(self, loss_function):
        """Test checkpoint loading when model architecture has changed."""
        
        # Create original model and save checkpoint
        original_model = SimpleModel(input_size=10, output_size=1)
        checkpoint = {'model_state_dict': original_model.state_dict()}
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)
            
            try:
                # Create model with different architecture
                different_model = SimpleModel(input_size=20, output_size=1)  # Different input size
                
                routine = SupervisedRoutine(
                    model=different_model,
                    loss=loss_function,
                    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    ckpt_path=f.name
                )
                
                # Should raise an error about mismatched state dict
                with pytest.raises((RuntimeError, ValueError)):
                    routine.setup(stage="fit")
                    
            finally:
                os.unlink(f.name)
   
    def test_metric_reset_between_epochs_bug(self, simple_model, loss_function):
        """Test potential bug where metrics aren't properly reset between epochs."""
        routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            metrics=MeanAbsoluteError()
        )
        
        # Simulate first epoch
        routine.train_metrics.update(torch.tensor([1.0, 2.0]), torch.tensor([1.1, 2.1]))
        first_epoch_result = routine.train_metrics.compute()
        
        # Manually reset (this should happen automatically in Lightning)
        routine.train_metrics.reset()
        
        # Simulate second epoch with different data
        routine.train_metrics.update(torch.tensor([10.0, 20.0]), torch.tensor([10.5, 20.5]))
        second_epoch_result = routine.train_metrics.compute()
        
        # Should be different if reset worked properly
        assert first_epoch_result != second_epoch_result
    
    def test_checkpoint_resume_training_state(self, simple_model, loss_function):
        """Test that training is actually resumed from the provided checkpoint."""
        # Create an optimizer and train for one step to generate state
        original_routine = SupervisedRoutine(
            model=simple_model,
            loss=loss_function,
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
            lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=2)
        )
        
        # Perform one training step to create optimizer/scheduler state
        sample_inputs = torch.randn(4, 10)
        sample_targets = torch.randn(4)
        
        original_config = original_routine.configure_optimizers()
        assert isinstance(original_config, dict)  # Should be dict when scheduler is present
        original_optimizer = original_config["optimizer"]
        original_scheduler = original_config["lr_scheduler"]["scheduler"]
        
        # Perform a dummy training step to update optimizer state
        loss = original_routine.loss(original_routine.model(sample_inputs), sample_targets)
        loss.backward()
        original_optimizer.step()
        original_scheduler.step()
        
        # Get the state after training
        original_optimizer_state = original_optimizer.state_dict()
        original_scheduler_state = original_scheduler.state_dict()
        original_model_state = original_routine.model.state_dict()
        
        # Create checkpoint with all states
        checkpoint = {
            'model_state_dict': original_model_state,
            'optimizer_state_dict': original_optimizer_state,
            'lr_scheduler_state_dict': original_scheduler_state
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)
            
            try:
                # Create new routine with fresh model and resume training
                new_model = SimpleModel()  # Fresh model with random weights
                new_routine = SupervisedRoutine(
                    model=new_model,
                    loss=loss_function,
                    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
                    lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=2),
                    ckpt_path=f.name,
                    resume_training=True
                )
                
                # Mock logging
                new_routine.log = Mock()
                
                # Setup should load the checkpoint
                new_routine.setup(stage="fit")
                
                # Configure optimizers should restore the state
                new_config = new_routine.configure_optimizers()
                assert isinstance(new_config, dict)  # Should be dict when scheduler is present
                new_optimizer = new_config["optimizer"]
                new_scheduler = new_config["lr_scheduler"]["scheduler"]
                
                # Verify that states were properly restored
                # Model state should be loaded
                for orig_param, new_param in zip(simple_model.parameters(), new_model.parameters()):
                    assert torch.allclose(orig_param.data, new_param.data), "Model weights should be restored"
                
                # Optimizer state should be loaded (check learning rate and step count)
                assert new_optimizer.state_dict()['param_groups'][0]['lr'] == original_optimizer_state['param_groups'][0]['lr']
                
                # Scheduler state should be loaded (check step count)
                assert new_scheduler.state_dict()['last_epoch'] == original_scheduler_state['last_epoch']
                
                # Verify that the routine was set up for resuming
                assert new_routine.resume_training == True
                
            finally:
                os.unlink(f.name)


# Test to verify that checkpoint loading behavior works correctly