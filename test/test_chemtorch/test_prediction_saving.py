import os
import tempfile
from pathlib import Path
from typing import List, Any
from unittest.mock import Mock, call
import pandas as pd
import pytest
import torch
from omegaconf import ListConfig

from chemtorch.utils.misc import handle_prediction_saving


class TestHandlePredictionSaving:
    """Test suite for the handle_prediction_saving function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock functions for predictions and reference data
        self.mock_get_preds_func = Mock()
        self.mock_get_reference_df_func = Mock()
        self.mock_log_func = Mock()
        self.mock_get_dataset_names_func = Mock()
        
        # Sample data
        self.sample_preds = [torch.tensor([1.0, 2.0, 3.0])]
        self.sample_df = pd.DataFrame({
            'smiles': ['CCO', 'CCC', 'CCCC'],
            'target': [0.1, 0.2, 0.3]
        })
        
        # Configure mocks to return sample data
        self.mock_get_preds_func.return_value = self.sample_preds
        self.mock_get_reference_df_func.return_value = self.sample_df
        # Default dataset names - can be overridden in individual tests
        self.mock_get_dataset_names_func.return_value = ["train", "val", "test"]

    def test_no_saving_when_no_save_paths(self):
        """Test that no saving occurs when neither save_dir nor save_path is provided."""
        handle_prediction_saving(
            get_preds_func=self.mock_get_preds_func,
            get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
            predictions_save_dir=None,
            predictions_save_path=None,
            save_predictions_for="test",
            tasks=["test"]
        )
        
        # Should not call the prediction functions
        self.mock_get_preds_func.assert_not_called()
        self.mock_get_reference_df_func.assert_not_called()

    def test_warning_when_save_predictions_for_but_no_paths(self, caplog):
        """Test warning is logged when save_predictions_for is specified but no paths given."""
        handle_prediction_saving(
            get_preds_func=self.mock_get_preds_func,
            get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
            predictions_save_dir=None,
            predictions_save_path=None,
            save_predictions_for="test",
            tasks=["test"]
        )
        
        assert "'save_predictions_for' specified but no 'predictions_save_dir' or 'predictions_save_path' given" in caplog.text

    def test_single_string_save_predictions_for(self):
        """Test handling when save_predictions_for is a single string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=None,
                predictions_save_path=str(temp_path / "test_preds.csv"),
                save_predictions_for="test",
                tasks=["test"]
            )
            
            # Should call functions for test dataset
            self.mock_get_preds_func.assert_called_once_with("test")
            self.mock_get_reference_df_func.assert_called_once_with("test")
            
            # Check file was created
            assert (temp_path / "test_preds.csv").exists()

    def test_list_save_predictions_for(self):
        """Test handling when save_predictions_for is a list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=str(temp_path),
                predictions_save_path=None,
                save_predictions_for=["test", "val"],
                tasks=["fit", "test"]
            )
            
            # Should call functions for both test and val datasets
            expected_calls = [call("test"), call("val")]
            self.mock_get_preds_func.assert_has_calls(expected_calls, any_order=True)
            self.mock_get_reference_df_func.assert_has_calls(expected_calls, any_order=True)
            
            # Check files were created
            assert (temp_path / "test_preds.csv").exists()
            assert (temp_path / "val_preds.csv").exists()

    def test_hydra_list_string_format(self):
        """Test handling when Hydra passes a list as a string (e.g., '[test,val]')."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test single item list
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=str(temp_path),
                predictions_save_path=None,
                save_predictions_for="[test]",
                tasks=["test"]
            )
            
            # Should call functions for test dataset
            self.mock_get_preds_func.assert_called_with("test")
            self.mock_get_reference_df_func.assert_called_with("test")
            
            # Check file was created
            assert (temp_path / "test_preds.csv").exists()
            
            # Reset mocks for next test
            self.mock_get_preds_func.reset_mock()
            self.mock_get_reference_df_func.reset_mock()
            
            # Test multi-item list
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=str(temp_path / "multi"),
                predictions_save_path=None,
                save_predictions_for="[test, val]",
                tasks=["fit", "test"]
            )
            
            # Should call functions for both datasets
            expected_calls = [call("test"), call("val")]
            self.mock_get_preds_func.assert_has_calls(expected_calls, any_order=True)
            self.mock_get_reference_df_func.assert_has_calls(expected_calls, any_order=True)

    def test_omegaconf_listconfig_save_predictions_for(self):
        """Test handling when save_predictions_for is an OmegaConf ListConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create ListConfig (simulating what Hydra passes)
            list_config = ListConfig(["test", "val"])
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=str(temp_path),
                predictions_save_path=None,
                save_predictions_for=list_config,
                tasks=["fit", "test"]
            )
            
            # Should call functions for both test and val datasets
            expected_calls = [call("test"), call("val")]
            self.mock_get_preds_func.assert_has_calls(expected_calls, any_order=True)
            self.mock_get_reference_df_func.assert_has_calls(expected_calls, any_order=True)
            
            # Check files were created
            assert (temp_path / "test_preds.csv").exists()
            assert (temp_path / "val_preds.csv").exists()

    def test_all_keyword_expands_to_available_partitions(self):
        """Test that 'all' keyword expands to available partitions based on tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Configure mock to return dataset names that match the expected partitions
            self.mock_get_dataset_names_func.return_value = ["train", "val", "test", "predict"]
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=str(temp_path),
                predictions_save_path=None,
                save_predictions_for="all",
                tasks=["fit", "test", "predict"]
            )
            
            # Should call functions for train, val, test, predict (all available from tasks)
            expected_calls = [call("train"), call("val"), call("test"), call("predict")]
            self.mock_get_preds_func.assert_has_calls(expected_calls, any_order=True)
            self.mock_get_reference_df_func.assert_has_calls(expected_calls, any_order=True)

    def test_validate_task_maps_to_val_dataset(self):
        """Test that 'validate' task correctly maps to 'val' dataset key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=None,
                predictions_save_path=str(temp_path / "val_preds.csv"),
                save_predictions_for=None,  # Will use task
                tasks=["validate"]
            )
            
            # Should call functions for val dataset (not validate)
            self.mock_get_preds_func.assert_called_once_with("val")
            self.mock_get_reference_df_func.assert_called_once_with("val")

    def test_single_partition_uses_predictions_save_path(self):
        """Test that single partition scenarios use predictions_save_path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            custom_path = temp_path / "custom_predictions.csv"
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=str(temp_path),
                predictions_save_path=str(custom_path),
                save_predictions_for="test",
                tasks=["test"]
            )
            
            # Should use custom path, not default naming
            assert custom_path.exists()
            assert not (temp_path / "test_preds.csv").exists()

    def test_multi_partition_uses_predictions_save_dir(self):
        """Test that multi-partition scenarios use predictions_save_dir with default naming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=str(temp_path),
                predictions_save_path=str(temp_path / "custom.csv"),
                save_predictions_for=["test", "val"],
                tasks=["fit"]
            )
            
            # Should use default naming pattern, not custom path
            assert (temp_path / "test_preds.csv").exists()
            assert (temp_path / "val_preds.csv").exists()
            assert not (temp_path / "custom.csv").exists()

    def test_invalid_save_predictions_for_value_raises_error(self):
        """Test that invalid values in save_predictions_for raise ValueError."""
        with pytest.raises(ValueError, match="Invalid entries in 'save_predictions_for'"):
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir="/tmp",
                predictions_save_path=None,
                save_predictions_for=["invalid_key"],
                tasks=["test"]
            )

    def test_invalid_save_predictions_for_type_raises_error(self):
        """Test that invalid type for save_predictions_for raises ValueError."""
        with pytest.raises(ValueError, match="'save_predictions_for' must be a string or list"):
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir="/tmp",
                predictions_save_path=None,
                save_predictions_for=123,  # Invalid type
                tasks=["test"]
            )

    def test_multi_partition_without_save_dir_raises_error(self, caplog):
        """Test that multi-partition scenarios without save_dir log warning and return early."""
        handle_prediction_saving(
            get_preds_func=self.mock_get_preds_func,
            get_reference_df_func=self.mock_get_reference_df_func,
            get_dataset_names_func=self.mock_get_dataset_names_func,
            predictions_save_dir=None,
            predictions_save_path="/tmp/test.csv",
            save_predictions_for=["test", "val"],
            tasks=["fit"]
        )
        
        # Should log warning and not call prediction functions
        assert "'save_predictions_for' has to be set to save predictions" in caplog.text
        self.mock_get_preds_func.assert_not_called()
        self.mock_get_reference_df_func.assert_not_called()

    def test_logging_function_called_when_provided(self):
        """Test that log_func is called when predictions are saved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=None,
                predictions_save_path=str(temp_path / "test_preds.csv"),
                save_predictions_for="test",
                tasks=["test"],
                log_func=self.mock_log_func
            )
            
            # Should call log function
            self.mock_log_func.assert_called_once()
            call_args = self.mock_log_func.call_args[0][0]
            assert "predictions_file" in call_args
            assert "num_predictions" in call_args

    def test_no_predictions_skips_saving(self):
        """Test that when no predictions are returned, saving is skipped."""
        # Configure mock to return empty predictions
        self.mock_get_preds_func.return_value = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            save_path = temp_path / "test_preds.csv"
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=None,
                predictions_save_path=str(save_path),
                save_predictions_for="test",
                tasks=["test"]
            )
            
            # File should not be created
            assert not save_path.exists()

    def test_root_dir_path_resolution(self):
        """Test that root_dir is properly used for path resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            relative_path = "predictions/test_preds.csv"
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=None,
                predictions_save_path=relative_path,
                save_predictions_for="test",
                tasks=["test"],
                root_dir=root_path
            )
            
            # Should create file at root_dir/relative_path
            expected_path = root_path / relative_path
            assert expected_path.exists()

    def test_predictions_saved_correctly_with_proper_content(self):
        """Test that predictions are saved with correct content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            save_path = temp_path / "test_preds.csv"
            
            handle_prediction_saving(
                get_preds_func=self.mock_get_preds_func,
                get_reference_df_func=self.mock_get_reference_df_func,
                get_dataset_names_func=self.mock_get_dataset_names_func,
                predictions_save_dir=None,
                predictions_save_path=str(save_path),
                save_predictions_for="test",
                tasks=["test"]
            )
            
            # Read saved file and verify content
            saved_df = pd.read_csv(save_path)
            assert len(saved_df) == 3
            assert 'prediction' in saved_df.columns
            assert 'smiles' in saved_df.columns
            assert 'target' in saved_df.columns
            assert saved_df['prediction'].tolist() == [1.0, 2.0, 3.0]