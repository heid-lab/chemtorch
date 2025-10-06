"""
Integration tests for config integrity and API compatibility.

This test suite ensures that all experiment configs can run without errors
and serves as an early warning system for breaking changes.
"""

from glob import glob
import pytest
import subprocess
import sys
import tempfile
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from time import time
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Test configuration flags
ENABLE_EXTENDED_TESTS = False  # Set to True to enable 3-epoch tests by default
RUN_EXTENDED_TESTS = os.getenv("RUN_EXTENDED_TESTS", str(ENABLE_EXTENDED_TESTS)).lower() == "true"
DEBUG_SAVE_PREDICTIONS = False  # Set to True to save test predictions to preds/ folder for debugging

# Test configuration
TEST_CONFIG = {
    "common_overrides": [
        "log=false",
        "save_predictions_for=test",  # Only save test predictions for comparison
        "+predictions_save_path=/tmp/chemtorch_test_preds.csv",
        "~predictions_save_dir",  # Disable predictions_save_dir (used in reference config)
        "+data_module.subsample=0.01",
        "trainer.enable_checkpointing=false",
    ],
    "1_epoch_overrides": [
        "trainer.max_epochs=1",
    ],
    "3_epoch_overrides": [
        "trainer.max_epochs=3",
    ],
    "precompute_timeout": 10,
    "prediction_tolerance": 1e-4,  # Tolerance for prediction comparison (allows small numerical differences)
}

# Expected baseline metrics for regression testing (subsample=0.01, max_epochs=1 or max_epochs=3)
BASELINES = {
    "atom_han": {
        "val_loss_epoch_1": 1.34,
        "val_loss_epoch_3": 1.39,
        "precompute_timeout": 10,
        "timeout_per_epoch": 60
    },
    "cgr_dmpnn": {
        "val_loss_epoch_1": 1.36,
        "val_loss_epoch_3": 1.49,
        "precompute_timeout": 10,
        "timeout_per_epoch": 60
    },
    "drfp_mlp": {
        "val_loss_epoch_1": 1.30,
        "val_loss_epoch_3": 1.34,
        "precompute_timeout": 30,
        "timeout_per_epoch": 30
    },
    "dimereaction": {
        "val_loss_epoch_1": 1.37,
        "val_loss_epoch_3": 1.16,
        "precompute_timeout": 20,
        "timeout_per_epoch": 80
    }
}


def compare_predictions(test_preds_path: Path, reference_preds_path: Path, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare test predictions with reference predictions."""
    try:
        # Load predictions
        test_df = pd.read_csv(test_preds_path)
        ref_df = pd.read_csv(reference_preds_path)
        
        # Basic validation
        if len(test_df) != len(ref_df):
            return {
                "match": False,
                "error": f"Different number of predictions: {len(test_df)} vs {len(ref_df)}"
            }
        
        # Check if prediction columns exist
        pred_cols = [col for col in test_df.columns if 'pred' in col.lower()]
        if not pred_cols:
            return {
                "match": False,
                "error": "No prediction columns found"
            }
        
        # Compare predictions
        results = {"match": True, "column_comparisons": {}}
        
        for col in pred_cols:
            if col not in ref_df.columns:
                results["match"] = False
                results["column_comparisons"][col] = {
                    "error": f"Column {col} not found in reference"
                }
                continue
                
            test_values = np.array(test_df[col].values)
            ref_values = np.array(ref_df[col].values)
            
            # Handle NaN values
            valid_mask = ~(np.isnan(test_values) | np.isnan(ref_values))
            if not np.any(valid_mask):
                results["column_comparisons"][col] = {
                    "error": "All values are NaN"
                }
                continue
                
            test_valid = test_values[valid_mask]
            ref_valid = ref_values[valid_mask]
            
            # Calculate metrics
            max_diff = np.max(np.abs(test_valid - ref_valid))
            mean_diff = np.mean(np.abs(test_valid - ref_valid))
            rmse = np.sqrt(np.mean((test_valid - ref_valid) ** 2))
            
            column_match = max_diff <= tolerance
            if not column_match:
                results["match"] = False
                
            results["column_comparisons"][col] = {
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "rmse": float(rmse),
                "tolerance": tolerance,
                "match": column_match,
                "num_valid": int(np.sum(valid_mask))
            }
        
        return results
        
    except Exception as e:
        return {
            "match": False,
            "error": f"Error comparing predictions: {str(e)}"
        }


def find_reference_predictions(config_name: str, epochs: int, test_dir: Path) -> Optional[Path]:
    """Find reference test predictions for a given config and epoch count."""
    if config_name not in BASELINES:
        return None
        
    # Look for reference file in fixtures directory
    fixtures_dir = test_dir / "fixtures"
    reference_file = fixtures_dir / f"rdb7_subsample_0.01_{config_name}_seed_0_epoch_{epochs}.csv"
    
    if reference_file.exists():
        return reference_file
    
    return None


class ConfigIntegrityTester:
    """Test runner for config integrity checks."""
    
    def __init__(self, root_dir: Path, configs_to_test: List[str]):
        self.root_dir = root_dir
        self.conf_dir = root_dir / "conf"
        self.configs_to_test = configs_to_test
        self.results = {}
        
    def discover_configs(self) -> List[str]:
        """Discover all available configs for testing."""
        
        # Verify that the configs exist
        optimal_configs_dir = self.conf_dir / "saved_configs" / "chemtorch_benchmark" / "optimal_model_configs"
        available_configs = []
        
        if optimal_configs_dir.exists():
            for config_name in self.configs_to_test:
                config_file = optimal_configs_dir / f"{config_name}.yaml"
                if config_file.exists():
                    available_configs.append(config_name)
                else:
                    print(f"Warning: Config {config_name} not found at {config_file}")
        
        return available_configs
    
    def validate_config_syntax(self, config_name: str) -> bool:
        """Validate that a config can be loaded without syntax errors."""
        try:
            GlobalHydra.instance().clear()
            with initialize(config_path="../../conf/saved_configs/chemtorch_benchmark/optimal_model_configs", version_base=None):
                cfg = compose(config_name=config_name)
                return True
        except Exception as e:
            print(f"Config syntax error in {config_name}: {e}")
            return False
        finally:
            GlobalHydra.instance().clear()
    
    def run_config_test(self, config_name: str, extended_test: bool = False) -> Dict[str, Any]:
        """Run a single config test."""
        result = {
            "config_name": config_name,
            "extended_test": extended_test,
            "syntax_valid": False,
            "execution_success": False,
            "execution_time": None,
            "error_message": None,
            "full_stdout": None,
            "full_stderr": None,
            "metrics": None
        }
        
        # First check syntax
        result["syntax_valid"] = self.validate_config_syntax(config_name)
        if not result["syntax_valid"]:
            return result
        
        # Run execution test
        try:
            cmd = [
                sys.executable, "chemtorch_cli.py",
                "--config-path", "conf/saved_configs/chemtorch_benchmark/optimal_model_configs",
                "--config-name", config_name
            ]
            
            # Add common overrides first
            cmd.extend(TEST_CONFIG["common_overrides"])
            
            # Add epoch-specific overrides
            if extended_test:
                cmd.extend(TEST_CONFIG["3_epoch_overrides"])
                training_epochs = 3
            else:
                cmd.extend(TEST_CONFIG["1_epoch_overrides"])
                training_epochs = 1
            
            # Calculate timeout - use per-model settings if available, otherwise skip config
            if config_name in BASELINES:
                precompute_timeout = BASELINES[config_name].get("precompute_timeout", TEST_CONFIG["precompute_timeout"])
                timeout_per_epoch = BASELINES[config_name]["timeout_per_epoch"]
            else:
                result["error_message"] = f"Config {config_name} not in BASELINES - skipping test"
                return result
            
            timeout = precompute_timeout + timeout_per_epoch * training_epochs

            # Run with timeout
            start_time = time()
            process = subprocess.run(
                cmd,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time()
            
            result["execution_success"] = (process.returncode == 0)
            result["execution_time"] = end_time - start_time
            result["full_stdout"] = process.stdout
            result["full_stderr"] = process.stderr
            
            # Print output for debugging (since we captured it)
            if process.stdout:
                print("STDOUT:")
                print(process.stdout)
            if process.stderr:
                print("STDERR:")
                print(process.stderr)
            
            if not result["execution_success"]:
                result["error_message"] = f"Process failed with return code {process.returncode}"
            else:
                # Check if test predictions were saved to the specified path
                test_preds_file = Path("/tmp/chemtorch_test_preds.csv")
                if test_preds_file.exists():
                    result["test_preds_path"] = str(test_preds_file)
                
                # Extract metrics from stdout
                if process.stdout:
                    result["metrics"] = self._extract_metrics_from_output(process.stdout)
                else:
                    result["metrics"] = None
                
        except subprocess.TimeoutExpired as e:
            result["error_message"] = f"Test timed out after {timeout} seconds"
            if hasattr(e, 'stdout') and e.stdout:
                result["full_stdout"] = e.stdout
            if hasattr(e, 'stderr') and e.stderr:
                result["full_stderr"] = e.stderr
        except Exception as e:
            result["error_message"] = str(e)
        
        return result
    
    def _extract_metrics_from_output(self, output: str) -> Optional[Dict[str, float]]:
        """Extract validation and test metrics from CLI output."""
        metrics = {}
        lines = output.split('\n')
        
        # Look for Lightning progress bar output with val_loss_epoch
        for line in lines:
            # Look for validation metrics in progress bar: "val_loss_epoch=1.280"
            if "val_loss_epoch=" in line:
                try:
                    # Extract val_loss_epoch value from progress bar
                    parts = line.split("val_loss_epoch=")
                    if len(parts) > 1:
                        val_part = parts[1].split(",")[0].split("]")[0].strip()
                        metrics["val_loss_epoch"] = float(val_part)
                except (ValueError, IndexError):
                    continue
            
        return metrics if metrics else None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all config integrity tests."""
        # Clean up prediction file if it exists
        test_preds_file = Path("/tmp/chemtorch_test_preds.csv")
        if test_preds_file.exists():
            test_preds_file.unlink()
        
        config_list = self.discover_configs()
        results = {
            "summary": {
                "total_configs": len(config_list),
                "passed_basic": 0,
                "passed_extended": 0,
                "failed": 0,
                "syntax_errors": 0,
                "execution_errors": 0
            },
            "detailed_results": {}
        }
        
        for config_name in config_list:
            # Clean up any existing prediction file before each test
            test_preds_file = Path("/tmp/chemtorch_test_preds.csv")
            if test_preds_file.exists():
                test_preds_file.unlink()
                
            print(f"Testing saved_config/{config_name} (1 epoch)...")
            result_basic = self.run_config_test(config_name, extended_test=False)
            # Use a simple tuple key instead of a complex string
            results["detailed_results"][(config_name, 1)] = result_basic
            
            # Update summary for basic test
            if not result_basic["syntax_valid"]:
                results["summary"]["syntax_errors"] += 1
            elif not result_basic["execution_success"]:
                results["summary"]["execution_errors"] += 1
            else:
                results["summary"]["passed_basic"] += 1
                
                # Only run extended test if enabled
                if RUN_EXTENDED_TESTS:
                    # Clean up prediction file before extended test
                    test_preds_file = Path("/tmp/chemtorch_test_preds.csv")
                    if test_preds_file.exists():
                        test_preds_file.unlink()
                        
                    print(f"Testing saved_config/{config_name} (3 epochs)...")
                    result_extended = self.run_config_test(config_name, extended_test=True)
                    results["detailed_results"][(config_name, 3)] = result_extended
                    
                    if result_extended["execution_success"]:
                        results["summary"]["passed_extended"] += 1
                    else:
                        results["summary"]["execution_errors"] += 1
                else:
                    print(f"Skipping extended test for {config_name} (extended tests disabled)")
                
        results["summary"]["failed"] = (
            results["summary"]["syntax_errors"] + 
            results["summary"]["execution_errors"]
        )
        
        return results


# Pytest test functions
@pytest.fixture(scope="session")
def tester():
    """Create a ConfigIntegrityTester instance."""
    root_dir = Path(__file__).parent.parent.parent
    return ConfigIntegrityTester(root_dir, configs_to_test=list(BASELINES.keys()))


@pytest.fixture(scope="session")
def test_results(tester):
    """Run all tests and return results."""
    return tester.run_all_tests()


def test_config_discovery(tester):
    """Test that we can discover saved configs."""
    found_configs = tester.discover_configs()
    assert len(found_configs) > 0, "No saved configs found"
    
    # Check that cgr_dmpnn config exists (the one we're currently testing)
    
    for expected in BASELINES.keys():
        assert expected in found_configs, f"Expected config {expected} not found"


@pytest.mark.integration
@pytest.mark.order("last")
class TestConfigExecution:
    """Tests that require running actual configs (slow tests)."""
    
    def test_all_configs_syntax_valid(self, test_results):
        """Test that all configs have valid syntax."""
        syntax_errors = []
        for (config_name, epochs), result in test_results["detailed_results"].items():
            if not result["syntax_valid"]:
                syntax_errors.append(f"{config_name}_{epochs}epoch")
        
        assert len(syntax_errors) == 0, f"Configs with syntax errors: {syntax_errors}"

    def test_all_configs_execute_successfully(self, test_results):
        """Test that all configs execute without errors."""
        execution_errors = []
        for (config_name, epochs), result in test_results["detailed_results"].items():
            if result["syntax_valid"] and not result["execution_success"]:
                execution_errors.append({
                    "config": f"{config_name}_{epochs}epoch",
                    "error": result["error_message"],
                    "execution_time": result.get("execution_time"),
                })
        
        if execution_errors:
            # Print detailed error information for debugging
            error_details = []
            for error_info in execution_errors:
                detail = f"\n{'='*80}\nConfig: {error_info['config']}\n{'='*80}\n"
                detail += f"Execution time: {error_info['execution_time']}\n"
                detail += f"Error:\n{error_info['error']}\n"
                error_details.append(detail)
            
            full_error_msg = "Configs with execution errors:" + "".join(error_details)
            assert False, full_error_msg

    def test_prediction_consistency(self, test_results, tester):
        """Test that predictions match reference predictions."""
        prediction_errors = []
        successful_configs = []
        configs_with_baselines = []
        
        test_dir = Path(__file__).parent
        
        # Create preds directory for debugging only if debug mode is enabled
        preds_dir = None
        if DEBUG_SAVE_PREDICTIONS:
            preds_dir = test_dir / "preds"
            preds_dir.mkdir(exist_ok=True)
        
        for (config_name, epochs), result in test_results["detailed_results"].items():
            if not result["execution_success"]:
                continue
                
            successful_configs.append((config_name, epochs))
            
            if config_name not in BASELINES:
                continue
                
            configs_with_baselines.append((config_name, epochs))
            
            # Check if we have test predictions
            test_preds_path = result.get("test_preds_path")
            if not test_preds_path or not Path(test_preds_path).exists():
                prediction_errors.append({
                    "config": f"{config_name}_{epochs}epoch",
                    "error": "No test predictions found"
                })
                continue
            
            # Save predictions to preds directory for debugging (only if debug mode is enabled)
            if DEBUG_SAVE_PREDICTIONS and preds_dir is not None:
                debug_preds_path = preds_dir / f"rdb7_subsample_0.01_{config_name}_seed_0_epoch_{epochs}.csv"
                try:
                    import shutil
                    shutil.copy2(test_preds_path, debug_preds_path)
                    print(f"Saved predictions for debugging: {debug_preds_path}")
                except Exception as e:
                    print(f"Warning: Could not save predictions for debugging: {e}")
                
            # Find reference predictions
            reference_path = find_reference_predictions(config_name, epochs, test_dir)
            if not reference_path or not reference_path.exists():
                prediction_errors.append({
                    "config": f"{config_name}_{epochs}epoch",
                    "error": f"No reference predictions found for {config_name} with {epochs} epochs"
                })
                continue
                
            # Compare predictions
            comparison = compare_predictions(
                Path(test_preds_path), 
                reference_path,
                tolerance=TEST_CONFIG["prediction_tolerance"]
            )
            
            if not comparison["match"]:
                prediction_errors.append({
                    "config": f"{config_name}_{epochs}epoch",
                    "error": f"Predictions don't match reference",
                    "details": comparison
                })
        
        # Check if we have any successful configs to validate
        if not successful_configs:
            pytest.skip("No configs executed successfully, cannot check prediction consistency")
        
        # Check if any successful configs have baselines to compare against
        if not configs_with_baselines:
            successful_config_names = [name for name, epochs in successful_configs]
            pytest.skip(f"No successful configs have reference baselines to compare against. "
                       f"Successful configs: {successful_config_names}, "
                       f"Configs with baselines: {list(BASELINES.keys())}")
        
        if prediction_errors:
            # Print detailed error information
            error_details = []
            for error_info in prediction_errors:
                detail = f"\n{'='*80}\nConfig: {error_info['config']}\n{'='*80}\n"
                detail += f"Error: {error_info['error']}\n"
                if "details" in error_info:
                    detail += f"Details: {error_info['details']}\n"
                error_details.append(detail)
            
            full_error_msg = "Configs with prediction mismatches:" + "".join(error_details)
            assert False, full_error_msg

    @pytest.mark.parametrize("config_name", BASELINES.keys())
    def test_config_metrics_1_epoch(self, test_results, config_name):
        """Test that 1-epoch metrics are within expected ranges."""
        _test_config_metrics_for_epoch(test_results, config_name, 1)

    @pytest.mark.parametrize("config_name", BASELINES.keys())
    @pytest.mark.extended  # Mark as extended test
    def test_config_metrics_3_epochs(self, test_results, config_name):
        """Test that 3-epoch metrics are within expected ranges."""
        if not RUN_EXTENDED_TESTS:
            pytest.skip("Extended tests disabled. Set RUN_EXTENDED_TESTS=true or ENABLE_EXTENDED_TESTS=True to run.")
        _test_config_metrics_for_epoch(test_results, config_name, 3)


def _test_config_metrics_for_epoch(test_results, config_name, epoch):
    """Test that epoch metrics are within expected ranges."""
    if (config_name, epoch) not in test_results["detailed_results"]:
        pytest.skip(f"Test for config {config_name} with {epoch} epoch not found or failed to run")
    
    result = test_results["detailed_results"][(config_name, epoch)]
    
    if not result["execution_success"] or not result["metrics"]:
        pytest.skip(f"Test for config {config_name} did not produce metrics")
    
    # Check metrics against baselines
    if config_name in BASELINES:
        actual_metrics = result["metrics"]
        assert "val_loss_epoch" in actual_metrics
        value = actual_metrics["val_loss_epoch"]
        expected_val = BASELINES[config_name][f"val_loss_epoch_{epoch}"]
        
        # Compare rounded values (since progress bar rounds to 2 decimal places)
        assert round(value, 2) == expected_val, (
            f"val_loss_epoch = {round(value, 2)} differs from expected {expected_val} "
            f"for epoch {epoch} of config {config_name} (actual unrounded: {value})"
        )
