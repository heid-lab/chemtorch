"""
Integration tests for saved config integrity.

This test suite ensures that saved optimal model configs can be loaded and executed
correctly, with validation against reference predictions and expected validation losses.
"""

import os
import sys
from pathlib import Path
from typing import List

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from test.test_integration.helpers import (
    BASE_OVERRIDES,
    CONFIG_ROOT,
    ConfigTester,
    PROJECT_ROOT,
    TEST_DIR,
    compare_predictions,
    extract_val_loss,
    get_baselines,
    load_baselines,
    parametrize_config_tests,
)

# Config-specific paths
CONFIG_DIR_FROM_CONFIG_ROOT = Path("saved_configs/chemtorch_benchmark/optimal_model_configs")
TEST_SEARCH_DIR = CONFIG_ROOT / CONFIG_DIR_FROM_CONFIG_ROOT

# Test fixtures paths
PREDS_FIXTURES_DIR = TEST_DIR / "fixtures" / "ref_preds"
PREDS_DEBUG_DIR = TEST_DIR / "debug_preds"
BASELINES_PATH = TEST_DIR / "fixtures" / "baselines.yaml"

# Test configuration
ENABLE_EXTENDED_TESTS = False  # Set to True to enable 3-epoch tests by default
RUN_EXTENDED_TESTS = os.getenv("RUN_EXTENDED_TESTS", str(ENABLE_EXTENDED_TESTS)).lower() == "true"
TEST_PREDS_SAVE_PATH = Path("/tmp/chemtorch_test_preds.csv")

# Prediction-specific overrides (extend BASE_OVERRIDES)
PREDICTION_OVERRIDES = BASE_OVERRIDES + [
    f"++save_predictions_for=test",
    f"++predictions_save_path={TEST_PREDS_SAVE_PATH}",
]

class SavedConfigTester(ConfigTester):
    """Tester for saved config files."""
    
    def _init_config(self, rel_config_path: Path, config_name: str) -> DictConfig:
        abs_config_path = str(self.search_dir_path / rel_config_path)
        with initialize_config_dir(config_dir=abs_config_path, version_base=None):
            cfg: DictConfig = compose(config_name=config_name)
        return cfg
    
    def _build_base_cmd(self, rel_config_path: Path, config_name: str) -> List[str]:
        abs_config_path = str(self.search_dir_path / rel_config_path)
        cmd = [
            sys.executable,
            "chemtorch_cli.py",
            "--config-path",
            abs_config_path,
            "--config-name",
            config_name,
        ]
        return cmd 


@pytest.fixture(scope="session")
def saved_config_tester():
    """Create a SavedConfigTester instance."""
    return SavedConfigTester(CONFIG_DIR_FROM_CONFIG_ROOT, CONFIG_ROOT, PROJECT_ROOT)


@pytest.fixture(scope="session", autouse=True)
def baselines():
    """Load baseline configurations."""
    return load_baselines(BASELINES_PATH)


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for all discovered configs."""
    parametrize_config_tests(metafunc, TEST_SEARCH_DIR)


def _run_epoch_test(
    config_info,
    saved_config_tester: SavedConfigTester,
    num_epochs: int,
):
    """Helper function to run config tests for a specific number of epochs."""
    rel_config_path, config_name = config_info
    baselines_config = get_baselines()
    assert baselines_config is not None, "Baselines not loaded"
    
    # Skip if no baseline defined for this config
    if not baselines_config.has_baseline(config_name):
        pytest.skip(f"No baseline defined for config '{config_name}'")
    
    model = baselines_config.get_model(config_name)
    assert model is not None, f"Model baseline not found for '{config_name}'"
    
    # Calculate timeout based on number of epochs
    timeout = model.calculate_timeout(num_epochs)
    
    # Prepare overrides
    extra_overrides = PREDICTION_OVERRIDES.copy()
    extra_overrides.append(f"trainer.max_epochs={num_epochs}")
    
    # Clean up previous predictions
    if TEST_PREDS_SAVE_PATH.exists():
        TEST_PREDS_SAVE_PATH.unlink()
    
    # Run the config
    std_out, execution_time = saved_config_tester.test_config(
        rel_config_path=rel_config_path, 
        config_name=config_name,
        timeout=timeout,
        remove_keys=["predictions_save_dir"],
        extra_overrides=extra_overrides,
        common_overrides=[],  # Already included in extra_overrides
    )
    
    # Validate validation loss
    val_loss = extract_val_loss(std_out)
    assert val_loss is not None, "Failed to extract val_loss from output"
    expected_val_loss = model.get_val_loss(num_epochs)
    assert val_loss == expected_val_loss, (
        f"Expected val_loss to be {expected_val_loss}, but got {val_loss}"
    )
    
    # Compare predictions if reference exists
    ref_filename = model.get_reference_prediction_file(num_epochs)
    if ref_filename:
        reference_preds_path = PREDS_FIXTURES_DIR / ref_filename
        if reference_preds_path.exists():
            compare_predictions(
                TEST_PREDS_SAVE_PATH,
                reference_preds_path,
                tolerance=baselines_config.tolerance,
                debug_dir=PREDS_DEBUG_DIR,
            )


@pytest.mark.dependency(name="init")
def test_config_init(config_info, saved_config_tester: SavedConfigTester):
    """Ensure saved configs can be initialised before running execution checks."""
    rel_config_path, config_name = config_info
    saved_config_tester.init_config(rel_config_path, config_name)


@pytest.mark.integration
@pytest.mark.dependency(depends=["init"])
def test_saved_config_1_epoch(config_info, saved_config_tester: SavedConfigTester):
    """Test saved config execution for 1 epoch."""
    _run_epoch_test(config_info, saved_config_tester, num_epochs=1)


@pytest.mark.integration
@pytest.mark.extended
@pytest.mark.dependency(depends=["init"])
def test_saved_config_3_epoch(config_info, saved_config_tester: SavedConfigTester):
    """Test saved config execution for 3 epochs (extended test)."""
    if not RUN_EXTENDED_TESTS:
        pytest.skip("Extended tests disabled. Set RUN_EXTENDED_TESTS=true or ENABLE_EXTENDED_TESTS=True to run.")
    _run_epoch_test(config_info, saved_config_tester, num_epochs=3)

