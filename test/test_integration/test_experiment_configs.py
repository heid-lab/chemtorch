"""
Integration tests for experiment config integrity.

This test suite ensures that all experiment configs can be loaded and executed
without syntax or runtime errors. Tests all configs in conf/experiment/ including
subdirectories.
"""

import sys
from pathlib import Path
from typing import List

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from test.test_integration.helpers import (
    CONFIG_ROOT,
    ConfigTester,
    PROJECT_ROOT,
    TEST_DIR,
    parametrize_config_tests,
)

# Config-specific paths
CONFIG_DIR_FROM_CONFIG_ROOT = Path("experiment")
TEST_SEARCH_DIR = CONFIG_ROOT / CONFIG_DIR_FROM_CONFIG_ROOT

# Test configuration
BASE_CONFIG_NAME = "base"
KEYS_TO_REMOVE = [
    "predictions_save_path",
    "predictions_save_dir",
    "save_predictions_for",
]

# Configs to skip in CI/CD (temporarily exclude broken configs)
SKIP_CONFIGS = [
    "opi_tutorial/training",  # TODO: Fix and re-enable
]

class ExperimentConfigTester(ConfigTester):
    """Tester for experiment config files."""
    
    def _init_config(self, rel_config_path: Path, config_name: str) -> DictConfig:
        experiment_name = str(rel_config_path / config_name)
        with initialize_config_dir(config_dir=str(self.config_root), version_base=None):
            cfg: DictConfig = compose(
                config_name=BASE_CONFIG_NAME,
                overrides=[f"+experiment={experiment_name}"],
            )
        return cfg

    def _build_base_cmd(self, rel_config_path: Path, config_name: str) -> List[str]:
        experiment_name = str(rel_config_path / config_name)
        cmd = [
            sys.executable,
            "chemtorch_cli.py",
            f"+experiment={experiment_name}",
        ]
        return cmd


@pytest.fixture(scope="session")
def experiment_tester():
    """Create an ExperimentConfigTester instance."""
    return ExperimentConfigTester(CONFIG_DIR_FROM_CONFIG_ROOT, CONFIG_ROOT, PROJECT_ROOT)


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for all discovered configs."""
    parametrize_config_tests(metafunc, TEST_SEARCH_DIR)


@pytest.mark.dependency(name="init")
def test_config_init(config_info, experiment_tester: ExperimentConfigTester):
    """Ensure experiment configs can be initialized before running execution checks."""
    rel_config_path, config_name = config_info
    config_path_str = str(rel_config_path / config_name)
    
    # Skip configs that are temporarily broken
    if config_path_str in SKIP_CONFIGS:
        pytest.skip(f"Config '{config_path_str}' temporarily excluded from CI/CD")
    
    experiment_tester.init_config(rel_config_path, config_name)


@pytest.mark.integration
@pytest.mark.dependency(depends=["init"])
def test_experiment_config_exec(config_info, experiment_tester: ExperimentConfigTester):
    """Smoke test: Run experiment config for 1 epoch to check for runtime errors."""
    rel_config_path, config_name = config_info
    config_path_str = str(rel_config_path / config_name)
    
    # Skip configs that are temporarily broken
    if config_path_str in SKIP_CONFIGS:
        pytest.skip(f"Config '{config_path_str}' temporarily excluded from CI/CD")
    
    experiment_tester.test_config(
        rel_config_path=rel_config_path,
        config_name=config_name,
        remove_keys=KEYS_TO_REMOVE,
    )
