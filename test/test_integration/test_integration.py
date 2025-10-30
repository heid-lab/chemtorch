"""
Unified integration tests for all config types.

This test suite ensures that all registered configurations (experiments, saved models, etc.)
can be loaded and executed correctly. It supports two types of tests:
- 'smoke': Checks if a config can be initialized and run for one epoch without errors.
- 'baseline': Performs a smoke test and additionally validates the final validation
  loss and prediction outputs against predefined baselines.
"""

import os
import sys
from pathlib import Path
from typing import List

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

import yaml
from test.test_integration.helpers import (
    BASE_OVERRIDES,
    CONFIG_ROOT,
    PROJECT_ROOT,
    TEST_DIR,
    ConfigTester,
    compare_predictions,
    extract_val_loss,
    load_baselines,
)

# Path to the test registry
TEST_REGISTRY_PATH = TEST_DIR / "test_registry.yaml"

# Test fixtures paths
PREDS_DEBUG_DIR = TEST_DIR / "debug" / "preds"

# Test configuration
ENABLE_EXTENDED_TESTS = False
RUN_EXTENDED_TESTS = os.getenv("RUN_EXTENDED_TESTS", str(ENABLE_EXTENDED_TESTS)).lower() == "true"
TEST_PREDS_SAVE_PATH = Path("/tmp/chemtorch_test_preds.csv")

# Common keys to remove for smoke tests
SMOKE_KEYS_TO_REMOVE = [
    "predictions_save_path",
    "predictions_save_dir",
    "save_predictions_for",
]

# Overrides for baseline tests that generate predictions
PREDICTION_OVERRIDES = BASE_OVERRIDES + [
    f"++save_predictions_for=test",
    f"++predictions_save_path={TEST_PREDS_SAVE_PATH}",
]


class UnifiedConfigTester(ConfigTester):
    """A unified tester for different config invocation modes."""

    def __init__(self, invocation_mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invocation_mode = invocation_mode

    def _init_config(self, rel_config_path: Path, config_name: str) -> DictConfig:
        if self.invocation_mode == "experiment":
            # For experiments, rel_config_path is relative to the 'experiment' dir
            experiment_name = str(rel_config_path / config_name)
            with initialize_config_dir(config_dir=str(self.config_root), version_base=None):
                cfg: DictConfig = compose(
                    config_name="base",
                    overrides=[f"+experiment={experiment_name}"],
                )
            return cfg
        else:  # config_name mode
            # For saved_configs, rel_config_path is relative to the project root
            abs_config_path = str(self.project_root / rel_config_path)
            with initialize_config_dir(config_dir=abs_config_path, version_base=None):
                cfg: DictConfig = compose(config_name=config_name)
            return cfg

    def _build_base_cmd(self, rel_config_path: Path, config_name: str) -> List[str]:
        if self.invocation_mode == "experiment":
            experiment_name = str(rel_config_path / config_name)
            return ["chemtorch", f"+experiment={experiment_name}"]
        else:  # config_name mode
            abs_config_path = str(self.project_root / rel_config_path)
            return [
                "chemtorch",
                "--config-path",
                abs_config_path,
                "--config-name",
                config_name,
            ]


@pytest.fixture(scope="session")
def config_testers():
    """Create a dictionary of testers for each invocation mode."""
    return {
        "experiment": UnifiedConfigTester("experiment", Path("."), CONFIG_ROOT, PROJECT_ROOT),
        "config_name": UnifiedConfigTester("config_name", Path("."), CONFIG_ROOT, PROJECT_ROOT),
    }


def _load_test_sets():
    """Load all enabled test sets from the registry."""
    if not TEST_REGISTRY_PATH.exists():
        return []
    with open(TEST_REGISTRY_PATH, "r") as f:
        registry = yaml.safe_load(f)
    return [
        (name, data)
        for name, data in registry.get("test_sets", {}).items()
        if data.get("enabled", False)
    ]


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for all discovered configs from the registry."""
    if "config_info" not in metafunc.fixturenames:
        return
    
    # Determine which test function is being called
    test_function = metafunc.function.__name__
    
    # Map test function names to their corresponding test_case names
    test_case_map = {
        "test_config_init": "init",
        "test_smoke": "smoke",
        "test_baseline": "baseline",
        "test_extended": "extended",
    }
    
    required_test_case = test_case_map.get(test_function)
    if not required_test_case:
        return
    
    test_sets = _load_test_sets()
    all_configs = []
    
    for test_set_name, test_set_data in test_sets:
        # Skip this test set if it doesn't have the required test case enabled
        if required_test_case not in test_set_data.get("test_cases", []):
            continue
        
        search_path = PROJECT_ROOT / test_set_data["path"]
        base_path = test_set_data["path"]
        
        for config_file in search_path.glob("**/*.yaml"):
            if config_file.name == "baselines.yaml":
                continue
            
            if test_set_data["invocation_mode"] == "experiment":
                rel_path = config_file.parent.relative_to(search_path)
                config_name = config_file.stem
                # Construct test_id: normalize path (handle "." case)
                if str(rel_path) == ".":
                    test_id = f"{test_set_name}-{config_name}"
                else:
                    test_id = f"{test_set_name}-{rel_path}/{config_name}"
                param_rel_path = rel_path
            else: # config_name
                rel_path = Path(base_path)
                config_name = config_file.stem
                test_id = f"{test_set_name}-{config_name}"
                param_rel_path = rel_path

            all_configs.append(
                pytest.param(
                    (test_id, test_set_data, param_rel_path, config_name),
                    id=test_id,
                )
            )
    
    metafunc.parametrize("config_info", all_configs)


def _should_skip(request: pytest.FixtureRequest, test_id: str, skip_list: List[str]) -> bool:
    """Determine if a test should be skipped."""
    if any(test_id in arg for arg in request.config.invocation_params.args):
        return False
    
    # Match full test_id for experiment configs (e.g., "opi_tutorial/training")
    # and just the config name for saved_configs (e.g., "atom_han")
    config_identifier = test_id.split("-", 1)[1]
    return config_identifier in skip_list or test_id.split("-")[-1] in skip_list


def _run_smoke_test(config_info, tester: UnifiedConfigTester, request: pytest.FixtureRequest):
    """Helper to run a 1-epoch smoke test."""
    test_id, test_set_data, rel_config_path, config_name = config_info
    skip_list = test_set_data.get("skip_configs", [])

    if _should_skip(request, test_id, skip_list):
        pytest.skip(f"Config '{test_id}' temporarily excluded from CI/CD")

    tester.test_config(
        rel_config_path=rel_config_path,
        config_name=config_name,
        remove_keys=SMOKE_KEYS_TO_REMOVE,
    )


def _run_baseline_test(config_info, tester: UnifiedConfigTester, num_epochs: int, request: pytest.FixtureRequest):
    """Helper to run a baseline test for a specific number of epochs."""
    test_id, test_set_data, rel_config_path, config_name = config_info
    fixtures_path = PROJECT_ROOT / test_set_data["fixtures_path"]
    skip_list = test_set_data.get("skip_configs", [])

    if _should_skip(request, test_id, skip_list):
        pytest.skip(f"Config '{test_id}' temporarily excluded from CI/CD")

    baselines_path = fixtures_path / "baselines.yaml"
    if not baselines_path.exists():
        pytest.skip(f"No baselines.yaml found for test set '{test_id.split('-')[0]}' at {baselines_path}")

    baselines_config = load_baselines(baselines_path)
    assert baselines_config is not None, "Baselines not loaded"

    if not baselines_config.has_baseline(config_name):
        pytest.skip(f"No baseline defined for config '{config_name}' in {baselines_path}")

    model = baselines_config.get_model(config_name)
    assert model is not None, f"Model baseline not found for '{config_name}'"

    timeout = model.calculate_timeout(num_epochs)
    extra_overrides = PREDICTION_OVERRIDES + [f"trainer.max_epochs={num_epochs}"]

    if TEST_PREDS_SAVE_PATH.exists():
        TEST_PREDS_SAVE_PATH.unlink()

    std_out, _ = tester.test_config(
        rel_config_path=rel_config_path,
        config_name=config_name,
        timeout=timeout,
        remove_keys=["predictions_save_dir"],
        extra_overrides=extra_overrides,
        common_overrides=[],
    )

    val_loss = extract_val_loss(std_out)
    assert val_loss is not None, "Failed to extract val_loss from output"
    expected_val_loss = model.get_val_loss(num_epochs)
    assert val_loss == expected_val_loss, f"Expected val_loss to be {expected_val_loss}, but got {val_loss}"

    ref_filename = model.get_reference_prediction_file(num_epochs)
    if ref_filename:
        reference_preds_path = fixtures_path / "ref_preds" / ref_filename
        if reference_preds_path.exists():
            compare_predictions(
                TEST_PREDS_SAVE_PATH,
                reference_preds_path,
                tolerance=baselines_config.tolerance,
                debug_dir=PREDS_DEBUG_DIR,
            )


@pytest.mark.dependency()
def test_config_init(config_info, config_testers, request: pytest.FixtureRequest):
    """Ensure all registered configs can be initialized."""
    test_id, test_set_data, rel_config_path, config_name = config_info
    tester = config_testers[test_set_data["invocation_mode"]]
    skip_list = test_set_data.get("skip_configs", [])

    if _should_skip(request, test_id, skip_list):
        pytest.skip(f"Config '{test_id}' temporarily excluded from CI/CD")

    tester.init_config(rel_config_path, config_name)


@pytest.mark.integration
@pytest.mark.dependency()
def test_smoke(config_info, config_testers, request: pytest.FixtureRequest):
    """Run a 1-epoch smoke test (no validation)."""
    test_id, test_set_data, _, _ = config_info
    
    # Create dependency on the corresponding init test
    request.node.add_marker(
        pytest.mark.dependency(depends=[f"test_config_init[{test_id}]"])
    )
    
    tester = config_testers[test_set_data["invocation_mode"]]
    _run_smoke_test(config_info, tester, request)


@pytest.mark.integration
@pytest.mark.dependency()
def test_baseline(config_info, config_testers, request: pytest.FixtureRequest):
    """Run a 1-epoch baseline test with validation."""
    test_id, test_set_data, _, _ = config_info
    
    # Create dependency on the corresponding init test
    request.node.add_marker(
        pytest.mark.dependency(depends=[f"test_config_init[{test_id}]"])
    )
    
    tester = config_testers[test_set_data["invocation_mode"]]
    _run_baseline_test(config_info, tester, num_epochs=1, request=request)


@pytest.mark.integration
@pytest.mark.extended
@pytest.mark.dependency()
def test_extended(config_info, config_testers, request: pytest.FixtureRequest):
    """Run a 3-epoch extended baseline test with validation."""
    test_id, test_set_data, _, _ = config_info
    
    # Create dependency on the corresponding init test
    request.node.add_marker(
        pytest.mark.dependency(depends=[f"test_config_init[{test_id}]"])
    )

    if not RUN_EXTENDED_TESTS:
        pytest.skip("Extended tests disabled. Set RUN_EXTENDED_TESTS=true to run.")
        
    tester = config_testers[test_set_data["invocation_mode"]]
    _run_baseline_test(config_info, tester, num_epochs=3, request=request)
