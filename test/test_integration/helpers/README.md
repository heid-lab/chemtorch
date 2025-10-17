# Integration Test Helpers

This module provides shared utilities and presets for integration tests.

## Structure

- **`presets.py`**: Central configuration for all test settings
  - Paths (PROJECT_ROOT, CONFIG_ROOT, TEST_DIR)
  - Timeout defaults (PRECOMPUTE_TIMEOUT, DEFAULT_TIMEOUT)
  - Common test overrides (BASE_OVERRIDES)

- **`baselines.py`**: Load and manage baseline reference data from YAML
  - Reads from `fixtures/baselines.yaml`
  - Falls back to presets for timeout values
  - Models only specify values that differ from defaults

- **`comparison.py`**: Prediction comparison utilities
  - Position-wise CSV comparison with tolerance
  - Debug output generation on failure

- **`config_discovery.py`**: YAML config discovery for pytest parametrization
  - `discover_yaml_configs()`: Find all .yaml files in a directory tree
  - `parametrize_config_tests()`: Generate pytest parameters from discovered configs

- **`config_tester.py`**: Abstract base class for config testing
  - Uses BASE_OVERRIDES from constants by default
  - Handles Hydra initialization and subprocess execution
  - Subclass to implement specific test behavior

- **`extraction.py`**: Extract metrics from test outputs
  - Parse validation loss from Lightning output

## Usage

```python
from test.test_integration.helpers import (
    BASE_OVERRIDES,
    CONFIG_ROOT,
    ConfigTester,
    compare_predictions,
    load_baselines,
    parametrize_config_tests,
)

# Constants are ready to use
timeout = DEFAULT_TIMEOUT * 3

# Load baselines once at session start
@pytest.fixture(scope="session", autouse=True)
def baselines():
    return load_baselines(BASELINES_PATH)

# Use in tests
def test_my_config(config_info):
    baselines_config = get_baselines()
    model = baselines_config.get_model("cgr_dmpnn")
    timeout = model.calculate_timeout(num_epochs=3)
```