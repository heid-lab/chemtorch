# Integration Test Suite

Integration tests for ChemTorch configurations, ensuring configs load, execute correctly, and produce consistent results.

## Quick Start

```bash
# Run all integration tests
pytest test/test_integration/test_integration.py

# Run specific test type
pytest test/test_integration/test_integration.py::test_smoke -v
pytest test/test_integration/test_integration.py::test_baseline -v

# Run specific config
pytest test/test_integration/test_integration.py::test_baseline[chemtorch_benchmark-cgr_dmpnn] -v

# Run with 3-epoch extended tests
RUN_EXTENDED_TESTS=true pytest test/test_integration/test_integration.py::test_extended -v
```

## Overview

The integration test suite uses a **registry-based system** (`test_registry.yaml`) to manage test sets. This allows you to:
- Define custom test sets with different configurations
- Enable/disable test sets without modifying test code
- Run different test types (init, smoke, baseline, extended) for each test set
- Specify custom fixtures, timeouts, and skip lists per test set

### Two Types of Integration Tests

**1. Experiment Config Tests (Smoke Tests)**
- **Purpose**: Detect when experiment configs break (e.g., due to API changes)
- **What it does**: Instantiates each config and runs one epoch for each data loader
- **Default behavior**: Includes all configs in `conf/experiment/` and subdirectories
- **Test types**: `init` (config loads) and `smoke` (config runs for 1 epoch)

**2. Saved Config Tests (Baseline Validation)**
- **Purpose**: Ensure saved configs can reproduce previous results
- **What it does**: Runs the config and compares validation loss and predictions against reference data
- **Requires**: Reference predictions and baseline values in `fixtures/` directory
- **Test types**: `init`, `baseline` (1 epoch + validation), `extended` (3 epochs + validation)

## Directory Structure

```
test/test_integration/
├── test_registry.yaml         # Central registry for all test sets
├── test_integration.py        # Unified test implementation
├── fixtures/                  # Reference data organized by test set
│   └── <test_set_name>/
│       ├── baselines.yaml     # Expected losses, timeouts
│       └── ref_preds/         # Reference prediction CSVs
├── debug/                     # Auto-generated debug CSVs on failure
└── helpers/                   # Shared utilities (see helpers/README.md)
    ├── presets.py            # Constants: paths, timeouts, overrides
    ├── baselines.py          # Baseline loader
    ├── comparison.py         # Prediction comparison
    └── ...                   # Other utilities
```


## Test Registry Configuration

The test registry (`test_registry.yaml`) defines all test sets and their properties:

```yaml
test_sets:
  chemtorch_benchmark:
    path: "conf/saved_configs/chemtorch_benchmark/optimal_model_configs"
    enabled: true
    invocation_mode: "config_name"
    test_cases: ["init", "baseline"]
    fixtures_path: "test/test_integration/fixtures/chemtorch_benchmark"
    skip_configs: ["atom_han"]
    force_debug_log: true
  
  experiment_configs:
    path: "conf/experiment"
    enabled: true
    invocation_mode: "experiment"
    test_cases: ["init", "smoke"]
    skip_configs: ["opi_tutorial/training"]
```

### Test Set Fields

- **`path`**: Directory containing config files (relative to project root)
- **`enabled`**: Set to `false` to disable the entire test set
- **`invocation_mode`**: How to invoke configs
  - `"experiment"`: Uses `+experiment=path/config` (for `conf/experiment/` configs)
  - `"config_name"`: Uses `--config-path` and `--config-name` (for saved configs)
- **`test_cases`**: List of test types to run. Available options:
  - `"init"`: Check that config can be loaded (fast, always recommended)
  - `"smoke"`: Run for 1 epoch without validation (quick execution test)
  - `"baseline"`: Run for 1 epoch with validation against reference data
  - `"extended"`: Run for 3 epochs with validation (requires `RUN_EXTENDED_TESTS=true`)
- **`fixtures_path`**: Directory containing `baselines.yaml` and `ref_preds/` (required for `baseline` and `extended` tests)
- **`skip_configs`**: List of config names to temporarily exclude from tests
- **`force_debug_log`** *(optional)*: Generate debug CSVs on prediction mismatches

## Adding Your Own Test Set

### Example 1: Adding Experiment Configs (Smoke Tests Only)

If you want to test a new set of experiment configs:

**1. Add to test registry** (`test_registry.yaml`):
```yaml
test_sets:
  my_experiments:
    path: "conf/experiment/my_project"
    enabled: true
    invocation_mode: "experiment"
    test_cases: ["init", "smoke"]  # No baseline validation needed
    skip_configs: []
```

**2. Run the tests**:
```bash
pytest test/test_integration/test_integration.py -k "my_experiments" -v
```

That's it! The test suite will automatically discover all YAML files in `conf/experiment/my_project/` and test them.

### Example 2: Adding Saved Configs with Baseline Validation

For saved configs that need to reproduce results:

**1. Create fixtures directory**:
```bash
mkdir -p test/test_integration/fixtures/my_models
mkdir -p test/test_integration/fixtures/my_models/ref_preds
```

**2. Generate reference predictions** for each config:
```bash
python chemtorch_cli.py \
  --config-path conf/saved_configs/my_models \
  --config-name my_model \
  ++data_module.subsample=0.01 \
  ++save_predictions_for=test \
  ++predictions_save_path=test/test_integration/fixtures/my_models/ref_preds/my_model_epoch_1.csv \
  trainer.max_epochs=1
```

Note the final `val_loss_epoch` value from the output (e.g., 1.25).

**3. Create baselines file** (`fixtures/my_models/baselines.yaml`):
```yaml
defaults:
  tolerance: 1e-5

models:
  my_model:
    val_loss:
      epoch_1: 1.25  # From step 2
    reference_predictions:
      epoch_1: "my_model_epoch_1.csv"
    # Optional: customize timeouts if needed
    timeout:
      precompute: 30  # seconds for data loading (default: 30)
      per_epoch: 60   # seconds per epoch (default: 60)
```

**4. Add to test registry** (`test_registry.yaml`):
```yaml
test_sets:
  my_models:
    path: "conf/saved_configs/my_models"
    enabled: true
    invocation_mode: "config_name"
    test_cases: ["init", "baseline"]  # or ["init", "baseline", "extended"]
    fixtures_path: "test/test_integration/fixtures/my_models"
    skip_configs: []
```

**5. Run the tests**:
```bash
# Run all tests for your models
pytest test/test_integration/test_integration.py -k "my_models" -v

# Run just baseline tests
pytest test/test_integration/test_integration.py::test_baseline -k "my_models" -v

# Run a specific model
pytest test/test_integration/test_integration.py::test_baseline[my_models-my_model] -v
```

### Adding Extended (3-Epoch) Tests

To add 3-epoch validation:

**1. Generate 3-epoch reference**:
```bash
python chemtorch_cli.py \
  --config-path conf/saved_configs/my_models \
  --config-name my_model \
  ++data_module.subsample=0.01 \
  ++save_predictions_for=test \
  ++predictions_save_path=test/test_integration/fixtures/my_models/ref_preds/my_model_epoch_3.csv \
  trainer.max_epochs=3
```

**2. Update baselines** (`fixtures/my_models/baselines.yaml`):
```yaml
models:
  my_model:
    val_loss:
      epoch_1: 1.25
      epoch_3: 1.10  # Add this
    reference_predictions:
      epoch_1: "my_model_epoch_1.csv"
      epoch_3: "my_model_epoch_3.csv"  # Add this
```

**3. Enable extended tests** in registry:
```yaml
test_sets:
  my_models:
    test_cases: ["init", "baseline", "extended"]  # Add "extended"
```

**4. Run extended tests**:
```bash
RUN_EXTENDED_TESTS=true pytest test/test_integration/test_integration.py::test_extended -k "my_models" -v
```
## Managing Test Sets

### Temporarily Excluding Configs

If a config is broken, you can exclude it from CI/CD via the `skip_configs` field:

```yaml
test_sets:
  my_models:
    skip_configs:
      - "broken_model"      # TODO: Fix and re-enable
      - "experimental_v2"   # Work in progress
```

Skipped configs will show as `SKIPPED` in test output.

### Disabling Entire Test Sets

To disable an entire test set temporarily:

```yaml
test_sets:
  my_old_models:
    enabled: false  # Set to false
    # ... rest of config
```

### Selectively Running Test Types

Control which tests run for each set via `test_cases`:

```yaml
test_sets:
  quick_checks:
    test_cases: ["init"]  # Only check configs load
  
  smoke_tests:
    test_cases: ["init", "smoke"]  # Check configs run
  
  full_validation:
    test_cases: ["init", "baseline", "extended"]  # Full validation
```

## Debugging Failed Tests

### Validation Loss Mismatch
```
AssertionError: Expected val_loss to be 1.36, but got 1.37
```
→ Update baseline in your test set's `fixtures/<test_set>/baselines.yaml` or investigate training changes.

### Prediction Mismatch
```
Found 5 lines with prediction mismatches:
  Line 10: Value exceeds tolerance (test=2.3456789, ref=2.3456788, diff=1.1e-07)
Debug CSV saved to: test/test_integration/debug/...
```
→ Check debug CSV in `test/test_integration/debug/`. If acceptable, regenerate reference predictions.

### Config Initialization Failed
```
Error: Could not instantiate class XYZ
```
→ Check for API changes or missing dependencies. Update config or fix code.

## Configuration Options

### Adjust Global Defaults

Edit `helpers/presets.py`:
```python
PRECOMPUTE_TIMEOUT = 30  # seconds for data loading
DEFAULT_TIMEOUT = 60     # seconds per epoch
BASE_OVERRIDES = [...]   # Applied to all tests
```

### Adjust Tolerance

Edit your test set's `baselines.yaml`:
```yaml
defaults:
  tolerance: 1.0e-5  # Increase for more numerical tolerance
```

### Custom Timeouts

In your test set's `baselines.yaml`:
```yaml
models:
  slow_model:
    timeout:
      precompute: 60   # Override 30s default
      per_epoch: 300   # Override 60s default
```

## Advanced Usage

### Running Specific Test Sets or configs
The pytest `-k` option matches patterns in test names.
```bash
# Run only chemtorch_benchmark tests
pytest test/test_integration/test_init.py -k "chemtorch_benchmark" -v

# Run only the cgr_dmpnn config from chemtorch_benchmarks
pytest test/test_integration/test_integration.py -k "experiment_configs-cgr_dmpnn" -v
```

### Verbose Output
```bash
# Show full output (disable capturing)
pytest test/test_integration/test_integration.py -s -v

# Show only test names and results
pytest test/test_integration/test_integration.py -v
```

## Impementation Details
See `helpers/README.md` for implementation details of comparison, baseline loading, etc.