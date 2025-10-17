# Integration Test Suite

Integration tests for ChemTorch configurations, ensuring configs load, execute correctly, and produce consistent results.

## Quick Start

```bash
# Run all integration tests
pytest test/test_integration/

# Run experiment config smoke tests (fast)
pytest test/test_integration/test_experiment_configs.py

# Run saved config validation tests
pytest test/test_integration/test_saved_configs.py

# Run specific model
pytest test/test_integration/test_saved_configs.py::test_saved_config_1_epoch[cgr_dmpnn] -s

# Run with 3-epoch extended tests
RUN_EXTENDED_TESTS=true pytest test/test_integration/test_saved_configs.py -v
```

## Directory Structure

```
test/test_integration/
├── fixtures/                   # Reference data & baselines
│   ├── baselines.yaml         # Expected losses, timeouts
│   └── ref_preds/             # Reference prediction CSVs
├── debug_preds/               # Auto-generated debug CSVs on failure
├── helpers/                   # Shared utilities (see helpers/README.md)
│   ├── presets.py            # Constants: paths, timeouts, overrides
│   ├── baselines.py          # Baseline loader
│   ├── comparison.py         # Prediction comparison
│   └── ...                   # Other utilities
├── test_experiment_configs.py # Smoke tests for conf/experiment/
└── test_saved_configs.py      # Validation tests for saved configs
```

## How to Extend the Test Suite

### Adding a New Saved Config Test

**1. Add your config** to `conf/saved_configs/chemtorch_benchmark/optimal_model_configs/`:
```bash
cp my_new_model.yaml conf/saved_configs/chemtorch_benchmark/optimal_model_configs/
```

**2. Generate reference data** by running the config once:
```bash
python chemtorch_cli.py \
  --config-path conf/saved_configs/chemtorch_benchmark/optimal_model_configs \
  --config-name my_new_model \
  ++data_module.subsample=0.01 \
  ++save_predictions_for=test \
  ++predictions_save_path=test/test_integration/fixtures/ref_preds/rdb7_subsample_0.01_my_new_model_seed_0_epoch_1.csv \
  trainer.max_epochs=1

# Note the final val_loss_epoch value from output
```

**3. Add baseline** to `fixtures/baselines.yaml`:
```yaml
models:
  my_new_model:
    val_loss:
      epoch_1: 1.25    # From step 2
    reference_predictions:
      epoch_1: "rdb7_subsample_0.01_my_new_model_seed_0_epoch_1.csv"
    # Optional: only if different from defaults (30s precompute, 60s/epoch)
    timeout:
      per_epoch: 90    # Override if needed
```

**4. Test it**:
```bash
pytest test/test_integration/test_saved_configs.py::test_saved_config_1_epoch[my_new_model] -v
```

### Adding 3-Epoch Extended Tests

**1. Generate 3-epoch reference**:
```bash
python chemtorch_cli.py \
  --config-path conf/saved_configs/chemtorch_benchmark/optimal_model_configs \
  --config-name my_new_model \
  ++data_module.subsample=0.01 \
  ++save_predictions_for=test \
  ++predictions_save_path=test/test_integration/fixtures/ref_preds/rdb7_subsample_0.01_my_new_model_seed_0_epoch_3.csv \
  trainer.max_epochs=3
```

**2. Update baselines**:
```yaml
models:
  my_new_model:
    val_loss:
      epoch_1: 1.25
      epoch_3: 1.10    # Add this
    reference_predictions:
      epoch_1: "rdb7_subsample_0.01_my_new_model_seed_0_epoch_1.csv"
      epoch_3: "rdb7_subsample_0.01_my_new_model_seed_0_epoch_3.csv"  # Add this
```

**3. Test**:
```bash
RUN_EXTENDED_TESTS=true pytest test/test_integration/test_saved_configs.py::test_saved_config_3_epoch[my_new_model] -v
```

### Adding a New Experiment Config

Simply add your config YAML to `conf/experiment/` or any subdirectory:

```bash
# Config is auto-discovered, no setup needed
echo "# My new experiment config" > conf/experiment/my_new_experiment.yaml

# Test will automatically run
pytest test/test_integration/test_experiment_configs.py::test_experiment_config_exec[my_new_experiment] -v
```

### Temporarily Excluding Configs from CI/CD

If a config is broken and you need to temporarily exclude it from CI/CD:

**For experiment configs**, edit `test_experiment_configs.py`:
```python
SKIP_CONFIGS = [
    "opi_tutorial/training",  # TODO: Fix and re-enable
    "my_broken_config",
]
```

**For saved configs**, edit `test_saved_configs.py`:
```python
SKIP_CONFIGS = [
    "atom_han",  # TODO: Fix and re-enable
    "my_broken_model",
]
```

Skipped configs will show as `SKIPPED` in test output with a clear message.
## Debugging Failed Tests

### Validation Loss Mismatch
```
AssertionError: Expected val_loss to be 1.36, but got 1.37
```
→ Update baseline in `fixtures/baselines.yaml` or check for data/training changes.

### Prediction Mismatch
```
Found 5 lines with prediction mismatches:
  Line 10: Value exceeds tolerance (test=2.3456789, ref=2.3456788, diff=1.1e-07)
Debug CSV saved to: test/test_integration/debug_preds/...
```
→ Check debug CSV. If acceptable, regenerate reference predictions (see "Adding New Saved Config Test").

## Configuration

### Adjust Global Defaults

Edit `helpers/presets.py`:
```python
PRECOMPUTE_TIMEOUT = 30  # seconds for data loading
DEFAULT_TIMEOUT = 60     # seconds per epoch
BASE_OVERRIDES = [...]   # Applied to all tests
```

### Adjust Tolerance

Edit `fixtures/baselines.yaml`:
```yaml
defaults:
  tolerance: 1.0e-5  # Increase for more numerical tolerance
```

### Custom Timeouts

In `fixtures/baselines.yaml`:
```yaml
models:
  slow_model:
    timeout:
      precompute: 60   # Override 30s default
      per_epoch: 300   # Override 60s default
```

## Further Documentation

- **Helpers**: See `helpers/README.md` for implementation details
- **Test patterns**: Use `-k` flag to filter tests by name pattern