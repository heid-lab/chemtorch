# Integration tests (quick reference)

This folder contains the integration test suite for ChemTorch. For full documentation, see the hosted docs:

- Advanced Guide -> Reproducibility Regression Checks (experimentalists):
  https://heid-lab.github.io/chemtorch/advanced_guide/integration_tests.html
- Developer Guide -> Testing (CI/CD):
  https://heid-lab.github.io/chemtorch/developer_guide/testing.html

## Quick start

```bash
# Run all tests
pytest test/test_integration/test_integration.py -v

# Run all reproducibility baselines for a test set
pytest test/test_integration/test_integration.py::test_baseline -k "chemtorch_benchmark" -v

# Run a single saved-config baseline by parameter ID
pytest test/test_integration/test_integration.py::test_baseline[chemtorch_benchmark-cgr_dmpnn] -v

# Run extended (3-epoch) tests
RUN_EXTENDED_TESTS=true pytest test/test_integration/test_integration.py::test_extended -k "chemtorch_benchmark" -v

# Discover parameter IDs for a test set
pytest -q test/test_integration/test_integration.py --collect-only -k "chemtorch_benchmark"
```

Tip: Test parameter IDs are of the form `<test_set>-<config_name>`. For experiment configs inside subfolders, the relative path is included (e.g. `experiment_configs-subdir/config_name`).