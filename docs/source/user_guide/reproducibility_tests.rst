.. _reproducibility-tests:

=====================
Reproducibility Tests
=====================

This page explains how to:

#. set up reproducibility checks for saved configs using the integration test registry,
#. add baseline reference values against which to test, and 
#. run and debug these checks locally or in CI.

ChemTorch ships with a configurable, registry-driven integration test suite focused on catching reproducibility regressions early.
It verifies:

- Configs can be instantiated without errors
- Training runs for at least one epoch per dataloader
- Saved configs still reproduce reference results (validation loss and predictions)

.. note::
  Before you continue, make sure the :ref:`testing dependencies<dev-deps>` are installed.


Overview
========

Integration tests are defined in ``test/test_integration/`` and driven by a single registry file, ``test_registry.yaml``. Each entry in the registry declares a "test set" (for example, all experiment configs, or a curated set of saved configs) and which test cases to run for that set.

Directory Layout
----------------

::

   test/test_integration/
   ├── test_registry.yaml         # Central registry for all test sets
   ├── test_integration.py        # Unified test implementation
   ├── fixtures/                  # Reference data organized by test set
   │   └── <test_set_name>/
   │       ├── baselines.yaml     # Baseline metadata (expected losses, timeouts, references)
   │       └── ref_preds/         # Reference prediction CSVs
   ├── debug/                     # Auto-generated debug CSVs on failure
   └── helpers/                   # Shared utilities used by the suite


Quick Start
===========

.. code-block:: bash

  # Run a single saved-config baseline by parameter ID (most common)
  pytest test/test_integration/test_integration.py::test_baseline[chemtorch_benchmark-cgr_dmpnn] -v

  # Run all reproducibility baselines for a test set
  pytest test/test_integration/test_integration.py::test_baseline -k "chemtorch_benchmark" -v

   # Run extended (3-epoch) tests
   export RUN_EXTENDED_TESTS=true 
   pytest test/test_integration/test_integration.py::test_extended -k "chemtorch_benchmark" -v

  # Discover parameter IDs for a test set (to target a single config)
   pytest -q test/test_integration/test_integration.py --collect-only -k "chemtorch_benchmark"

.. tip::
  Execution-only smoke tests for experiment configs are documented in the Developer Guide. See :ref:`testing`.


Test Types
==========

The suite supports three complementary test types, with an emphasis on reproducibility for saved configs:

- **Reproducibility** (``test_baseline`` / ``test_extended``): run training and validate against stored baselines and reference predictions. This is the primary check for experimental results.
- **Instantiation** (``test_config_init``): ensure a config loads without errors (very fast).
- **Execution** (``test_smoke``): run one epoch per dataloader to catch runtime issues quickly.

.. note::
   Extended tests (3 epochs) are opt-in via the environment variable ``RUN_EXTENDED_TESTS=true`` so CI can skip them by default.


Registry Configuration
======================

Test sets are declared in ``test_registry.yaml``. Each set specifies where to find configs, how to invoke them, and which test cases to run.

Fields
------

- ``enabled``: set ``false`` to disable the entire set
- ``invocation_mode``: how configs are invoked

  - ``experiment``: for experiment configs that are invoked like 
    
    .. code-block:: chemtorch

        chemtorch +experiment=path/config_name

  - ``config_name``: for saved configs which are invoked like

    .. code-block:: chemtorch

       chemtorch \
         --config-dir conf/saved_configs/my_models \
         --config-name my_model

- ``path``: directory containing config files (relative to project root)
- ``test_cases``: list of test types to run. Any of 
    - ``init``
    - ``smoke``
    - ``baseline``
    - ``extended``
- ``fixtures_path``: path to ``baselines.yaml`` and ``ref_preds/`` (required only for reproducability tests)
- ``skip_configs`` (optional): list of config names (or paths for experiment sets) to temporarily exclude
- ``force_debug_log`` (optional): save detailed CSVs when prediction comparisons fail
- ``overrides`` (optional): mapping of config identifiers to lists of Hydra overrides appended at runtime; see :ref:`dev-config-overrides` in the Developer Guide for details and examples.

Example (saved configs)
-----------------------

.. code-block:: yaml

   test_sets:
     # Detects regressions in reproducibility for experiments from the paper
     chemtorch_benchmark:
       enabled: true
       invocation_mode: config_name
       path: conf/saved_configs/chemtorch_benchmark/optimal_model_configs
       test_cases:
         - init
         - baseline
         - extended
       fixtures_path: test/test_integration/fixtures/chemtorch_benchmark

.. note::
   CI/CD smoke testing for experiment configs (``experiment_configs``) is covered in the Developer Guide. See :ref:`testing`.

.. note::
   For how to use per‑config overrides in integration tests (including ordering with default removals and practical examples), see :ref:`dev-config-overrides` in the Developer Guide.


Saved Configs with Baseline Validation
======================================

To ensure saved configs remain reproducible, create a fixtures directory with a baseline file and reference predictions.

1) Create fixtures directories
------------------------------

.. code-block:: bash

   mkdir -p test/test_integration/fixtures/my_models/ref_preds

2) Generate reference predictions
---------------------------------

.. code-block:: chemtorch

   chemtorch \
     --config-dir conf/saved_configs/my_models \
     --config-name my_model \
     ++data_module.subsample=0.01 \
     ++save_predictions_for=test \
     ++predictions_save_path=test/test_integration/fixtures/my_models/ref_preds/my_model_epoch_1.csv \
     trainer.max_epochs=1

Record the final ``val_loss_epoch`` from the CLI output (e.g., ``1.25``).

3) Define baselines
-------------------

.. code-block:: yaml

   defaults:
     tolerance: 1e-5   # Set based on precision of target labels

   models:
     my_model:
       val_loss:
         epoch_1: 1.25  # From step 2
       reference_predictions:
         epoch_1: my_model_epoch_1.csv
       # Optional: customize timeouts if needed
       timeout:
         precompute: 30  # seconds for data preparation (default: 30)
         per_epoch: 60   # seconds per epoch (default: 60)

4) Register the test set
------------------------

.. code-block:: yaml

   test_sets:
     my_models:
       enabled: true
       invocation_mode: config_name
       path: conf/saved_configs/my_models
       test_cases:
         - init
         - baseline
       fixtures_path: test/test_integration/fixtures/my_models

5) Run
------

.. code-block:: bash

   # Run all tests for your set
   pytest test/test_integration/test_integration.py -k "my_models" -v

   # Only init tests
   pytest test/test_integration/test_integration.py::test_config_init -k "my_models" -v

   # Single model by parameter ID
   pytest test/test_integration/test_integration.py::test_baseline[my_models-my_model] -v

.. tip::
   Discover the exact parameter IDs generated for your set with:

   ``pytest -q test/test_integration/test_integration.py --collect-only -k "my_models"``


Extended Tests (3 Epochs)
=========================

To validate over 3 epochs, generate reference predictions and values for epoch 3 and enable the ``extended`` test case.

.. code-block:: bash

   chemtorch \
     --config-path conf/saved_configs/my_models \
     --config-name my_model \
     ++data_module.subsample=0.01 \
     ++save_predictions_for=test \
     ++predictions_save_path=test/test_integration/fixtures/my_models/ref_preds/my_model_epoch_3.csv \
     trainer.max_epochs=3

.. code-block:: yaml

   models:
     my_model:
       val_loss:
         epoch_1: 1.25
         epoch_3: 1.10
       reference_predictions:
         epoch_1: my_model_epoch_1.csv
         epoch_3: my_model_epoch_3.csv

.. code-block:: yaml

   test_sets:
     my_models:
       test_cases: ["init", "baseline", "extended"]

.. code-block:: bash

   RUN_EXTENDED_TESTS=true pytest test/test_integration/test_integration.py::test_extended -k "my_models" -v


Managing Test Sets
==================

Temporarily skip configs
------------------------

.. code-block:: yaml

   test_sets:
     my_models:
       skip_configs:
         - broken_model
         - experimental_v2

Disable entire test sets
------------------------

.. code-block:: yaml

   test_sets:
     my_old_models:
       enabled: false


Debugging Failed Tests
======================

Common issues and tips:

- Validation loss mismatch:

  .. code-block:: text

    AssertionError: Expected val_loss to be 1.36, but got 1.37

  - Compare runs in the Weights & Biases dashboard (*Workspace → Add panels → Run comparer*)
  - Check for hardware differences (low-level kernels can vary across architectures)
  - Reproduce the baseline on the original hardware; if acceptable, update the baseline values

- Prediction mismatch:

  .. code-block:: text

    Found 5 lines with prediction mismatches:
      Line 10: Value exceeds tolerance (test=2.3456789, ref=2.3456788, diff=1.1e-07)
    Debug CSV saved to: test/test_integration/debug/...

  When a mismatch exceeds tolerance, the suite will emit debug CSVs under ``test/test_integration/debug/`` (enable via ``force_debug_log``).


Hardware & environment note
---------------------------

Numerical results and model predictions can vary with the underlying hardware and low-level libraries. Differences can arise from:

- CPU architecture and instruction set (e.g. x86_64 vs ARM; AVX/AVX2/AVX-512 availability)
- GPU model, driver, CUDA and cuDNN versions
- BLAS provider and configuration (OpenBLAS, MKL, BLIS, etc.)
- PyTorch build flags, compiler, and Python / NumPy versions

Recommendations:

- Generate reference predictions on the same class of hardware where tests will run (your laptop/CI runner).
- Record hardware + software metadata with each baseline (e.g. CPU model, GPU model, CUDA/cuDNN, PyTorch, Python, NumPy, BLAS). Example YAML metadata block:

  .. code-block:: yaml

     metadata:
       hardware:
         cpu: "Intel(R) Xeon(R) Gold 6248"
         gpu: "NVIDIA V100"
       software:
         python: "3.10.18"
         pytorch: "2.2.0"
         cuda: "11.8"
         numpy: "1.25.0"
         blas: "OpenBLAS"

- Use containerisation (Docker) or pinned environments to reduce drift between generation and test environments.
- If you must run tests on different hardware than the one used to generate references, either regenerate baselines on the target hardware or increase numeric tolerances and document the rationale.

Helper script
-------------

We include a small helper script that collects common hardware and software metadata and writes it as YAML. Run it on the machine used to generate reference predictions and commit the resulting YAML alongside your baselines.

.. code-block:: bash

  # write to the fixtures directory next to your baseline YAML
  python scripts/collect_env.py --out test/test_integration/fixtures/my_models/baseline_env.yaml

The generated YAML includes CPU information, any NVIDIA GPU details exposed via `nvidia-smi`, key package versions (PyTorch, NumPy, etc.), a short `pip freeze` snapshot, and a NumPy/BLAS configuration block when available.

When updating baselines, include the hardware/software metadata and the reason for the change in the commit message so future investigators can trace regressions.


Test Parameter IDs
==================

Parameter IDs follow:

::

   <test_set_name>-<config_name>

For experiment configs inside subfolders, the relative path is included (for example, ``experiment_configs-subdir/config_name``).