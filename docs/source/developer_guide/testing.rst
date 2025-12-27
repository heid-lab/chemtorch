.. _testing:

=======
Testing
=======

This page summarizes ChemTorch's testing strategy and how to run tests locally and in CI.

.. note::
  Before you continue, make sure the :ref:`testing dependencies<dev-deps>` are installed.

Layers of testing
=================

1. Unit tests (code-level)
	- Small, focused tests for core utilities and components
	- Live under ``test/`` (e.g., ``test/test_scheduler``, ``test/test_routine``)
	- Coming soon: contributor guide for writing unit tests

2. Integration tests (config-level)
	- Ensure Hydra configs can be instantiated and executed without errors
	- Implemented in ``test/test_integration/test_integration.py`` and driven by ``test_registry.yaml``

3. Reproducibility checks (saved configs)
	- Validate that saved configs still reproduce baseline validation loss and predictions
	- Documented for experimentalists in :ref:`reproducibility-tests`


Integration tests for experiment configs (CI/CD)
===============================================
ChemTorch runs smoke tests on experiment configs in CI to catch breaking changes (e.g., API changes or runtime errors).
This tutorial assumes that you are already familiar with the structure and usage of the integration test suite as documented in :ref:`reproducibility-tests`.

Register the CI test set
------------------------

Add (or keep) the following entry in ``test/test_integration/test_registry.yaml``:

.. code-block:: yaml

	test_sets:
	  experiment_configs:
		 enabled: true
		 invocation_mode: experiment
		 path: conf/experiment
		 test_cases:
			- init
			- smoke

Run locally
-----------

.. code-block:: bash

	# Run only experiment config smoke tests
	pytest test/test_integration/test_integration.py::test_smoke -k "experiment_configs" -v

	# Or just ensure configs load
	pytest test/test_integration/test_integration.py::test_config_init -k "experiment_configs" -v

Target specific configs
-----------------------

Parametrized test IDs follow ``<test_set>-<config_path_or_name>``. Use ``--collect-only`` to discover IDs:

.. code-block:: bash

	pytest -q test/test_integration/test_integration.py --collect-only -k "experiment_configs"

Then target a specific config:

.. code-block:: bash

	pytest test/test_integration/test_integration.py::test_smoke[experiment_configs-subdir/my_config] -v

Temporary exclusions
--------------------

To keep CI green, temporarily skip configs via ``skip_configs`` in the registry:

.. code-block:: yaml

	test_sets:
	  experiment_configs:
		 skip_configs:
			- opi_tutorial/training  # TODO: fix & re-enable

Extended tests
--------------

Three-epoch "extended" tests are gated by the environment variable ``RUN_EXTENDED_TESTS`` and are typically disabled in CI:

.. code-block:: bash

	RUN_EXTENDED_TESTS=true pytest test/test_integration/test_integration.py::test_extended -k "experiment_configs" -v

For reproducibility checks on saved configs (including baselines and reference predictions), see :ref:`reproducibility-tests`.

.. _dev-config-overrides:

Config-specific overrides for integration tests
===============================================

Some configurations cannot run in CI unchanged because they rely on external artifacts (for example, training checkpoints) or require mandatory runtime settings (for example, an explicit predictions output path in inference mode).

Problem examples
----------------

- Inference runs (``tasks='[predict]'``, see :ref:`tasks-exec-control`) must set either ``predictions_save_path`` or ``predictions_save_dir``.
- Inference configs commonly set ``load_model=true`` and reference a checkpoint via ``ckpt_path`` that will not be present on CI workers.
- Tutorials or examples may point to datasets that are not committed to the repository.

Solution: patch the test invocation
-----------------------------------

Do not change production configs to satisfy CI. Instead, declare per‑config overrides in ``test/test_integration/test_registry.yaml``. The integration test runner applies its suite‑wide removals first (e.g. ``SMOKE_KEYS_TO_REMOVE``) and then appends your overrides. That ordering lets you reintroduce removed keys with safe values and redirect paths to lightweight fixtures.

Example
-------

.. code-block:: yaml

	 test_sets:
		 experiment_configs:
			 overrides:
				 prediction_pipeline/inference:
					 - ++data_module.data_pipeline.data_source.data_path=test/test_integration/fixtures/prediction_pipeline/inference.csv
					 - ++load_model=false
					 - ~ckpt_path
					 - ++predictions_save_path=/tmp/chemtorch_test_preds.csv

In this setup, the smoke test runs on a tiny CSV fixture and writes predictions to a temporary file while bypassing checkpoint loading. Overrides apply to all test cases that execute the config; ``init`` only composes the config, so runtime‑only overrides have no effect there.