.. _testing:
.. briefly describe out testing setup: 
.. 1. unit tests for code, 
.. 2. integration tests for configs,
.. 3. additional reproducability tests for experiments

=======
Testing
=======

This page summarizes ChemTorch's testing strategy and how to run tests locally and in CI.

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
	- Documented for experimentalists in :ref:`integration_tests`


Integration tests for experiment configs (CI/CD)
===============================================

ChemTorch runs smoke tests on experiment configs in CI to catch breaking changes (e.g., API changes or runtime errors).

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

For reproducibility checks on saved configs (including baselines and reference predictions), see :ref:`integration_tests`.