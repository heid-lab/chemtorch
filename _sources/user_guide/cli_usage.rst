.. _cli-usage:

==============================
CLI Usage & Config Reference
==============================

This page provides a comprehensive reference for using the ChemTorch command-line interface (CLI) and all ChemTorch-specific configuration options.


Command-Line Interface
======================

Basic Usage
-----------

The ChemTorch CLI follows this general pattern:

.. code-block:: bash

    chemtorch +experiment=NAME [key=value] [key.subkey=value] ...

**Examples:**

.. code-block:: bash

    # Run an experiment
    chemtorch +experiment=graph
    
    # Override values
    chemtorch +experiment=graph trainer.max_epochs=200 seed=42
    
    # Change a component
    chemtorch +experiment=graph model=gcn


Override Syntax
---------------

**Nested keys** - Use dot notation:

.. code-block:: bash

    chemtorch +experiment=graph model.hidden_channels=256

**Component selection** - Swap entire configs:

.. code-block:: bash

    chemtorch +experiment=graph model=gcn

**Lists** - Use square brackets with single/double quotes:

.. code-block:: bash

    chemtorch +experiment=graph tasks='[fit,test]'

There must be no white space before and after the comma!
Otherwise Hydra will throw an error.

**Booleans**:

.. code-block:: bash

    chemtorch +experiment=graph log=true

**Null values** - Use ``~`` prefix:

.. code-block:: bash

    chemtorch +experiment=graph ~group_name

You cannot use ``key=null`` because Hydra intreprets ``null`` as the string ``"null"`` when passed via the CLI.
Importantly, ``~`` removes the key but this only works if the key is present in the config in the first place.
Otherwise, Hydra will throw an error.


**Strings with spaces** - Use quotes:

.. code-block:: bash

    chemtorch +experiment=graph run_name="my experiment"

**Multiple runs** - Use multirun mode (``-m``):

.. code-block:: bash

    chemtorch -m +experiment=graph seed=0,1,2


ChemTorch Configuration Reference
==================================

This section documents all ChemTorch-specific configuration options that can be set in experiment configs or via the CLI.


Execution Control
-----------------

**tasks**

Controls which stages of the pipeline to execute.

* **Type**: List of strings
* **Options**: ``fit``, ``test``, ``validate``, ``predict``
* **Default**: ``null`` (must be specified)

.. code-block:: yaml

    tasks:
      - fit
      - test

.. code-block:: bash

    chemtorch +experiment=graph tasks='[fit,test]'

* ``fit``: Training and validation
* ``test``: Evaluate on test set(s)
* ``validate``: Evaluate on validation set only
* ``predict``: Run inference without training (see :ref:`inference`)


**seed**

Random seed for reproducibility.

* **Type**: Integer
* **Default**: ``0``

.. code-block:: yaml

    seed: 42

.. code-block:: bash

    chemtorch +experiment=graph seed=42

Sets the random seed for Python, NumPy, PyTorch, and PyTorch Lightning to ensure reproducible results.


Logging (Weights & Biases)
---------------------------

**log**

Enable or disable Weights & Biases logging.

* **Type**: Boolean
* **Default**: ``false``

.. code-block:: yaml

    log: true

.. code-block:: bash

    chemtorch +experiment=graph log=true


**project_name**

W&B project name for organizing runs.

* **Type**: String
* **Default**: ``"chemtorch"``

.. code-block:: yaml

    project_name: my_project

.. code-block:: bash

    chemtorch +experiment=graph project_name=my_project


**group_name**

W&B group name for organizing related runs (e.g., hyperparameter sweeps).

* **Type**: String or null
* **Default**: ``null``

.. code-block:: yaml

    group_name: hyperparameter_sweep_1

.. code-block:: bash

    chemtorch +experiment=graph group_name=my_group


**run_name**

W&B run name for identifying individual runs.

* **Type**: String or null
* **Default**: ``null`` (W&B generates a random name)

.. code-block:: yaml

    run_name: baseline_experiment

.. code-block:: bash

    chemtorch +experiment=graph run_name="baseline run"


Model Loading
-------------

**load_model**

Load a pre-trained model from a checkpoint.

* **Type**: Boolean
* **Default**: ``false``

.. code-block:: yaml

    load_model: true
    ckpt_path: path/to/checkpoint.ckpt


**ckpt_path**

Path to the checkpoint file to load.

* **Type**: String or null
* **Default**: ``null``
* **Required if**: ``load_model=true``

.. code-block:: yaml

    ckpt_path: lightning_logs/checkpoints/epoch=99.ckpt

.. code-block:: bash

    chemtorch +experiment=graph load_model=true ckpt_path=path/to/checkpoint.ckpt

See :ref:`inference` for more details on loading models.


Prediction Saving
-----------------

**predictions_save_path**

Path to save predictions (for single partition experiments).

* **Type**: String or null
* **Default**: ``null``

.. code-block:: yaml

    predictions_save_path: predictions/test_predictions.csv

Use when running a single task (``test``, ``validate``, or ``predict``).


**predictions_save_dir**

Directory to save predictions (for multi-partition experiments).

* **Type**: String or null
* **Default**: ``null``

.. code-block:: yaml

    predictions_save_dir: predictions/

Use when running multiple tasks (e.g., ``tasks: [fit, test]``).


**save_predictions_for**

Specify which dataset partitions to save predictions for.

* **Type**: String, list of strings, or null
* **Options**: ``"train"``, ``"val"``, ``"test"``, ``"predict"``, ``"all"``
* **Default**: ``null``

.. code-block:: yaml

    # Save for single partition
    save_predictions_for: test
    
    # Save for multiple partitions
    save_predictions_for:
      - train
      - val
      - test
    
    # Save for all partitions
    save_predictions_for: all

.. code-block:: bash

    chemtorch +experiment=graph save_predictions_for=test predictions_save_dir=predictions/


Data Subsampling
----------------

**data_module.subsample**

Subsample the dataset for quick testing (not a ChemTorch root-level key, but commonly used).

* **Type**: Float (0.0 to 1.0), int (0 to dataset size) or null
* **Default**: ``null`` (use full dataset)

.. code-block:: bash

    # Use 5% of data for quick testing
    chemtorch +experiment=graph data_module.subsample=0.05

    # Use 100 samples
    chemtorch +experiment=graph data_module.subsample=100


Useful for debugging or testing configurations quickly.


Advanced Options
----------------

**parameter_limit**

Limit the number of model parameters (for testing or architecture search).
If the parameter limit is exceeded ChemTorch will skip the run.

* **Type**: Integer or null
* **Default**: ``null`` (no limit)

.. code-block:: yaml

    parameter_limit: 1000000  # Max 1M parameters


Common CLI Patterns
===================

Quick Testing
-------------

.. code-block:: bash

    # Fast test run: subsample data, run a single batch from every dataloader, no logging
    chemtorch +experiment=graph \
        data_module.subsample=0.01 \
        +trainer.fast_dev_run=true \
        log=false


Inference
---------

.. code-block:: bash

    # Load model and run predictions
    chemtorch +experiment=graph \
        load_model=true \
        ckpt_path=path/to/checkpoint.ckpt \
        tasks=[predict] \
        predictions_save_path=predictions.csv

For advanced Hydra features, see :ref:`hydra` or the `official Hydra documentation <https://hydra.cc/docs/intro/>`__.

