.. _config-ref:

================
Config Reference
================

This page documents all ChemTorch-specific configuration options that can be set in experiment configs or via the CLI.

If you look for guidance on designing configuration files, see :ref:`hydra` which covers ChemTorch-specific extensions to the standard Hydra configuration system.

.. note::
    Defaults are set in ``conf/base.yaml``, however, not all supported options necessarily have a corresponding default value since specifyng ``key=null`` in the config is the same as not setting the key at all.
    Just know that you can still set all options listed below in your experiment config or add them in the CLI via ``+key``.


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

.. code-block:: chemtorch

    chemtorch +experiment=graph tasks='[fit,test]'

* ``fit``: Training and validation
* ``test``: Evaluate on test set(s)
* ``validate``: Evaluate on validation set only
* ``predict``: Run inference without training (see :ref:`inference`).

The ``predict`` task runs inference only (no training).
Before using it, you must already have a trained model checkpoint — first run ``fit`` to produce one, then set ``load_model: true`` together with ``ckpt_path`` (see :ref:`model-loading`).
For ``predict`` the data pipeline must not perform any splitting (set the data splitter to ``null``) and the column mapper must omit the label/target column if it exists. ChemTorch will enforce these automatically for a predict-only run, but declaring them explicitly can make configs clearer and easier to audit.

To persist outputs from a predict-only run, use the keys described under :ref:`prediction-saving`: ``predictions_save_path`` for a single partition (``tasks: [predict]``) or ``save_predictions_for`` plus ``predictions_save_dir`` when multiple partitions are available.
If instead you want to capture predictions for just one of the ``train``, ``val``, or ``test`` partitions during/after training (without a separate predict-only task), configure the same :ref:`prediction-saving` keys as explained in that section.

Note that to specify a single task you also need to use the list syntax.

.. code-block:: yaml

    tasks:
      - predict

.. code-block:: chemtorch

    chemtorch +experiment=graph tasks='[predict]'.


**seed**

Random seed for reproducibility.

* **Type**: Integer
* **Default**: ``0``

.. code-block:: yaml

    seed: 42

.. code-block:: chemtorch

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

.. code-block:: chemtorch

    chemtorch +experiment=graph log=true


**project_name**

W&B project name for organizing runs.

* **Type**: String
* **Default**: ``"chemtorch"``

.. code-block:: yaml

    project_name: my_project

.. code-block:: chemtorch

    chemtorch +experiment=graph project_name=my_project


**group_name**

W&B group name for organizing related runs (e.g., hyperparameter sweeps).

* **Type**: String or null
* **Default**: ``null``

.. code-block:: yaml

    group_name: hyperparameter_sweep_1

.. code-block:: chemtorch

    chemtorch +experiment=graph group_name=my_group


**run_name**

W&B run name for identifying individual runs.

* **Type**: String or null
* **Default**: ``null`` (W&B generates a random name)

.. code-block:: yaml

    run_name: baseline_experiment

.. code-block:: chemtorch

    chemtorch +experiment=graph run_name="baseline run"


.. _model-loading:

Model Loading
-------------

**load_model**

Load a pre-trained model from a checkpoint.

* **Type**: Boolean
* **Default**: ``false``
* If ``true``, ``ckpt_path`` must be specified.

**ckpt_path**

Path to the checkpoint file to load.

* **Type**: String or null
* **Default**: ``null``
* **Required if**: ``load_model=true``

.. code-block:: yaml

    load_model: true
    ckpt_path: path/to/checkpoint.ckpt

.. code-block:: chemtorch

    chemtorch +experiment=graph load_model=true ckpt_path=path/to/checkpoint.ckpt

.. _prediction-saving:

Prediction Saving
-----------------

**save_predictions_for**

Specify which dataset partitions to save predictions for.

* **Type**: String, list of strings, or null
* **Options**: ``"train"``, ``"val"``, ``"test"``, ``"predict"``, ``"all"``
* **Default**: ``null``

Only required if mutliple data partations are used. For example:

* ``tasks: [fit]``: training and validation partitions are available
* ``tasks: [fit, test]``: training, validation, and test partitions are available

Not required if only a single partition is used (e.g., ``tasks: [test]`` or ``tasks: [predict]``).

If set to ``all``, predictions will be saved for all available partitions.


**predictions_save_path**

Path to save predictions (for single data partition).

* **Type**: String or null
* **Default**: ``null``

.. code-block:: yaml

    tasks: predict
    predictions_save_path: my_experiment/predictions.csv

.. code-block:: chemtorch

    chemtorch +experiment=graph predictions_save_path=my_experiment/predictions.csv




**predictions_save_dir**

Directory to save predictions (for multiple data partitions).
The predictions will be saved in files named ``<partition>.csv`` within this directory, where ``<partition>`` is one of ``train``, ``val``, ``test``, or ``predict``.


* **Type**: String or null
* **Default**: ``null``
* **Requires** that ``save_predictions_for`` is set.

.. code-block:: yaml
    
    # Save for multiple partitions
    save_predictions_for:
      - train
      - test

.. code-block:: chemtorch

    chemtorch +experiment=graph save_predictions_for='[train,test]' predictions_save_dir=predictions/


Data Subsampling
----------------

**data_module.subsample**

Subsample the dataset for quick testing (not a ChemTorch root-level key, but very useful).

* **Type**: Float (0.0 to 1.0), int (0 to dataset size) or null
* **Default**: ``null`` (use full dataset)

.. code-block:: chemtorch

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

.. code-block:: chemtorch

    # Fast test run: subsample data, run a single batch from every dataloader, no logging
    chemtorch +experiment=graph \
        data_module.subsample=0.01 \
        +trainer.fast_dev_run=true \
        log=false


Inference
---------

.. code-block:: chemtorch

    # Load model and run predictions
    chemtorch +experiment=graph \
        load_model=true \
        ckpt_path=path/to/checkpoint.ckpt \
        tasks=[predict] \
        predictions_save_path=predictions.csv

For advanced Hydra features, see :ref:`hydra` or the `official Hydra documentation <https://hydra.cc/docs/intro/>`__.

