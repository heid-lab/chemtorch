.. _experiments:

===========
Create Experiments
===========

In the :ref:`quick-start` you saw that you can launch an experiment from the ChemTorch CLI by passing the ``+experiment`` option.
In this tutorial you will learn how to create your own experiment config step by step.

What is an Experiment Config?
==============================

An experiment config is a YAML file in ``conf/experiment/`` that brings together all the pieces of your ML pipeline:

* Which dataset to use
* How to represent the data (fingerprints, graphs, tokens, etc.)
* Which model architecture to use
* Training settings (optimizer, learning rate, epochs, etc.)
* Reproducibility settings (seed, logging, etc.)

By creating experiment configs, you can:

* Keep your experiments organized and documented
* Easily reproduce results
* Share configurations with collaborators
* Quickly try different combinations of components


Creating Your First Experiment Config
======================================

Let's create a simple experiment that trains a multi-layer perceptron (MLP) on molecular fingerprints.
This is a straightforward setup without complex dependencies, perfect for learning the basics.
For reference, the final config is provided at ``conf/experiment/example/fingerprint.yaml``.


Step 1: Create the Config File
-------------------------------

Create a new file ``conf/experiment/my_fingerprint_experiment.yaml``:

.. code-block:: text

    conf/
    ├── ...
    ├── experiment/
    │   ├── ...
    │   └── my_fingerprint_experiment.yaml
    └── base.yaml


Step 2: Add the Package Directive
----------------------------------

Start your experiment config with this special comment:

.. code-block:: yaml

    # @package _global_

This tells Hydra to treat this config as part of the global configuration namespace.
You'll need this line at the top of every experiment config.


Step 3: Select Components
--------------------------

Use the ``defaults`` list to select components from the config tree.
Each line picks a specific configuration from a config group:

.. code-block:: yaml

    # @package _global_
    defaults:
      - /data_module/data_pipeline: uspto_1k
      - /data_module/representation: drfp
      - /data_module/dataloader_factory: torch_dataloader
      - /model: mlp
      - /routine: classification
      - _self_

Let's break this down:

* ``/data_module/data_pipeline: uspto_1k`` - Use the USPTO-1k dataset
* ``/data_module/representation: drfp`` - Represent reactions as difference fingerprints
* ``/data_module/dataloader_factory: torch_dataloader`` - Use PyTorch's standard DataLoader
* ``/model: mlp`` - Use a multi-layer perceptron
* ``/routine: classification`` - This is a classification task
* ``_self_`` - If inlcuded as the last element of the defaults list this ensure that the overrides in this file are applied after the defaults (always include this)

.. note::

    The ``/`` prefix tells Hydra to look in the root ``conf/`` directory.
    See :ref:`pipeline_overview` for all available components.

Step 4: Add Logging Configuration
---------------------------------
Add logging configuration to enable logging to Weights & Biases.

.. code-block:: yaml

    # @package _global_
    defaults:
      - /data_module/data_pipeline: uspto_1k
      - /data_module/representation: drfp
      - /data_module/dataloader_factory: torch_dataloader
      - /model: mlp
      - /routine: classification
      - _self_

    log: true
    project_name: chemtorch
    group_name: fingerprint
    run_name: null

See :ref:`logging` for how to setup Weights & Biases and basic logging options.

Step 5: Add ChemTorch Settings
-------------------------------

Add ChemTorch-specific configuration to control execution:

.. code-block:: yaml

    # @package _global_
    defaults:
      - /data_module/data_pipeline: uspto_1k
      - /data_module/representation: drfp
      - /data_module/dataloader_factory: torch_dataloader
      - /model: mlp
      - /routine: classification
      - _self_

    log: false
    project_name: chemtorch
    group_name: fingerprint
    run_name: null

    # ChemTorch configuration
    seed: 0

    tasks:
      - fit
      - test

See :ref:`cli-usage` for a full list of ChemTorch-specific configuration options.

Step 5: Add Logging Configuration
---------------------------------

.. code-block:: yaml

    # @package _global_
    defaults:
      - /data_module/data_pipeline: uspto_1k
      - /data_module/representation: drfp
      - /data_module/dataloader_factory: torch_dataloader
      - /model: mlp
      - /routine: classification
      - _self_

    # Logging
    log: false
    project_name: chemtorch
    group_name: fingerprint
    run_name: null

    # ChemTorch configuration
    seed: 0

    tasks:
      - fit
      - test


Step 6: Override Component Parameters
--------------------------------------

Finally, override specific values to customize the components:

.. code-block:: yaml

    # @package _global_
    defaults:
      - /data_module/data_pipeline: uspto_1k
      - /data_module/representation: drfp
      - /data_module/dataloader_factory: torch_dataloader
      - /model: mlp
      - /routine: classification
      - _self_

    # Logging
    log: false
    project_name: chemtorch
    group_name: fingerprint
    run_name: null

    # ChemTorch configuration
    seed: 0

    tasks:
      - fit
      - test

    # Training settings
    trainer:
      max_epochs: 100

    # Task-specific settings
    num_classes: 1000
    fp_length: 2048

    # Override representation config
    data_module:
      representation:
        n_folded_length: ${fp_length}

    # Override model config
    model:
      in_channels: ${fp_length}
      hidden_size: 256
      out_channels: ${num_classes}
      num_hidden_layers: 2
      dropout: 0.02
      act: relu
      norm: null

The keys you can override correspond to the ``__init__()`` parameters of each component's Python class.
Notice the use of ``${fp_length}`` and ``${num_classes}`` - this is Hydra's variable interpolation syntax that lets you reference values defined elsewhere in the config.


Step 7: Run Your Experiment
----------------------------

Now run your experiment from the command line:

.. code-block:: bash

    chemtorch +experiment=my_fingerprint_experiment

That's it! ChemTorch will:

1. Load and preprocess the USPTO-1k dataset
2. Represent reactions as difference fingerprints
3. Train an MLP for 100 epochs
4. Evaluate on the test set

You've successfully created and run your first experiment config! 🎉
