.. _config:

=====================
Understand Configs
=====================

In the :ref:`quick-start`, you launched an experiment using the command line.
Behind the scenes, ChemTorch used configuration files to set up your entire machine learning pipeline.
This page explains the core concepts of ChemTorch's configuration system.

What is Configuration?
======================

Configuration files define *what* your experiment does without changing the source code.
They specify which model to use, how to process your data, which optimizer to use, and many other settings.

ChemTorch uses **YAML files** for configuration. If you're new to YAML, here's a quick primer:

.. code-block:: yaml

    # This is a comment
    key: value
    number: 42
    boolean: true
    
    nested:
      subkey: nested_value
    
    list:
      - item1
      - item2
      - item3

YAML is human-readable and uses indentation (like Python) to show structure.


The Configuration Folder
=========================

All configuration files live in the ``conf/`` folder at the root of the ChemTorch repository:

.. code-block:: text

    conf/
    ├── base.yaml
    ├── experiment/
    ├── data_module/
    ├── model/
    ├── routine/
    ├── trainer/
    ├── props/
    └── saved_configs/

Here's what each part does:

* ``base.yaml``: Top-level defaults loaded automatically
* ``data_module/``, ``model/``, ``routine/``, ``trainer/``: Component building blocks
* ``experiment/``: Your experiment configurations (see :ref:`experiments`)
* ``saved_configs/``: Saved configs for reproducibility (see :ref:`reproducing-experiments`)
* ``props/``: Runtime-computed properties (see :ref:`props`)


Hierarchical Configuration with Hydra
======================================

ChemTorch uses `Hydra <https://hydra.cc/>`__ for configuration management.
Hydra's key feature is **hierarchical composition** - the ability to build complex configs by combining simpler ones.

The Config Tree
---------------

The component folders (``model/``, ``routine/``, etc.) form a **config tree**.
Each folder is a **config group** - a category of interchangeable components.

For example, ``conf/model/`` contains different model architectures:

.. code-block:: text

    model/
    ├── dmpnn.yaml
    ├── gcn.yaml
    ├── gat.yaml
    └── ...

Each file configures a specific model that can be selected for your experiment.


Anatomy of a Config File
-------------------------

Let's look at an example config, ``model/dmpnn.yaml``:

.. code-block:: yaml

    _target_: chemtorch.components.model.gnn.gnn.GNN
    
    defaults:
      - encoder: directed_edge_enc
      - layer_stack: dmpnn_stack
      - pool: global_pool
      - head: mlp
    
    hidden_channels: 128

Key elements:

* ``_target_``: The Python class to instantiate (appears in every config file)
* ``defaults``: Sub-components this config depends on
* ``hidden_channels``: A parameter passed to the class's ``__init__()`` method


Hierarchical Composition
-------------------------

Configs can include other configs via the ``defaults`` list, forming a hierarchy.

For example, ``routine/regression.yaml`` composes multiple sub-components:

.. code-block:: yaml

    _target_: chemtorch.core.routine.RegressionRoutine
    
    defaults:
      - loss: mse
      - optimizer: adamw
      - lr_scheduler: cosine_with_warmup

This means: "Use the regression routine with MSE loss, AdamW optimizer, and cosine warmup scheduler."

The config tree for ``routine/`` looks like:

.. code-block:: text

    routine/
    ├── regression.yaml
    ├── loss/
    │   ├── mse.yaml
    │   ├── cross_entropy.yaml
    │   └── ...
    ├── optimizer/
    │   ├── adamw.yaml
    │   └── ...
    └── lr_scheduler/
        ├── cosine_with_warmup.yaml
        └── ...

Each sub-component can be swapped independently, making configs highly modular and reusable.


Changing Configuration Values
==============================

There are three ways to customize your experiments, listed from most to least common:


1. Via Command Line (Quick Experiments)
----------------------------------------

The easiest way to experiment is using command-line overrides.
You can change any configuration value without editing files:

.. code-block:: chemtorch

    # Change the number of training epochs
    chemtorch +experiment=graph trainer.max_epochs=200
    
    # Use a different model
    chemtorch +experiment=graph model=gcn
    
    # Change multiple values
    chemtorch +experiment=graph model=gcn model.hidden_channels=256 seed=42

The syntax is simple: ``key.subkey=value``

This is perfect for quick experiments and hyperparameter tuning.
For more CLI patterns, see :ref:`cli-usage`.


2. Via Editing Component Configs (Not Recommended)
---------------------------------------------------

You *can* directly edit files in ``conf/model/``, ``conf/routine/``, etc.
For example, you could open ``conf/model/dmpnn.yaml`` and change ``hidden_channels: 128`` to ``hidden_channels: 256``.

However, this is **not recommended** because:

- It changes defaults for *all* experiments
- It makes tracking changes difficult
- It can break existing experiments

Only edit component configs if you want to permanently change default behavior across all experiments.


3. Via Experiment Configs (Recommended)
----------------------------------------

Instead of editing component configs, create a new experiment config in ``conf/experiment/``:

.. code-block:: yaml

    # @package _global_
    defaults:
      - /data_module/data_pipeline: rdb7_fwd
      - /data_module/representation: cgr
      - /model: dmpnn
      - /routine: regression
    
    # Override values
    model:
      hidden_channels: 256
    
    trainer:
      max_epochs: 200
    
    tasks:
      - fit
      - test

Then run: ``chemtorch +experiment=my_experiment``

This approach keeps your experiments organized and component configs reusable.
Learn more in :ref:`experiments`.


How Values Get Resolved
========================

When you run ChemTorch, configuration values are determined by priority (highest to lowest):

1. **Command-line overrides**: ``trainer.max_epochs=200``
2. **Experiment config**: Values in ``conf/experiment/your_experiment.yaml``
3. **Component defaults**: Values in component configs (e.g., ``conf/model/dmpnn.yaml``)
4. **Base config**: Values in ``conf/base.yaml``

This means command-line overrides always win, followed by your experiment config, and so on.
