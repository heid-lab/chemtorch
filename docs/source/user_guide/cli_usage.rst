.. _cli-usage:

=========
CLI Usage
=========

This page provides a comprehensive reference for using the ChemTorch command-line interface (CLI)

Command-Line Interface
======================
This section covers basic usage patterns and syntax for the ChemTorch CLI.

The ChemTorch CLI is built on top of Hydra, so it uses the same override syntax and supports all Hydra CLI features.
We highlight the most important patterns and special Hydra quirks here.
For more details on Hydra's override syntax (how CLI overrides are parsed), see the official Hydra documentation:

- Basic override grammar: https://hydra.cc/docs/advanced/override_grammar/basic/
- Extended override grammar: https://hydra.cc/docs/advanced/override_grammar/extended/

Basic Usage
-----------

The ChemTorch CLI follows this general pattern:

.. code-block:: chemtorch

    chemtorch +experiment=NAME [key=value] [key.subkey=value] ...

**Examples:**

.. code-block:: chemtorch

    # Run an experiment
    chemtorch +experiment=graph
    
    # Override values
    chemtorch +experiment=graph trainer.max_epochs=200 seed=42
    
    # Change a component
    chemtorch +experiment=graph model=gcn


Override Syntax
---------------

**Nested keys** - Use dot notation:

.. code-block:: chemtorch

    chemtorch +experiment=graph model.hidden_channels=256

**Component selection** - Swap entire configs:

.. code-block:: chemtorch

    chemtorch +experiment=graph model=gcn

**Lists** - Use square brackets with single/double quotes:

.. code-block:: chemtorch

    chemtorch +experiment=graph tasks='[fit,test]'

There must be no white space before and after the comma!
Otherwise Hydra will throw an error.

**Booleans**:

.. code-block:: chemtorch

    chemtorch +experiment=graph log=true

**Null values** - Use ``~`` prefix to remove a key from the config:

.. code-block:: chemtorch

    chemtorch +experiment=graph ~group_name

You cannot use ``key=null`` because Hydra intreprets ``null`` as the string ``"null"`` when passed via the CLI.
Importantly, ``~`` removes the key but this only works if the key is present in the config in the first place.
Otherwise, Hydra will throw an error.


**Strings with spaces** - Use quotes:

.. code-block:: chemtorch

    chemtorch +experiment=graph run_name="my experiment"

**Multiple runs** - Use multirun mode (``-m`` or ``--multirun``) with comma-separated values:

.. code-block:: chemtorch

    chemtorch -m +experiment=graph seed=0,1,2

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

For details on ChemTorch specific CLI options and configuration keys, see :ref:`config-ref`.
For advanced Hydra features, see :ref:`hydra` or the `official Hydra documentation <https://hydra.cc/docs/intro/>`__.

