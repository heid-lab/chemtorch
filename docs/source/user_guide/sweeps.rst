.. _sweeps:

=====================
Hyperparameter Sweeps
=====================
This tutorial shows you how to run hyperparameter sweeps with ChemTorch and Weights & Biases (W&B).
A sweep allows you to systematically explore different hyperparameter configurations to find the best performing model.
Please refer to the `official W&B sweeps documentation <https://docs.wandb.ai/guides/sweeps>`__ for more details on sweeps.

Overview
========
W&B sweeps integrate seamlessly with ChemTorch's Hydra-based configuration system. The main considerations for ChemTorch sweeps are:

1. **Command Configuration**: Use ChemTorch's CLI with proper argument formatting
2. **Parameter Overrides**: Leverage Hydra's dot notation for nested configurations
3. **Argument Formatting**: Use ``args_no_hyphens`` for proper Hydra compatibility

Setup a Sweep
=============
To setup a sweep, you need to define a sweep configuration in a YAML file. 
Here's a simple example for tuning a D-MPNN model:

.. code-block:: yaml
   :class: sweep-chemtorch-command

    program: chemtorch
    command:
      - ${env}
      - ${program}
      - "+experiment=graph"
      - ${args_no_hyphens}

    project: chemtorch
    name: dmpnn_sweep
    method: bayes
    metric:
      goal: minimize
      name: val_loss_epoch

    early_terminate:
        type: hyperband
        min_iter: 30
        eta: 1.5

    parameters:
      routine.optimizer.lr:
        min: 0.0001
        max: 0.01
        distribution: log_uniform_values
      
      dataloader.batch_size:
        values: [32, 64, 128]
        distribution: categorical

      model.hidden_channels:
        values: [64, 128, 256]
        distribution: categorical

      model.layer_stack.depth:
        values: [3, 6, 9]
        distribution: categorical

You might also want to tune additional hyperparameters such as dropout rate, weight decay, or specific model architecture parameters depending on your use case.

ChemTorch-Specific Configuration
================================

Command Setup
-------------
In the ``command`` we specify the same command that one would use to run 
ChemTorch from the CLI:

.. code-block:: yaml
   :class: sweep-chemtorch-command

    command:
      - ${env}
      - ${program}
      - "+experiment=graph"
      - ${args_no_hyphens}

Key points:

- ``${env}``: Uses the current environment
- ``${program}``: References the ``program`` field (``chemtorch``)
- ``"+experiment=graph"``: Loads a base experiment configuration for graph-based models
- ``${args_no_hyphens}``: formats sweep parameters without hyphens for Hydra compatibility

Optimization Goal
-----------------
In this config we tell the agent to minimize the epoch-level validation
loss:

.. code-block:: yaml

    metric:
      goal: minimize
      name: val_loss_epoch

This requires that `val_loss_epoch` is logged to W&B during training.
ChemTorch does this by default, but if you want to optimize hyperparameters
with respect to another metric you must ensure that it is logged under the
same name as specified in the sweep config.


Parameter Naming
----------------
Use Hydra's dot notation to override nested configuration values:

.. code-block:: yaml

    parameters:
      routine.optimizer.lr:        # Overrides config.routine.optimizer.lr
        min: 0.0001
        max: 0.01
      model.hidden_channels:       # Overrides config.model.hidden_channels
        values: [64, 128, 256]
      model.layer_stack.depth:     # Overrides config.model.layer_stack.depth
        values: [3, 6, 9]
      dataloader.batch_size:       # Overrides config.dataloader.batch_size
        values: [32, 64, 128]

This corresponds to the hierarchical structure in your Hydra configs:

.. code-block:: yaml

    # In your config files
    routine:
      optimizer:
        lr: 0.001
    model:
      hidden_channels: 128
      layer_stack:
        depth: 6
    dataloader:
      batch_size: 64

Running a Sweep
===============

1. **Initialize the sweep**:

   .. code-block:: wandb

       wandb sweep path/to/your_sweep.yaml

   This returns a sweep ID that you'll use to start agents.

2. **Start sweep agents**:

   .. code-block:: wandb

       wandb agent <your-entity>/<your-project>/<sweep-id>

   You can run multiple agents in parallel on different machines or GPUs.

3. **Monitor progress**:

   View your sweep results in the W&B dashboard at ``https://wandb.ai/<your-entity>/<your-project>/sweeps/<sweep-id>``

Best Practices
==============

- **Start small**: Begin with a few parameters and expand gradually
- **Use appropriate distributions**: 
  - ``log_uniform_values`` for learning rates and regularization
  - ``categorical`` for discrete choices
  - ``uniform`` for continuous ranges
- **Set early termination**: Use Hyperband or other early stopping to save compute
- **Cap the number of runs**: Limit total runs by setting the `--count` flag in the `wandb agent` command
- **Use base experiments**: Leverage existing experiment configs with ``+experiment=<name>`` to reduce configuration complexity