.. _reproducability:
=======================
Reproducing Experiments
=======================

In this tutorial you will learn how to seamlessly reproduce experiments run with ChemTorch.
This tutorial assumes that you have already :ref:`setup logging with W&B <logging>`.

#.  Run your experiments with ``log=true``.
    
    .. code-block:: chemtorch

        chemtorch +experiment=my_experiment log=true
    
    When logging is active, ChemTorch will also log the fully resolved config to W&B and it will be available in the ``wandb/`` folder.

#.  The W&B config format is slightly different from that used by Hydra.
    So you first need to parse the config to Hydra format using the ``wandb_to_hydra.py`` script located in the ``scripts/`` folder of ChemTorch.

    .. program-output:: python ../../scripts/wandb_to_hydra.py -h
        :caption: ``wandb_to_hydra.py`` usage


#.  If we save the config to ``conf/saved_configs/my_experiment/my_run.yaml``, we can run the following command to reproduce the experiment:

    .. code-block:: chemtorch

        chemtorch -cd=conf/saved_configs/my_experiment -cn=my_run

    This will run the exact same experiment as before, using the same config parameters.

Reproducing The ChemTorch Benchmarks
====================================
The `ChemTorch white paper <https://chemrxiv.org/engage/chemrxiv/article-details/687556a4fc5f0acb52358b65>`__ includes a set of benchmark experiments that can be easily reproduced using the steps outlined above.
The configs for these experiments are available in the ``conf/saved_configs/chemtorch_benchmarks/`` folder:

.. code-block:: text

    conf/saved_configs/chemtorch_benchmarks/
    ├── optimal_model_configs
    │   ├── atom_han.yaml
    │   ├── cgr_dmpnn.yaml
    │   ├── dimreaction.yaml
    │   ├── drfp_mlp.yaml
    │   └── ...
    ├── dmpnn_data_split_benchmark
    │   ├── random_split.yaml
    │   ├── reactant_scaffold_split.yaml
    │   ├── reaction_core_split.yaml
    │   └── ...
    └── ...

The optimal model configs contain the best hyper-parameters found for each model/reaction representation on the benchmark datasets.
The data split benchmark configs use the same hyper-parameters as the optimal model config, but vary the data splitting strategy to demonstrate the effect of data splits on model performance.

For example, run the following command to reproduce the CGR/D-MPNN experiment with seeds 0 to 9:

.. code-block:: chemtorch

    chemtorch -m -cd=conf/saved_configs/chemtorch_benchmarks/optimal_model_configs -cn=cgr_dmpnn +experiment=chemtorch_benchmarks seed=0,1,2,3,4,5,6,7,8,9

The multirun (``-m``) flag will run the experiment with all the specified seeds in sequence (see :ref:`cli_usage`).