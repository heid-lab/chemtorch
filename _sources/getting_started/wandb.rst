.. _wandb:

=========================
Intro to Weights & Biases
=========================

ChemTorch uses Weights & Biases (W&B) for logging, visualization, and run management.
This guide will show you how to quickly setup W&B for ChemTorch.

Connect W&B
===========
If you are new to W&B, you will need to create an account and authorize your machine:

1. First, create a free account at https://wandb.ai/signup (if you don't have one yet).

2. Then, go to https://wandb.ai/authorize, copy your API key, and set the :code:`WANDB_API_KEY` environment variable (if you haven't done so already):

   .. code-block:: wandb

      export WANDB_API_KEY="your_api_key_here"

3. Login to W&B from the command line:
   
   .. code-block:: wandb

      wandb login

Whenever you use another machine or a new environment (e.g. GPU cluster), you will need to repeat step 2 and 3.

Log Your ChemTorch Runs
=======================
Once you are connected to W&B, you can log your ChemTorch runs by appending :code:`log=true` when launching an experiment.
For example, using the quick start command from the :ref:`quick-start` guide:

.. code-block:: chemtorch

   chemtorch +experiment=graph dataset.subsample=0.05 log=true

Now, you run will be logged to your W&B account.
Open the `W&B web interface <https://wandb.ai/signup>`__ and navigate to `Projects → chemtorch` to see your run in the project workspace.
By default, the run will be logged to the project :code:`chemtorch` under your username.
You can change the project name via the :code:`project_name` option:

.. code-block:: chemtorch

   chemtorch +experiment=graph dataset.subsample=0.05 log=true project_name=my_project

To avoid having to specify :code:`project_name` every time, you can override the default project name
in :code:`conf/base.yaml` to your desired project name:

.. code-block:: yaml

   # conf/base.yaml
   project_name: my_project

By default, the runs will have arbitrary names.
You can name your runs by setting :code:`run_name`:

.. code-block:: chemtorch

   chemtorch +experiment=graph dataset.subsample=0.05 log=true run_name=my_first_run

You can also group related runs together by setting :code:`group_name`:

.. code-block:: chemtorch

   chemtorch +experiment=graph dataset.subsample=0.05 log=true group_name=my_experiment

