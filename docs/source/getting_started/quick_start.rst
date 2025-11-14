.. _quick-start:

===============
Quick Start
===============

This page will help you install ChemTorch and run your first experiment.

Installation
============

First clone this repo and navigate to it:

.. code-block:: install

    git clone https://github.com/heid-lab/chemtorch.git
    cd chemtorch   

Next, you can install ChemTorch either via :code:`uv` (recommended) or :code:`conda`.
We recommend using :code:`uv` for a seamless, fast, and lightweight installation.

Via :code:`uv` (recommended)
----------------------------

.. code-block:: install

    uv sync
    uv scripts/install_pyg_deps.py

To also install development and documentation dependencies add the `--groups` option followed by `dev` or `docs`.
Alternatively, you can also use `--all-groups` to install both.

Via :code:`conda`
------------------------------------------------------

If you prefer using :code:`conda` to manage the base interpreter, create a minimal environment and then let :code:`uv` install all Python dependencies from :code:`pyproject.toml` and :code:`uv.lock`:

.. code-block:: install

    conda env create -f env/environment.yml
    conda activate chemtorch
    # now run uv installation commands from above

This keeps a single source of truth for Python packages (uv) while still letting you use conda to manage the environment itself.


Import Data
===========
Now that you have installed ChemTorch, you need to import some data to the :code:`data/` folder to run your first experiment.
For this quick start guide, we will use our reaction database which you can download with: 

.. code-block:: install

   git clone https://github.com/heid-lab/reaction_database.git data

.. If you want to use your own dataset checkout the :ref:`custom-dataset` tutorial.


Launch Your First Experiment
=====================
Finally, you are ready to run your first experiment with ChemTorch.
You can do so conveniently via the command line interface (CLI).
To launch a simple experiment, run the following command from the root directory of the repository:

.. code-block:: chemtorch

   chemtorch +experiment=graph data_module.subsample=0.05 log=false

The experiment will use the default graph learning pipeline to train and evaluate a directed message passing neural network (D-MPNN) on a random subset of the RDB7 dataset.
The output will display the run details, training progress, and final evaluation metrics.
Congratulations, you have successfully run your first experiment with ChemTorch!🎉


.. Next Steps
.. ==========
.. * :ref:`wandb` shows you how to setup Weights & Biases (W&B) for logging, visualizing, and managing your runs.
.. * :ref:`cli-usage` shows you how to use the command line interface (CLI) to streamline your workflow.
.. * :ref:`custom-dataset`: learn how to use your own dataset with ChemTorch.


