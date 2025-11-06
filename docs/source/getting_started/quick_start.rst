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

Next, you can install ChemTorch either via :code:`conda` or :code:`uv`.
We recommend using :code:`uv` for a seamless, fast, and lightweight installation.

Via :code:`uv`
--------------

.. code-block:: install

    uv sync
    uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric  --no-build-isolation

To also install development and documentation dependencies add the `--groups` option followed by `dev` or `docs`.
Alternatively, you can also use `--all-groups` to install both.

Via :code:`conda`
-----------------

.. code-block:: install

    conda create -n chemtorch python=3.10 && \
    conda activate chemtorch && \
    pip install rdkit numpy==1.26.4 scikit-learn pandas && \
    pip install torch && \
    pip install hydra-core && \
    pip install torch_geometric && \
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html && \
    pip install wandb && \
    pip install ipykernel && \
    pip install -e .

If you have a GPU, follow the `official PyTorch installation instructions <https://pytorch.org/get-started/locally/>`__ to install 
the appropriate :code:`torch` version for your CUDA setup and then install the corresponding :code:`torch-scatter` and :code:`torch-sparse` 
versions:

.. code-block:: install

    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

where :code:`TORCH` is your installed torch version (e.g. :code:`2.5.0+cu121`) and :code:`CUDA` is your CUDA version (e.g. :code:`cu121`).


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


