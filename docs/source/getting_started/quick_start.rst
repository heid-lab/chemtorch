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

We recommend using `uv <https://docs.astral.sh/uv/#installation>`__ for a quick and easy installation:

.. code-block:: install

    uv sync --extra <backend> && uv run scripts/install_pyg_deps.py --run

Replace ``<backend>`` with the accelerator you plan to use (e.g. ``cpu``, ``macos``, ``cu128``).

Please refer to the :ref:`detailed installation reference <install-details>` for installation using ``conda``, the full list of supported backends, and additional dependencies for developers. PyTorch Geometric-specific notes live in :ref:`pyg-installation`.

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


.. _install-details:

Detailed Installation Reference
================================

The quick command above installs everything most users need.
Use the reference below when you want to double-check backend options, tweak dependency groups, or follow an alternative setup path.

Backend matrix
--------------
Use the ``--extra`` flag to pick a PyTorch wheel matching your hardware.
If you skip the extra, the CPU build is installed by default.

- ``cpu`` — PyTorch 2.5.x through 2.8.x CPU wheels (Linux)
- ``macos`` — PyTorch 2.5.x–2.8.x universal2 wheels (CPU/MPS)
- ``cu118`` — PyTorch 2.5.x–2.7.x with CUDA 11.8
- ``cu121`` — PyTorch 2.5.x–2.7.x with CUDA 12.1
- ``cu124`` — PyTorch 2.5.x–2.7.x with CUDA 12.4
- ``cu126`` — PyTorch 2.6.x with CUDA 12.6
- ``cu128`` — PyTorch 2.7.x–2.8.x with CUDA 12.8
- ``cu129`` — PyTorch 2.8.x with CUDA 12.9
- ``cu130`` — PyTorch 2.9.x with CUDA 13.0 (PyG wheels pending; see :ref:`pyg-installation`)

Need AMD/ROCm support?
Reach out on GitHub and we will figure out a ROCm installation path together with you.

Alternative base interpreter via ``conda``
-----------------------------------------
Prefer managing the Python runtime with ``conda``?
Create a lightweight environment, activate it, and then let ``uv`` handle the project dependencies:

.. code-block:: install

    conda env create -f environment.yml
    conda activate chemtorch
    # run uv installation as described above

.. _dev-deps:
Developer dependency groups
---------------------------
Append optional groups to the ``uv sync`` command when you need tooling for documentation or testing:

.. code-block:: install

    uv sync --extra <backend> [--groups {test,docs}|--all-groups]

- ``--groups docs`` installs the documentation toolchain.
- ``--groups test`` installs everything required to run the test suite.
- ``--all-groups`` installs both sets at once.

.. _pyg-installation:

PyG Installation
----------------
ChemTorch relies on PyTorch Geometric (PyG) and its companion wheels for graph learning.
The installation script ``scripts/install_pyg_deps.py`` inspects the installed PyTorch distribution, prints the matching PyG install command, and executes it when you pass ``--run`` (or ``-y``).
Rerun it after every ``uv sync`` so your PyG wheels stay aligned.

Trouble Shooting
^^^^^^^^^^^^^^^^
No wheel PyG wheel available for your installation?
PyG releases often trail the newest PyTorch wheels.
If PyG does not offer a wheel for you PyTorch build yet you have three options:
1. Simply skip the PyG installation step and just run ``uv sync`` if you don't plan to run graph workloads.
2. Downgrade to a supported hardware driver if you can (e.g. CUDA 12.9 instead of 13.0) and rerun installation.
   Check the `PyG installation selector <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`__ lists the combinations that are officially supported and will be updated once newer wheels become available.
3. Need to stay on an unsupported combination? Build the PyG wheels from source instead: `PyG source install guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-source>`__.