.. ChemTorch documentation master file, created by
   sphinx-quickstart on Mon Aug 18 15:33:39 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ChemTorch!
=====================

ChemTorch is an open-source deep learning framework for chemical reaction modeling that streamlines core research workflows.
It is designed to facilitate rapid prototyping, experimentation, and benchmarking.

* 🔬 **No More Boilerplate**: Focus on research, not engineering. Pre-built data handling, model training, and evaluation pipelines.
* 🧩 **Reusable Component Library**: Out-of-the-box implementations of common  models, reaction representations, and data splitting strategies.
* 🏗️ **Modular Architecture**: Easy to extend with custom datasets, representations, models, and training routines.
* 🔁 **Reproducibility By Design**: Built-in configuration and run management for reproducible experiments.
* 📊 **Built-in Benchmarks**: Standard datasets and evaluation protocols.
* 🚀 **Seamless Workflow**: CLI interface for easy experimentation, hyper-parameter tuning, and one-stop reproducibility.

Ready to dive in? Follow the :ref:`quick-start` to install ChemTorch and run your first experiment!

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/quick_start
   getting_started/wandb
   getting_started/cli_usage
   getting_started/custom_dataset

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   :hidden:
   
   examples/sweeps


.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer_guide/hydra
   developer_guide/framework_structure