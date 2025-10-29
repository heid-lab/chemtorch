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

.. TODO: explain what ChemTorch actually does (see paper)

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/quick_start
   getting_started/logging
   getting_started/config
   getting_started/experiments

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/cli_usage
   user_guide/config_ref
   user_guide/reproducibility
   user_guide/pipeline_overview
   user_guide/custom_data
   user_guide/custom_components
   user_guide/sweeps

.. toctree::
   :maxdepth: 1
   :caption: Advanced Guide
   :hidden:
   
   advanced_guide/hydra
   advanced_guide/props

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:
   
   examples/training_curves
   examples/ood_benchmarking

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide
   :hidden:

   developer_guide/contributing
   developer_guide/framework_structure
   developer_guide/testing