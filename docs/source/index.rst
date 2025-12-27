.. ChemTorch documentation master file, created by
   sphinx-quickstart on Mon Aug 18 15:33:39 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ChemTorch!
=====================

ChemTorch is a modular open-source research framework for deep learning of chemical reactions.

- 🚀 **Streamline your research workflow**: seamlessly assemble modular deep learning pipelines, track experiments, conduct hyperparameter sweeps, and run benchmarks.
- 💡 **Multiple reaction representations** with baseline implementations including SMILES tokenizations, molecular graphs, 3D geometries, and fingerprint descriptors.
- ⚙️ **Preconfigured data pipelines** for common benchmark datasets including RDB7, cycloadditions, USPTO-1k, and more.
- 🔬 **OOD evaluation** via chemically informed data splitters (size, target, scaffold, reaction core, ...).
- 🗂️ **Extensible component library** (growing) for all parts of the ChemTorch pipeline.
- 🔄 **Reproducibility by design** with Weights & Biases experiment tracking and a guide for setting up reproducibility tests.

Ready? Start with the :ref:`quick-start`.
For a few examples of what you can already do with ChemTorch read the `white paper <https://chemrxiv.org/engage/chemrxiv/article-details/690357d9a482cba122e366b6>`__ on ChemRxiv.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/quick_start
   getting_started/logging
   getting_started/overview
   getting_started/config
   getting_started/experiments

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/cli_usage
   user_guide/custom_data
   user_guide/custom_components
   user_guide/sweeps
   user_guide/reproducing_experiments
   user_guide/reproducibility_tests
   user_guide/config_ref

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

   developer_guide/custom_components
   developer_guide/props
   developer_guide/testing
   developer_guide/hydra
   developer_guide/contributing

.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:

   api_autodoc/index