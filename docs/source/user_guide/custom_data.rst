.. _custom-dataset:

===================
Using Your Own Data
===================

This guide shows how to use your own data with ChemTorch by configuring the **data pipeline**.
The data pipeline is composed of three main components that you can mix and match:

1. **Data Source**: loads your raw data (from CSV files, databases, etc.)
2. **Column Mapper**: filters and renames columns to match ChemTorch's expected format
3. **Data Splitter**: splits your data into train/val/test sets (optional for pre-split data)

Data Pipeline Overview
======================

The ``SimpleDataPipeline`` orchestrates these three components in sequence:

.. code-block:: python

    raw_data = data_source.load()           # 1. Load raw data
    processed = column_mapper(raw_data)     # 2. Map columns
    split_data = data_splitter(processed)   # 3. Split data (if needed)

You configure the pipeline in your Hydra config under ``data_module.data_pipeline``.

Configuring the Data Pipeline
=============================

Configuring the Data Source
---------------------------
The data source loads your raw data. ChemTorch provides two built-in options:

SingleCSVSource
^^^^^^^^^^^^^^^

For loading data from a single CSV file that will be split later:

.. code-block:: yaml

    data_module:
      data_pipeline:
        data_source:
          _target_: chemtorch.components.data_pipeline.data_source.SingleCSVSource
          data_path: /path/to/your/data.csv

PreSplitCSVSource
^^^^^^^^^^^^^^^^^

For data already split into separate train/val/test CSV files:

.. code-block:: yaml

    data_module:
      data_pipeline:
        data_source:
          _target_: chemtorch.components.data_pipeline.data_source.PreSplitCSVSource
          train_path: /path/to/train.csv
          val_path: /path/to/val.csv
          test_path: /path/to/test.csv

.. note::
   When using ``PreSplitCSVSource``, you must use ``IndexSplitter`` as the data splitter to load a ``.pkl`` file which partitions the data points by their row indices.

Configuring the Column Mapper
------------------------------

The column mapper transforms your CSV columns to match ChemTorch's expected format.
At minimum, you need a column containing SMILES strings.

ColumnFilterAndRename
^^^^^^^^^^^^^^^^^^^^^

The standard column mapper filters and renames columns:

.. code-block:: yaml

    data_module:
      data_pipeline:
        column_mapper:
          _target_: chemtorch.components.data_pipeline.column_mapper.ColumnFilterAndRename
          smiles: "reaction_smiles"    # your column name -> ChemTorch's expected name
          label: "barrier_kcal_mol"    # optional: only needed for training

The left side (``smiles``, ``label``) is what ChemTorch expects. The right side
(``"reaction_smiles"``, ``"barrier_kcal_mol"``) is your actual column name in the CSV.

Configuring the Data Splitter
------------------------------

The data splitter divides your data into train/validation/test sets.
ChemTorch provides several splitters for different use cases:

.. list-table:: Available data splitters
   :header-rows: 1

   * - Splitter
     - Description
     - Use case
   * - ``RatioSplitter``
     - Randomly shuffles the data and splits it according to train/val/test ratios (default behavior).
     - Standard train/val/test split.
   * - ``ScaffoldSplitter``
     - Splits by molecular scaffold (Bemis–Murcko) to test generalization to novel scaffolds.
     - Evaluating scaffold generalization; can split on reactants, products, or both.
   * - ``TargetSplitter``
     - Splits by target value range (e.g., low/mid/high barriers).
     - Evaluating extrapolation in property space.
   * - ``SizeSplitter``
     - Splits by molecular size (ascending or descending).
     - Evaluating extrapolation to larger/smaller molecules.
   * - ``ReactionCoreSplitter``
     - Splits by reaction core identity (reaction SMARTS).
     - Evaluating generalization to novel reaction types.
   * - ``IndexSplitter``
     - Splits by explicit row indices (requires a ``.pkl`` file mapping split names to indices).
     - Pre-computed or custom splits.

For a complete list of available options and parameters, see ``conf/data_module/data_pipeline/data_splitter/``.

Complete Example
----------------

Here's a complete data pipeline config for training on custom data:

.. code-block:: yaml

    data_module:
      data_pipeline:
        _target_: chemtorch.components.data_pipeline.SimpleDataPipeline
        
        data_source:
          _target_: chemtorch.components.data_pipeline.data_source.SingleCSVSource
          data_path: /path/to/my_reactions.csv
        
        column_mapper:
          _target_: chemtorch.components.data_pipeline.column_mapper.ColumnFilterAndRename
          smiles: "rxn_smiles"
          label: "activation_energy"
        
        data_splitter:
          _target_: chemtorch.components.data_pipeline.data_splitter.RatioSplitter
          train_ratio: 0.8
          val_ratio: 0.1
          test_ratio: 0.1

Defining Config Defaults
========================

Instead of specifying the full data pipeline inline every time, create reusable config files
for each dataset under ``conf/data_module/data_pipeline/``. This lets you swap data pipelines
from the CLI without touching your experiment config.

For example, create ``conf/data_module/data_pipeline/my_dataset.yaml``:

.. code-block:: yaml

    _target_: chemtorch.components.data_pipeline.SimpleDataPipeline
    
    data_source:
      _target_: chemtorch.components.data_pipeline.data_source.SingleCSVSource
      data_path: /path/to/my_dataset.csv
    
    column_mapper:
      _target_: chemtorch.components.data_pipeline.column_mapper.ColumnFilterAndRename
      smiles: "rxn_smiles"
      label: "activation_energy"
    
    data_splitter:
      _target_: chemtorch.components.data_pipeline.data_splitter.RatioSplitter
      train_ratio: 0.8
      val_ratio: 0.1
      test_ratio: 0.1

Then, in your experiment config, simply reference it via defaults:

.. code-block:: yaml

    defaults:
      - /data_module/data_pipeline: my_dataset

Or swap data pipelines from the CLI:

.. code-block:: chemtorch

    chemtorch experiment=my_experiment data_module.data_pipeline=my_dataset
    chemtorch experiment=my_experiment data_module.data_pipeline=other_dataset

See ``conf/data_module/data_pipeline/`` for examples like ``rdb7_fwd.yaml``, ``rgd1.yaml``, etc.

Customizing the Data Pipeline
=============================

If the built-in components don't fit your needs, you can implement custom ones.
See :ref:`custom-components` for details on extending ChemTorch.

.. _inference:

Running Inference
=================

Running inference with a pretrained ChemTorch model uses the same data pipeline configuration
as training, with a few key differences.

Key config differences vs. training
-----------------------------------

Set or override the following when running inference (see :ref:`config-ref` for detailed field descriptions):

.. list-table:: Inference-specific config tweaks
   :header-rows: 1

   * - Setting
     - Description
   * - ``tasks: [predict]``
     - Required.
   * - | ``load_model: true``
       | ``ckpt_path: /path/to/model.ckpt``
     - Required to load trained weights.
   * - ``predictions_save_path`` **or** ``predictions_save_dir``
     - Exactly one must be set to write predictions.
   * - ``save_predictions_for: [predict]``
     - Optional; can also be left empty (``[]``) since there is only one data partition.

What ``main.py`` does automatically
-----------------------------------

When ``predict`` is in ``tasks``, ``main.py`` will:

- Clear ``column_mapper.label`` and ``data_pipeline.data_splitter``.
- Clear ``routine.standardizer`` so it is loaded from the checkpoint.
- Enforce that either ``predictions_save_path`` or ``predictions_save_dir`` is set.

Even though these are handled automatically, we recommend setting them explicitly
in the config to keep things clear and reproducible.

Minimal inference config
------------------------

Below is a minimal patch you can apply on top of your training config:

.. code-block:: yaml

    # ...existing code...
    tasks: [predict]

    load_model: true
    ckpt_path: /path/to/model.ckpt

    predictions_save_path: /path/to/preds.csv  # or use predictions_save_dir
    save_predictions_for: [predict]
    # ...existing code...

You'll also need to configure the data pipeline to point to your unlabeled data
(see the sections above for data source, column mapper, and data splitter configuration).

For a complete example, see ``conf/experiment/opi_tutorial/inference.yaml``
and ``conf/experiment/opi_tutorial/training.yaml`` for the corresponding training config.