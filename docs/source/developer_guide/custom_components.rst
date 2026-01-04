.. _custom-components:

==========================
Defining Custom Components
==========================

This page shows how to extend ChemTorch with custom components, highlights recurring design patterns, and points you to real, well-documented implementations in the codebase and API docs to get you started.

Development Workflow
====================

Use this lightweight checklist before you start coding:

#. Check existing components in the API docs and source.
#. Implement by inheriting and reusing existing components wherever possible.
#. Test thoroughly: start with unit tests; add integration tests when helpful.

Component Interfaces
=====================

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Component
     - Required Method
     - Input → Output
     - See Also
   * - Data Source
     - ``load()``
     - → ``pd.DataFrame``
     - API docs
   * - Column Mapper
     - ``__call__()``
     - ``pd.DataFrame`` → ``pd.DataFrame``µ
     - API docs
   * - Data Splitter
     - ``__call__()``
     - ``pd.DataFrame`` → ``DataSplit``
     - API docs
   * - Representation
     - ``construct()``
     - ``str`` → ``T`` (e.g., ``pyg.Data``, ``torch.Tensor``)
     - API docs
   * - Transform
     - ``__call__()``
     - ``T`` → ``T``
     - API docs
   * - Model
     - ``forward()``
     - batch → predictions
     - API docs

Refer to the API docs for full signatures, parameters, and existing implementations.

Data Pipeline
=============

A data pipeline is anything that returns a ``pd.DataFrame`` or ``DataSplit`` when called.
There is no explicit abstract interface—the contract is implicit.
In practice, data pipelines typically load data, apply transformations, and split into train/val/test.
:class:`~chemtorch.components.data_pipeline.simple_data_pipeline.SimpleDataPipeline` provides a reference implementation that composes a data source, column mapper, and splitter.
You can use it as a starting point or build your own from scratch.
See :ref:`custom-dataset` for a comprehensive walkthrough of building a complete data pipeline.

You may want to implement your own data source component for your specific data format, or a custom data splitter tailored to your use case. The examples below illustrate both.

**Example: custom data source (SQLite database)**

.. code-block:: python

    import sqlite3
    import pandas as pd
    from chemtorch.components.data_pipeline.data_source import AbstractDataSource
    
    class SQLiteDataSource(AbstractDataSource):
        """Loads chemical data from a SQLite database."""
        
        def __init__(self, db_path: str, table_name: str, query: str = None):
            """
            Args:
                db_path: Path to SQLite database file
                table_name: Name of the table to query
                query: Optional custom SQL query (overrides table_name)
            """
            self.db_path = db_path
            self.table_name = table_name
            self.query = query or f"SELECT * FROM {table_name}"
        
        def load(self) -> pd.DataFrame:
            """Load data from SQLite database."""
            conn = sqlite3.connect(self.db_path)
            try:
                df = pd.read_sql_query(self.query, conn)
                return df
            finally:
                conn.close()

**Example: custom data splitter (time-based)**

The following example demonstrates inheriting from :class:`~chemtorch.components.data_pipeline.data_splitter.ratio_splitter.RatioSplitter` rather than implementing :class:`~chemtorch.components.data_pipeline.data_splitter.abstract_data_splitter.AbstractDataSplitter` directly.
This approach reuses ratio validation and other infrastructure:

.. code-block:: python

    import pandas as pd
    from chemtorch.components.data_pipeline.data_splitter import RatioSplitter
    from chemtorch.utils import DataSplit
    
    class TimeBasedSplitter(RatioSplitter):
        """
        Splits data chronologically based on a timestamp column.
        
        Useful for time-series data where you want to predict future outcomes.
        Inherits ratio validation from RatioSplitter.
        """
        
        def __init__(
            self,
            timestamp_col: str,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
        ):
            """
            Args:
                timestamp_col: Column name containing timestamps
                train_ratio: Fraction for training (earliest data)
                val_ratio: Fraction for validation
                test_ratio: Fraction for testing (most recent data)
            """
            super().__init__(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            self.timestamp_col = timestamp_col
        
        def _split(self, df: pd.DataFrame) -> DataSplit:
            # Sort by timestamp instead of random shuffle
            df_sorted = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            
            n = len(df_sorted)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            
            train_df = df_sorted.iloc[:train_end]
            val_df = df_sorted.iloc[train_end:val_end]
            test_df = df_sorted.iloc[val_end:]
            
            return DataSplit(train=train_df, val=val_df, test=test_df)

To further avoid code duplication, we could add a ``_sort()`` method to :class:`~chemtorch.components.data_pipeline.data_splitter.ratio_splitter.RatioSplitter` with random shuffling as default implementation and call it in the ``_split()`` method.
Then the ``TimeBasedSplitter`` could simply override the ``_sort()`` method instead of re-implementing the whole ``_split()`` method.

Representation
==============
Representations convert SMILES strings into data structures suitable for neural networks (graphs, tensors, etc.).
Below is a high-level overview of representation classes, how they are implemented, and source code examples to get you started.

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Representation
     - Description
     - Examples
   * - Fingerprint
     - Fixed-length binary or count vectors encoding molecular substructures occuring in (reaction) SMILES.
     - :class:`~chemtorch.components.representation.fingerprint.drfp.DRFP`
   * - Graph (CGR)
     - Graph representation encoding molecular connectivity as nodes (atoms) and edges (bonds). Uses composable featurizers to extract node and edge features. For reactions, uses condensed graph of reaction (CGR) or similar.
     - :class:`~chemtorch.components.representation.graph.cgr.CGR`, :class:`~chemtorch.components.representation.graph.featurizer.abstract_featurizer.AbstractFeaturizer`, :class:`~chemtorch.components.representation.graph.featurizer.featurizer_compose.FeaturizerCompose`
   * - Token
     - Sequence-based representation encoding molecules/reactions as sequences of discrete tokens (similar to text). Uses external vocabulary file which must first be created and validated (see ``scripts/check_vocab.py``), and uses tokenizers which specify the tokenization scheme.
     - :class:`~chemtorch.components.representation.token.token_representation_base.TokenRepresentationBase`, :class:`~chemtorch.components.representation.token.tokenizer.reaction_tokenizer.ReactionTokenizer`, :class:`~chemtorch.components.representation.token.tokenizer.molecule_tokenizer.MoleculeTokenizerBase`, ``scripts/check_vocab.py``
   * - 3D Graph
     - Graph representation with 3D spatial coordinates capturing molecular geometry and conformational information. Requires external ``.xyz`` files which imposes a constraint on the storage format (requires indexing to associate reactions with 3D data) and requires a custom data pipeline to add a column with folder paths to the 3D data of each reaction.
     - :class:`~chemtorch.components.representation.graph.reaction_3d_graph.Reaction3DGraph`

See :mod:`chemtorch.components.representation` for full API details and implementations.

Best Practices
--------------

1. **Type Hints**: Always specify the generic type parameter indicating the type of the produced data object:

   .. code-block:: python
   
       class MyRep(AbstractRepresentation[torch.Tensor]):  # ✓ Good
           ...
       
       class MyRep(AbstractRepresentation):  # ✗ Missing type info
           ...

2. **Error Handling**: Validate SMILES and provide clear error messages:

   .. code-block:: python
   
       def construct(self, smiles: str):
           mol = Chem.MolFromSmiles(smiles)
           if mol is None:
               raise ValueError(f"Invalid SMILES: {smiles}")
           # ... rest of implementation

3. **Statefulness**: Keep representations stateless. Don't store mutable state:

   .. code-block:: python
   
       # ✓ Good: parameters are immutable
       def __init__(self, radius: int = 2):
           self.radius = radius
       
       # ✗ Bad: mutable state
       def __init__(self):
           self.cache = {}  # Don't cache results

Transform / Augmentation
========================

**Transforms** are commonly used in computer vision to preprocess or augment images (e.g., normalization, rotation, cropping). In ChemTorch, transforms serve a similar purpose for molecular representations.
For example, graph transforms modify graph structure or features (e.g., positional encodings, dummy nodes, feature normalization) and 3D transforms modify spatial coordinates (e.g., random noise, rotation, translation).

Transforms can be composed into pipelines via :class:`~chemtorch.utils.callable_compose.CallableCompose`.
You can also leverage transforms to create additional test data loaders by providing a list or dict of test transforms in the :class:`~chemtorch.core.data_module.DataModule`.
Each entry becomes its own test dataset/dataloader with the specified transform applied.

**Augmentations** are built on top of transforms to augment training data with transformed versions, making models invariant to specific perturbations.

See :mod:`chemtorch.components.transform` and :mod:`chemtorch.components.augmentation` for available transforms and augmentations, and consult the API docs for implementation details.

Best Practices
--------------

#. **Immutability**: Avoid modifying the input in-place. Clone if needed:

   .. code-block:: python
   
       def __call__(self, data: Data) -> Data:
           data = data.clone()  # ✓ Safe
           data.x = data.x * 2
           return data

#. **Type Consistency**: Preserve the object type:

   .. code-block:: python
   
       def __call__(self, data: Data) -> Data:
           # Return same type
           return data

#. **Docstrings**: Document what the transform does:

   .. code-block:: python
   
       class MyTransform(AbstractTransform[Data]):
           """
           Brief description of what this transform does.
           
           Args:
               param1: Description
               param2: Description
           """

Model
=====
ChemTorch models are standard PyTorch modules with no special abstract base class.
The only requirement is that the ``forward()`` method accepts batched inputs matching the representation type and returns predictions as ``torch.Tensor`` with shape ``(batch_size, output_dim)``.
For example, graph models typically accepts PyTorch Geometric ``Batch`` objects.

Best practices:
---------------
#. **Device Handling**: Let PyTorch Lightning handle device placement (don't call ``.to(device)`` manually)
#. **Documentation**: Document the ``__init__()`` method and the input/output shapes and requirements of the ``forward()`` method.
#. **Modularization**: start simple and protype your neural network in a single file but modlarize it as variants emerge. The :class:`~chemtorch.components.model.gnn.gnn.GNN` is the perfact example since it decomposes into encoder blocks, a configurable layer stack, pooling, and a head. See, :mod:`chemtorch.components.model.gnn` and :mod:`chemtorch.components.layer`. This decomposition makes it easy to swap components, run ablations, and scale complexity while keeping everything testable.

.. _common-patterns:

Common Patterns
===============

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Pattern
     - Description
     - Examples
   * - Composition
     - Build complex behavior from small, focused components.
     - :class:`~chemtorch.components.representation.graph.featurizer.featurizer_compose.FeaturizerCompose`, :class:`~chemtorch.src.chemtorch.utils.callable_compose.CallableCompose`
   * - Orchestration & Delegation
     - Keep top-level components lean; orchestrate and delegate specifics.
     - :class:`~chemtorch.components.representation.graph.cgr.CGR`, :class:`~chemtorch.components.representation.token.token_representation_base.TokenRepresentationBase`, :class:`~chemtorch.components.representation.token.tokenizer.reaction_tokenizer.ReactionTokenizer`
   * - External Artifact Contracts
     - Components require external data/artifacts referenced via paths.
     - :class:`~chemtorch.components.representation.token.token_representation_base.TokenRepresentationBase`, :class:`~chemtorch.components.representation.graph.reaction_3d_graph.Reaction3DGraph`

Next Steps
==========

- **Create default configs** for your components to make them swappable from the CLI, see :ref:`config` and the ``conf`` folder for composition patterns and overrides.
- **Add integration tests** to validate your components work together with the data pipelines, transforms, and models end-to-end, and detect regressions, see :ref:`testing`.
- **Contribute** you tested components to help us grow ChemTorch, see :ref:`contributing`.
