.. _overview:

===================
ChemTorch Overview
===================

This overview gives you a quick glimpse of the main modules and how they fit together.
What we'll cover:

#. the high-level components,
#. how the core modules wire them together, and 
#. how to assemble everything in code. 

Configuration is covered separately in :ref:`config`.

.. figure:: ../_static/chemtorch_pipeline.svg
   :alt: ChemTorch pipeline architecture
   :width: 100%
   :align: center

   **ChemTorch pipeline**: raw data → data pipeline → representation → transform → model

.. note::

   If you prefer a linear walk-through, check out
   :doc:`../examples/pipeline_from_scratch` or run the interactive notebook
   at ``docs/source/examples/pipeline_from_scratch.ipynb``.


Components (``chemtorch.components``)
=====================================

Data Pipeline
-------------

The **data pipeline** loads raw chemical data, standardizes column names, and splits into train/val/test sets. It is composed of three swappable components that transform data step-by-step:

#.  **Data Source** — Loads raw data from CSV or other sources:

    .. code-block:: python

        from chemtorch.components.data_pipeline import SingleCSVSource
        
        source = SingleCSVSource(data_path="data.csv")
        df = source.load()  # Returns pandas.DataFrame

#.  **Column Mapper** — Filters and renames columns to match ChemTorch's expected format (e.g., ``smiles``, ``label``):

    .. code-block:: python

        from chemtorch.components.data_pipeline import ColumnFilterAndRename
        
        mapper = ColumnFilterAndRename(
            smiles="rxn_smiles",
            label="barrier_energy",
        )
        df_mapped = mapper(df)  # Returns pandas.DataFrame (same rows, renamed/filtered columns)

#.  **Data Splitter** — Divides data into train/val/test sets. Returns a ``DataSplit`` object:

    .. code-block:: python

        from chemtorch.components.data_pipeline import RatioSplitter
        
        splitter = RatioSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        data_split = splitter(df_mapped)  # Returns DataSplit(train=..., val=..., test=...)

The three are orchestrated by **SimpleDataPipeline**:

.. code-block:: python

    from chemtorch.components.data_pipeline import SimpleDataPipeline
    
    pipeline = SimpleDataPipeline(
        data_source=source,
        column_mapper=mapper,
        data_splitter=splitter,
    )
    data_split = pipeline()  # Runs: load → map → split

Representation
--------------

The **representation** converts chemical structures (typically SMILES strings) into data structures suitable for neural networks (e.g. tensors or graphs).
ChemTorch provides multiple representations, for example the **Condensed Graph of Reaction (CGR)**:

.. code-block:: python

    from chemtorch.components.representation.graph import CGR
    
    representation = CGR(
        atom_featurizer=...,  # Composition of atomic featurizers (e.g., atom type, charge, aromaticity)
        bond_featurizer=...,  # Composition of bond featurizers (e.g., bond type, aromatic)
    )
    
    # Apply to a single SMILES
    data_obj = representation.construct("CC>>CCO")  # Returns a graph data object

Transform / Augmentation
------------------------

**Transforms** are optional pre-processing or augmentation steps applied to individual/batched data objects.
For example, we can add positional encoding to graphs:

.. code-block:: python

    from chemtorch.components.transform import RandomWalkPETransform
    
    # Add random walk positional encodings to graph nodes
    transform = RandomWalkPETransform(walk_length=16)
    
    data_obj = representation.construct("CC>>CCO")
    transformed = transform(data_obj)  # Adds positional encodings to the graph

Other graph transforms could include dummy node injection, node/edge masking, and subgraph sampling.

Model
-----

**Models** are PyTorch ``nn.Module`` architectures that take in data objects and produce predictions.
ChemTorch emphasizes modular, composable models.
**Graph Neural Networks (GNNs)** are a prime example because they can be hierarchically decomposed into four components:


- **Encoder**: projects node/edge features to hidden dimension
- **Layer Stack**: repeatedly applies message-passing blocks (e.g., DMPNN, GIN, GAT)
- **Pool**: aggregates node embeddings to graph embedding
- **Head**: final fully-connected layer(s) for prediction

Each component can be swapped (e.g., different encoder, different message-passing layer, different pooling strategy) without touching the others, making GNNs highly modular.

**Directed Message-Passing Neural Network (D-MPNN)** example:

.. code-block:: python

    import torch

    from chemtorch.components.layer import LayerStack
    from chemtorch.components.layer.gnn_layer import (
        DMPNNBlock,
        DMPNNConv,
        DMPNNStack,
        EdgeToNodeEmbedding,
    )
    from chemtorch.components.model.gnn import DirectedEdgeEncoder, GNN, GlobalPool

    def make_dmpnn_block():
        return DMPNNBlock(
            graph_conv=DMPNNConv(in_channels=256, out_channels=256),
            hidden_channels=256,
            residual=True,
            ffn=True,
        )

    model = GNN(
        encoder=DirectedEdgeEncoder(
            in_channels=num_node_features + num_edge_features,
            out_channels=256,
        ),
        layer_stack=DMPNNStack(
            dmpnn_blocks=LayerStack(layer=make_dmpnn_block, depth=4),
            edge_to_node_embedding=EdgeToNodeEmbedding(
                embedding_size=256,
                num_node_features=num_node_features,
            ),
        ),
        pool=GlobalPool(aggr="mean"),
        head=torch.nn.Linear(256, output_dim),
    )
    
    # Forward pass
    output = model(batch)  # batch is a batched graph of single graph data objects



Core Modules (``chemtorch.core``)
=================================
ChemTorch's core modules are built on top of `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`__, which provides a standardized, high-quality training loop (device placement, logging, checkpointing, distributed execution). You do not need to be a Lightning expert to use ChemTorch—common workflows and simple components work out of the box. If you plan to go beyond implementing individual components and want to adapt the core modules themselves, we recommend skimming the Lightning tutorials or a short crash course to get familiar with ``LightningModule``, ``LightningDataModule``, and ``Trainer``.

Data Module
-----------

The **Data Module** is a ``LightningDataModule`` and wires together the data pipeline, representation, and optional transforms/augmentations:

- Runs the pipeline (load → map → split) to get a ``DataSplit``/``DataFrame``.
- Applies the representation to construct model-ready data objects.
- Wraps datasets and instantiates dataloaders for ``train``, ``val``, ``test``, and ``predict``.

Basic usage:

.. code-block:: python

    from functools import partial

    from torch_geometric.loader import DataLoader

    from chemtorch.core import DataModule

    data_module = DataModule(
        data_pipeline=pipeline,
        representation=representation,
        dataloader_factory=partial(DataLoader, batch_size=64),
        transform=None,
        augmentations=None,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

Routine
-------

The **Routine** is a ``LightningModule`` that wraps the model and training logic (loss, optimizer, LR scheduler, metrics, and step hooks). Use a specific routine for your task (e.g., regression or classification).

Example (regression):

.. code-block:: python

    import torch

    from chemtorch.core import RegressionRoutine

    routine = RegressionRoutine(
        model=model,
        loss=torch.nn.MSELoss(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
        metrics=None,
    )

Supervised routines expose the usual Lightning hooks (``training_step``, ``validation_step``, etc.) internally, so you focus on component selection rather than boilerplate.

Trainer
-------

The **Trainer** (PyTorch Lightning) drives execution. It handles device placement, logging, checkpointing, and distributed training.

.. code-block:: python

    from lightning import Trainer

    trainer = Trainer(max_epochs=10, log_every_n_steps=10)
    trainer.fit(routine, datamodule=data_module)
    trainer.test(routine, datamodule=data_module)
    # trainer.predict(routine, datamodule=data_module)

Together: **DataModule** provides dataloaders; **Routine** runs optimization and evaluation; **Trainer** orchestrates the loop.

Putting everything together in code
===================================

Below is a minimal hands on example showing how the pieces fit together.

.. code-block:: python

    from functools import partial

    import torch
    from lightning import Trainer
    from torch_geometric.loader import DataLoader

    from chemtorch.components.data_pipeline import (
        ColumnFilterAndRename,
        RatioSplitter,
        SimpleDataPipeline,
        SingleCSVSource,
    )
    from chemtorch.components.layer import LayerStack
    from chemtorch.components.layer.gnn_layer import (
        DMPNNBlock,
        DMPNNConv,
        DMPNNStack,
        EdgeToNodeEmbedding,
    )
    from chemtorch.components.model.gnn import DirectedEdgeEncoder, GNN, GlobalPool
    from chemtorch.components.representation.graph import CGR
    from chemtorch.core import DataModule, RegressionRoutine

    # 1) Data pipeline
    pipeline = SimpleDataPipeline(
        data_source=SingleCSVSource(data_path="data.csv"),
        column_mapper=ColumnFilterAndRename(smiles="rxn_smiles", label="barrier"),
        data_splitter=RatioSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1),
    )

    # 2) Representation (CGR)
    representation = CGR(
        atom_featurizer=...,   # compose your atom featurizers
        bond_featurizer=...,   # compose your bond featurizers
    )

    # 3) DataModule (wires pipeline + representation + optional transforms/augmentations)
    dataloader_factory = partial(DataLoader, batch_size=64)
    data_module = DataModule(
        data_pipeline=pipeline,
        representation=representation,
        dataloader_factory=dataloader_factory,
        transform=None,
        augmentations=None,
    )

    # 4) Model (GNN with D-MPNN stack)
    num_node_features = 88  # match your representation
    num_edge_features = 22  # match your representation
    hidden_channels = 256

    def make_dmpnn_block():
        return DMPNNBlock(
            graph_conv=DMPNNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
            ),
            hidden_channels=hidden_channels,
            residual=True,
            ffn=True,
        )

    model = GNN(
        encoder=DirectedEdgeEncoder(
            in_channels=num_node_features + num_edge_features,
            out_channels=hidden_channels,
        ),
        layer_stack=DMPNNStack(
            dmpnn_blocks=LayerStack(layer=make_dmpnn_block, depth=4),
            edge_to_node_embedding=EdgeToNodeEmbedding(
                embedding_size=hidden_channels,
                num_node_features=num_node_features,
            ),
        ),
        pool=GlobalPool(aggr="mean"),
        head=torch.nn.Linear(hidden_channels, 1),
    )

    # 5) Routine (wraps model with training logic)
    routine = RegressionRoutine(
        model=model,
        loss=torch.nn.MSELoss(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
        metrics=None,
    )

    # 6) Trainer (executes)
    trainer = Trainer(max_epochs=10, log_every_n_steps=10)
    trainer.fit(routine, datamodule=data_module)
    trainer.test(routine, datamodule=data_module)
    # trainer.predict(routine, datamodule=data_module)

This is very close to what ChemTorch actually does under the hood (ignoring all other software features for a second).
