.. _api-autodoc-components:

==========
Components
==========

The components package contains all swappable building blocks for the ChemTorch pipeline.

Data Pipeline
=============

Data Source
-----------

.. automodule:: chemtorch.components.data_pipeline.data_source.abstract_data_source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_source.single_csv_source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_source.pre_split_csv_source
   :members:
   :undoc-members:
   :show-inheritance:

Column Mapper
-------------

.. automodule:: chemtorch.components.data_pipeline.column_mapper.abstract_column_mapper
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.column_mapper.column_filter_rename
   :members:
   :undoc-members:
   :show-inheritance:

Data Splitter
-------------

.. automodule:: chemtorch.components.data_pipeline.data_splitter.abstract_data_splitter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_splitter.data_splitter_base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_splitter.ratio_splitter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_splitter.scaffold_splitter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_splitter.target_splitter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_splitter.size_splitter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_splitter.reaction_core_splitter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.data_pipeline.data_splitter.index_splitter
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline
--------

.. automodule:: chemtorch.components.data_pipeline.simple_data_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Representation
==============

Base Classes
------------

.. automodule:: chemtorch.components.representation.abstract_representation
   :members:
   :undoc-members:
   :show-inheritance:

Graph Representations
---------------------

.. automodule:: chemtorch.components.representation.graph.cgr
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.representation.graph.reaction_3d_graph
   :members:
   :undoc-members:
   :show-inheritance:

Fingerprint Representations
---------------------------

.. automodule:: chemtorch.components.representation.fingerprint.drfp
   :members:
   :undoc-members:
   :show-inheritance:

Transform
=========

Base Classes
------------

.. automodule:: chemtorch.components.transform.abstract_transform
   :members:
   :undoc-members:
   :show-inheritance:

Graph Transforms
----------------

.. automodule:: chemtorch.components.transform.graph_transform
   :members:
   :undoc-members:
   :show-inheritance:

Augmentation
============

.. automodule:: chemtorch.components.augmentation.abstract_augmentation
   :members:
   :undoc-members:
   :show-inheritance:

Model
=====

Graph Neural Networks
---------------------

.. automodule:: chemtorch.components.model.gnn.gnn
   :members:
   :undoc-members:
   :show-inheritance:

Encoder
~~~~~~~

.. automodule:: chemtorch.components.model.gnn.encoder
   :members:
   :undoc-members:
   :show-inheritance:

Pooling
~~~~~~~

.. automodule:: chemtorch.components.model.gnn.pool
   :members:
   :undoc-members:
   :show-inheritance:

Other Models
------------

.. automodule:: chemtorch.components.model.mlp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.model.dimenet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: chemtorch.components.model.han
   :members:
   :undoc-members:
   :show-inheritance:

Layers
======

Layer Stack
-----------

.. automodule:: chemtorch.components.layer.layer_stack
   :members:
   :undoc-members:
   :show-inheritance:

GNN Layers
----------

.. automodule:: chemtorch.components.layer.gnn_layer.dmpnn_stack
   :members:
   :undoc-members:
   :show-inheritance:
