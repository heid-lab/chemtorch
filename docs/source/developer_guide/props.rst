.. _props:

==================
Runtime Properties
==================

Model configurations often depend on values that are external to the configuration itself---properties of the dataset or the data representation being used.
For example:

-   **Standardization parameters**:
    Standardization rescales targets to zero mean and unit variance using statistics computed on the training split to improve optimization stability.
    During training, targets are standardized; at inference, predictions are de-standardized (inverse-transformed) back to the original scale.
    In ChemTorch this is handled by the regression routine's :py:class:`~chemtorch.utils.standardizer.Standardizer` but it depends on the mean and standard deviation of the training targets.
    
    .. code-block:: yaml

        routine:
            _target_: chemtorch.core.routine.RegressionRoutine
            target_mean: ???  # Depends on training data
            target_std: ???   # Depends on training data
    
    When you change the split you need to recompute the mean and standard deviation of the training targets.

-   **GNN with interchangeable featurizers**:
    Encoders consume per-node (or per-edge) feature vectors produced by your featurizer; the ``input_dim`` must match the length of these vectors.

    .. code-block:: yaml

        # conf/model/gnn.yaml
        _target_: chemtorch.components.model.GNNModel
        encoder:
            _target_: chemtorch.components.model.NodeEncoder
            input_dim: ???  # This depends on the featurizer!
            hidden_dim: 128

    Because featurizers are interchangeable, hardcoding `input_dim` couples configs to one featurizer and breaks portability.

To reduce friction and avoid hardcoding values, ChemTorch uses a *property system* that lets you declare properties to be inferred at runtime and specify where they should be injected.

Configuring Runtime Properties
==============================
Let's return to our GNN encoder example.
Instead of hardcoding the input dimension we can declare the :py:class:`~chemtorch.core.property_system.NumNodeFeatures` property to infer the size of the feature vector at runtime:

.. code-block:: yaml

    # conf/experiment/my_experiment.yaml
    props:
      num_node_features:
        _target_: chemtorch.core.property_system.NumNodeFeatures
        name: num_node_features
        source: any
        add_to_cfg: true

      label_mean:
        _target_: chemtorch.core.property_system.LabelMean
        name: label_mean
        source: train
        add_to_cfg: true

      label_std:
        _target_: chemtorch.core.property_system.LabelStd
        name: label_std
        source: train
        add_to_cfg: true

      precompute_time:
        _target_: chemtorch.core.property_system.PrecomputeTime
        name: precompute_time
        source: all
        log: true


Here, the top-level config key ``num_node_features`` is just an index within ``props`` (useful for Hydra defaults list inclusion shown below).

.. list-table:: Property config keys
   :header-rows: 1
   :widths: 15 30 55

   * - Key
     - Allowed values
     - Purpose
   * - ``_target_``
     - Any valid import path
     - Python path to the property class
   * - ``source``
     - ``train`` | ``val`` | ``test`` | ``predict`` | ``any`` | ``all``
     - Dataset partition used to compute the property. For some properties this does not matter (e.g., feature vector size is typically the same across partitions, so ``any`` is fine). For standardization, however, the mean and standard deviation must be computed on the ``train`` set to avoid data leakage. When ``source`` is not ``any``, the injected key is prefixed with the source name followed by an underscore (see example below).
   * - ``add_to_cfg``
     - ``true`` | ``false`` (default ``false``)
     - Inject the computed value back into the config at the very top level for interpolation from anywhere in the config tree.
   * - ``log``
     - ``true`` | ``false`` (default ``false``)
     - Log the computed value to W&B
   * - ``name``
     - String
     - Top-level config key under which the value is injected into the config or the name under which the property is logged to W&B.

Interpolating Runtime Properties
================================
Once you configure your runtime properties, you can use the ``name`` to interpolate their runtime values in your configs:

.. code-block:: yaml

    # conf/model/gnn.yaml
    _target_: chemtorch.components.model.GNNModel
    encoder:
      _target_: chemtorch.components.model.NodeEncoder
      input_dim: ${num_node_features}  # Automatically computed!
    hidden_dim: 128

    # conf/routine/regression.yaml
    _target_: chemtorch.core.routine.RegressionRoutine
    target_mean: ${train_label_mean}
    target_std: ${train_label_std}

Now the same config works with any featurizer or dataset, the standardization parameters are computed automatically from the training dataset, and the time it takes to precompute the representations for each property are logged to W&B.

Reusing Existing Properties
===========================
ChemTorch already includes a series of useful properties.
For a complete list of available built-in properties, see :py:mod:`chemtorch.core.property_system`.

Each built-in property comes with a default config at ``conf/props`` so that you can just include them in the defaults list of your experiment config:

.. code-block:: yaml

    # conf/experiment/my_experiment.yaml
    defaults:
      - /props/num_node_features@props.num_node_features
      - /props/num_edge_features@props.num_edge_features
      - /props/label_mean@props.label_mean
      - /props/label_std@props.label_std

This declares four properties to be computed.
Each points to a config file in ``conf/props/``.

.. note::
  The defaults list syntax is a bit special here because we include configs that do not sit directly under the base config. Use the absolute path from the config root to the file, followed by the target node where it should be placed: ``/path/to/config@place.to.go``.
  See the `Hydra documentation <https://hydra.cc/docs/advanced/defaults_list/>`__ for full details.

Implementing Custom Properties
===============================

To create a custom property, subclass :py:class:`~chemtorch.core.property_system.DatasetProperty` and implement the ``compute()``.
The method takes in an instance of :py:class:`~chemtorch.core.dataset_base.DatasetBase` corresponding to the partition selected as ``source``.
Tip: For properties that depend on a single sample's representation (e.g., feature vector length), access the first item: ``first_item = dataset[0]`` and extract the data object from tuples as needed.

Once you've implemented your property, create a config file for it in ``conf/props/`` and use it in your configs as described above.

Implementation Guidelines
-------------------------

1. **Handle empty datasets**: Always check if the dataset is empty and return a sensible default
2. **Extract data correctly**: The dataset may return tuples of ``(data, label)`` if labeled, so use:
   
   .. code-block:: python
   
       first_item = dataset[0]
       data = first_item[0] if isinstance(first_item, tuple) else first_item

3. **Validate requirements**: Check for required attributes (e.g., labels, precomputation status) and raise informative errors
4. **Type hints**: Use proper type hints for the return value to enable IDE support
5. **Documentation**: Document what data the property requires and any assumptions
