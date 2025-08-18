Installation
============

.. role:: bash(code)
    :language: bash

First clone this repo and navigate to it:

.. parsed-literal::

    git clone https://github.com/heid-lab/chemtorch.git
    cd chemtorch   

Via conda
---------

See

.. parsed-literal::

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

For GPU usage:

.. parsed-literal::

    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

Via uv
-----------

For installing with `uv`, first install `torch` and `uv`, for example via

.. parsed-literal::

    pip install torch uv
    uv sync -n
    uv add torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric  --no-build-isolation -n chemtorch
