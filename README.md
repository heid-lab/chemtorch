# DeepRxn

DeepRxn software package

## Installation

Clone this repository and change directory:
```
git clone ssh://git@gitlab.tuwien.ac.at:822/e165-03-1_theoretische_materialchemie/deeprxn.git
cd deeprxn
```
We recommend to install the package inside a conda environment (or any other virtual environment of your choice). Follow the pytorch and torch_geometric installation instructions to install for GPUs (here, for CPUs):

For torch scatter and torch sparse, you need might also need to install specific binaries.
```
conda create -n deeprxn python=3.10
conda activate deeprxn
pip install --upgrade pip setuptools wheel
pip install rdkit numpy scikit-learn torch pandas
pip install hydra-core --upgrade
pip install torch_scatter torch_sparse
pip install torch_geometric
pip install -e .

```

## Test

The scripts provided in the `scripts` folder are meant as a small example:
```
python scripts/test.py mode=train data=barriers_e2  epochs=100 representation=CGR
```

## Copyright

Copyright (c) 2024, E. Heid
