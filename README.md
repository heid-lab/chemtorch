<div align="center">

![ChemTorch](images/chemtorch.png)

[Installation](#installation) | [Data](#data) | [Usage](#usage) | [Citation](#citation)

</div>

## Introduction

ChemTorch is a modular framework for developing and benchmarking deep learning models on chemical reaction data. The framework supports multiple families of reaction representations, neural network architectures, and downstream tasks.

The code is provided under MIT license, making it freely available for both academic and commercial use.

## Installation

### Via conda

```
conda create -n chemtorch python=3.10 && \
conda activate chemtorch && \
pip install rdkit numpy==1.26.4 scikit-learn pandas && \
pip install torch==2.5.1 && \
pip install hydra-core && \
pip install torch_geometric && \
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html && \
pip install wandb && \
pip install ipykernel && \
pip install -e .
```

For GPU usage
```
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

### Via uv

For installing with uv, first install uv, for example via
```
pip install uv
```

Then run
```
uv sync -n
uv add torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric  --no-build-isolation -n
```

## Data

Get the data from https://github.com/heid-lab/reaction_database and add it to the `data` folder.

## Usage

For a short demo, see `scripts/demo.ipynb`.

To run the experiments, you can use the following commands:

Graph-based: GNN + CGR
```
python scripts/main.py +experiment=graph dataset.subsample=0.05
```
Token-based: HAN + Tokenized SMILES
```
python scripts/main.py +experiment=token dataset.subsample=0.05
```
Fingerprint-based: MLP + DRFP
```
python scripts/main.py +experiment=fingerprint dataset.subsample=0.001
```
3D-based: DimeNetplusplus + XYZ coordinates
```
python scripts/main.py +experiment=xyz dataset.subsample=0.05
```

Using the terminal, you can easily change hyperparameters. For example, to change the dataset:
``` 
python scripts/main.py +experiment=graph dataset.subsample=0.05 data_ingestor=sn2
```

For simple sweeps, you can:
```
python scripts/main.py --multirun +experiment=graph dataset.subsample=0.05 data_ingestor=sn2,e2,cycloadd
```

## Citation
If you use this code in your research, please cite the following paper:

```
@article{}
```

This framework was inspired by:
- [GraphGPS](https://github.com/rampasek/GraphGPS/tree/main)
- [GraphGym](https://github.com/snap-stanford/GraphGym)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
