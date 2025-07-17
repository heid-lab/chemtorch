<div align="center">

![ChemTorch](images/chemtorch.png)

[![tests](https://github.com/heid-lab/chemtorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/heid-lab/chemtorch/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/heid-lab/chemtorch/main/pyproject.toml&query=$.project.requires-python&label=python&color=blue)](#)
<!--
When chemtorch is on PyPI uncomment this ^^
[![PyPI version](https://img.shields.io/pypi/v/chemtorch.svg)](https://pypi.org/project/chemtorch)
[![Python versions](https://img.shields.io/pypi/pyversions/chemtorch.svg)](https://pypi.org/project/chemtorch)
[![Downloads](https://img.shields.io/github/downloads/heid-lab/chemtorch/total.svg)](https://github.com/heid-lab/chemtorch/releases)
-->

[Installation](#installation) | [Data](#data) | [Usage](#usage) | [Citation](#citation)

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
pip install torch && \
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

```
uv sync --locked --all-extras --dev
uv pip install torch_geometric \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-${{ env.TORCH }}+${{ env.CUDA }}.html
```

## Data

Get the data from https://github.com/heid-lab/reaction_database and add it to the `data` folder.

<!-- TODO: Add the following sections:
## ✨ Highlights/Features
## 🤝 Contributing
## 💬 Support
## 🚀 Roadmap

<!-- TODO: move this to web documentation -->
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
@article{landsheere_chemtorch_2025,
	title = {ChemTorch: A Deep Learning Framework for Benchmarking and Developing Chemical Reaction Property Prediction Models},
	doi = {10.26434/chemrxiv-2025-9mggj},
	journal = {ChemRxiv},
	author = {De Landsheere, Jasper and Zamyatin, Anton and Karwounopoulos, Johannes and Heid, Esther},
	year = {2025},
}
```

This framework was inspired by:
- [GraphGPS](https://github.com/rampasek/GraphGPS/tree/main)
- [GraphGym](https://github.com/snap-stanford/GraphGym)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
