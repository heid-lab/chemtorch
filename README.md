<div align="center">

![ChemTorch](docs/source/_static/chemtorch_logo_light.png#gh-light-mode-only)
![ChemTorch](docs/source/_static/chemtorch_logo_dark_lightbackground.png#gh-dark-mode-only)

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

First clone this repo and navigate to it:
```bash
git clone https://github.com/heid-lab/chemtorch.git
cd chemtorch
```
Then follow the instructions below to install ChemTorch's dependencies using you package manager of choice.

### Via conda

```bash
conda create -n chemtorch python=3.10 && \
conda activate chemtorch && \
pip install rdkit numpy==1.26.4 scikit-learn pandas && \
pip install torch && \
pip install hydra-core && \
pip install wandb && \
pip install ipykernel && \
pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html && \
pip install -e . && \
```

For GPU usage
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

### Via uv
First install `uv` following the [official installation instructions](https://docs.astral.sh/uv/).
Then run:
```bash
uv sync
uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric  --no-build-isolation
```

To also install development and docuemntation dependencies add the `--groups` option followed by `dev` or `docs`.
Alternatively, you can also use `--all-groups` to install both.

## Data

Get the data from https://github.com/heid-lab/reaction_database and add it to the `data` folder.

<!-- TODO: Add the following sections:
## ✨ Highlights/Features
## 🤝 Contributing
## 💬 Support
## 🚀 Roadmap
-->

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
