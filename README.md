<div align="center">

<picture>
	<source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/chemtorch_logo_dark.svg">
	<source media="(prefers-color-scheme: light)" srcset="docs/source/_static/chemtorch_logo_light.svg">
	<img alt="ChemTorch logo" src="docs/source/_static/chemtorch_logo_light.svg" width="360" />
</picture>

<h3>A Modular Deep Learning for Reactive Chemistry</h3>

[![tests](https://github.com/heid-lab/chemtorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/heid-lab/chemtorch/actions)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://heid-lab.github.io/chemtorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/heid-lab/chemtorch/main/pyproject.toml&query=$.project.requires-python&label=python&color=blue)](#)
<!-- 
When chemtorch is on PyPI uncomment this ^^
[![PyPI version](https://img.shields.io/pypi/v/chemtorch.svg)](https://pypi.org/project/chemtorch)
[![Python versions](https://img.shields.io/pypi/pyversions/chemtorch.svg)](https://pypi.org/project/chemtorch)
[![Downloads](https://img.shields.io/github/downloads/heid-lab/chemtorch/total.svg)](https://github.com/heid-lab/chemtorch/releases) -->

[Quick Start](#quick-start) |
[Documentation](https://heid-lab.github.io/chemtorch) |
[Contributing](#contributing) |
[White Paper](#white-paper) |
[Citation](#citation)

</div>

ChemTorch is a modular research framework for deep learning of chemical reactions.

- 🚀 **Streamline your research workflow**: seamlessly assemble modular deep learning pipelines, track experiments, conduct hyperparameter sweeps, and run benchmarks.
- 💡 **Multiple reaction representations** with baseline implementations including SMILES tokenizations, molecular graphs, 3D geometries, and fingerprint descriptors.
- ⚙️ **Preconfigured data pipelines** for common benchmark datasets including RDB7, cycloadditions, USPTO-1k, and more.
- 🔬 **OOD evaluation** via chemically informed data splitters (size, target, scaffold, reaction core, ...).
- 🗂️ **Extensible component library** (growing) for all parts of the ChemTorch pipeline.
- 🔄 **Reproducibility by design** with Weights & Biases experiment tracking and a guide for setting up reproducibility tests.

<!-- TODO: link to hosted GIF/mp4 of the ChemTorch CLI in action -->

<a id="quick-start"></a>

## 🐎 Quick Start
Follow the [Quick Start guide](https://heid-lab.github.io/chemtorch/getting_started/quick_start.html) to install all dependencies, download some data, and run your first experiment!
For more, checkout the [official ChemTorch documentation](https://heid-lab.github.io/chemtorch)!

<a id="white-paper"></a>

## 📄 Read the white paper
Check out the [software/benchmarking paper](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02645) published in JCIM for a few examples of what you can already do with ChemTorch!

## 💬 Support
If you want to ask a question, report a bug, or suggest a feature feel free to open an issue on our [issue tracker](https://github.com/heid-lab/chemtorch/issues) and we will get back to you :)
<!-- TODO: add Discord -->

## 🧭 Stability & Roadmap
ChemTorch is in active development and the public CLI/configuration API may change between releases.
To detect breaking changes early and safeguard your workflows:
- Check the [release notes](https://github.com/heid-lab/chemtorch/releases).
- Add and run [Integrity & Reproducibility tests](https://heid-lab.github.io/chemtorch/advanced_guide/integration_tests.html) for your experiments to ensure reproducibility of past results with newer releases.

<a id="contributing"></a>

## 🤝 Contributing
We welcome contributions.
Please read the [contribution guide](CONTRIBUTING.md) before opening issues or PRs.

<a id="citation"></a>

## ❤️ Citation
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

## 📋 License
This project is licensed under the [MIT License](LICENSE).

## Thanks & inspiration

ChemTorch builds on and was inspired by many excellent open-source projects and community work — thank you to the authors and maintainers <3

- [Hydra](https://hydra.cc/) — flexible configuration and experiment management
- [PyTorch Lightning](https://www.pytorchlightning.ai/) — cleaner training loops and logging
- [Weights & Biases](https://wandb.ai/site/models/) — experiment tracking and visualization in one place
- [GraphGPS](https://github.com/rampasek/GraphGPS) and [GraphGym](https://github.com/snap-stanford/GraphGym) — modular GNN repos which inspired this framework 
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) — project structure and integration patterns
