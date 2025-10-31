"""Utility scripts

This folder contains small helper scripts used during development and for reproducibility workflows.

Top-level helpers
-----------------

These two are the most commonly used helpers; they appear at the top of this README and are linked to the hosted documentation.

- `wandb_to_hydra.py` — convert a Weights & Biases run into a Hydra config snippet (see the hosted docs for usage and examples):
  https://heid-lab.github.io/chemtorch/user_guide/reproducibility.html
- `collect_env.py` — collect hardware and software metadata (CPU, GPU, PyTorch, NumPy, BLAS, pip freeze) and write YAML. See hosted docs for guidance on recording metadata with baselines:
  https://heid-lab.github.io/chemtorch/advanced_guide/integration_tests.html

Other scripts
-------------

Below are the other utility scripts in this directory.

Vocabulary analysis — `check_vocab.py`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script provides comprehensive vocabulary analysis for chemical datasets using Hydra configuration and reuses the existing ChemTorch data pipeline and tokenizer configurations.

Features

- Reuses ChemTorch configs: leverages existing data_pipeline and tokenizer configurations
- Tokenization statistics: token frequencies, length distributions, unique token counts
- OOV analysis: out-of-vocabulary analysis against existing vocabularies
- Artifact detection: detection of problematic tokens (whitespace, Unicode issues, malformed chemical notation)
- Split data support: handle both single datasets and pre-split data automatically

Usage

Basic usage

```bash
# Run with default configuration (uses USPTO-1K data pipeline)
python scripts/check_vocab.py

# Use different dataset by changing data pipeline
python scripts/check_vocab.py data_pipeline=rdb7_fwd
python scripts/check_vocab.py data_pipeline=rgd1
python scripts/check_vocab.py data_pipeline=sn2

# Limit analysis and change reporting
python scripts/check_vocab.py max_samples=1000 report_top_k=50
```

Configuration parameters

- `data_pipeline`: Which data pipeline config to use
- `tokenizer`: Which tokenizer config to use
- `max_samples`: Limit number of samples (null = all)
- `report_top_k`: Number of top tokens to report
- `min_frequency`: Minimum frequency threshold
- `detect_artifacts`: Enable artifact detection
- `unicode_analysis`: Enable Unicode category analysis
- `whitespace_analysis`: Enable whitespace detection
- `vocab_path`: Path to existing vocabulary for OOV analysis
- `display_bad_tokens`: Displays a summary of bad tokens causing OOV issues or artifacts
- `seed`: Random seed for reproducibility

Output

The script outputs a comprehensive summary to the console including:

- Total and unique token counts
- Average and max token lengths
- OOV rate (if vocabulary provided)
- Detected artifacts by category
- Top frequent tokens
- Token length distribution
- Unicode category summary

Data requirements

The script expects the data pipeline to produce data with a standard `smiles` column containing SMILES strings. This is the convention used throughout ChemTorch.

Hydra target resolver — `resolve_target.py`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This small helper lets you quickly open the Python file corresponding to a `_target_:` entry in any Hydra YAML config. It's convenient to wire to a VS Code task + keybinding so you can jump from a config to the class definition even when the class is re-exported through an `__init__.py`.

Setup instructions

1. Create a VS Code task in `.vscode/tasks.json` (example):

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Hydra: Go to _target_ at cursor",
      "type": "shell",
      "command": "python",
      "args": [
        "scripts/resolve_target.py",
        "--at-cursor",
        "${file}",
        "${lineNumber}"
      ],
      "presentation": {
        "echo": true,
        "reveal": "silent",
        "focus": false,
        "panel": "shared"
      },
      "problemMatcher": []
    }
  ]
}
```

2. Add a keybinding in `keybindings.json` (example):

```json
{
  "key": "ctrl+f12",
  "command": "workbench.action.tasks.runTask",
  "args": "Hydra: Go to _target_ at cursor",
  "when": "editorTextFocus && editorLangId == yaml"
}
```

Usage

1. Open a Hydra YAML config in VS Code.
2. Place the cursor on the `_target_:` line you want to resolve.
3. Press your keybinding (e.g. `Ctrl+F12`). The script will open the Python file at the class definition.

Tip

To resolve the first `_target_` in a file (not at cursor):

```sh
python scripts/resolve_target.py --top-target path/to/config.yaml
```

Or resolve a specific target string:

```sh
python scripts/resolve_target.py package.module.ClassName
```

If you want me to further polish this README (add examples for `wandb_to_hydra.py`, usage examples for `collect_env.py`, or add a small index in `scripts/`), I can do that next.

"""
# Documentation of Utiliry Scripts
## Vocabulary Analysis with `check_vocab.py`

This script provides comprehensive vocabulary analysis for chemical datasets using Hydra configuration and reuses the existing ChemTorch data pipeline and tokenizer configurations.

### Features

- **Reuses ChemTorch Configs**: Leverages existing data_pipeline and tokenizer configurations
- **Tokenization Statistics**: Token frequencies, length distributions, unique token counts
- **OOV Analysis**: Out-of-vocabulary analysis against existing vocabularies
- **Artifact Detection**: Detection of problematic tokens (whitespace, Unicode issues, malformed chemical notation)
- **Split Data Support**: Handle both single datasets and pre-split data automatically

### Usage

#### Basic Usage

```bash
# Run with default configuration (uses USPTO-1K data pipeline)
python scripts/check_vocab.py

# Use different dataset by changing data pipeline
python scripts/check_vocab.py data_pipeline=rdb7_fwd
python scripts/check_vocab.py data_pipeline=rgd1
python scripts/check_vocab.py data_pipeline=sn2

# Limit analysis and change reporting
python scripts/check_vocab.py max_samples=1000 report_top_k=50
```

### Configuration Parameters

- `data_pipeline`: Which data pipeline config to use
- `tokenizer`: Which tokenizer config to use
- `max_samples`: Limit number of samples (null = all)
- `report_top_k`: Number of top tokens to report
- `min_frequency`: Minimum frequency threshold
- `detect_artifacts`: Enable artifact detection
- `unicode_analysis`: Enable Unicode category analysis
- `whitespace_analysis`: Enable whitespace detection
- `vocab_path`: Path to existing vocabulary for OOV analysis
- `display_bad_tokens`: Displays a summary of bad tokens causing OOV issues or artifacts
- `seed`: Random seed for reproducibility

### Output

The script outputs a comprehensive summary to the console including:

- Total and unique token counts
- Average and max token lengths
- OOV rate (if vocabulary provided)
- Detected artifacts by category
- Top frequent tokens
- Token length distribution
- Unicode category summary

### Data Requirements

The script expects the data pipeline to produce data with a standard `smiles` column containing SMILES strings. This is the convention used throughout ChemTorch.
