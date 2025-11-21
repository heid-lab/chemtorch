# Utility scripts

This directory hosts helper scripts used during development, troubleshooting, and reproducibility workflows.

## Top-level helpers

- `wandb_to_hydra.py` — convert a Weights & Biases run into a Hydra config snippet (examples: https://heid-lab.github.io/chemtorch/user_guide/reproducibility.html)
- `collect_env.py` — capture hardware and Python package metadata for baselines (guidance: https://heid-lab.github.io/chemtorch/advanced_guide/integration_tests.html)

## PyG wheel helper — `install_pyg_deps.py`

Inspect the active `torch` install, derive the CUDA tag, and construct the correct PyG wheel index URL.

```bash
# Show the recommended pip command without executing it
python scripts/install_pyg_deps.py

# Install the detected wheel set immediately
python scripts/install_pyg_deps.py --run  # or: -y
```

If the pip command fails because no wheel exists for your platform, follow the official fallback steps: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels

## Vocabulary analysis — `check_vocab.py`

Provides detailed tokenizer and vocabulary diagnostics using the existing ChemTorch data pipelines and tokenizers.

**Features**

- Reuses configured data pipelines and tokenizers from ChemTorch
- Token frequency statistics and length distributions
- Out-of-vocabulary analysis versus stored vocabularies
- Artifact detection (whitespace, Unicode, malformed chemistry tokens)
- Works with single datasets or pre-split datasets

**Basic usage**

```bash
# Default configuration (USPTO-1K pipeline)
python scripts/check_vocab.py

# Switch datasets by overriding the data pipeline
python scripts/check_vocab.py data_pipeline=rdb7_fwd
python scripts/check_vocab.py data_pipeline=rgd1
python scripts/check_vocab.py data_pipeline=sn2

# Limit samples and adjust reporting depth
python scripts/check_vocab.py max_samples=1000 report_top_k=50
```

**Key parameters**

- `data_pipeline`: select the Hydra data pipeline config
- `tokenizer`: override the tokenizer config
- `max_samples`: cap the sample count (default: all)
- `report_top_k`: number of tokens to display in summaries
- `min_frequency`: minimum token frequency when reporting
- `detect_artifacts`: enable artifact detection checks
- `unicode_analysis`: include Unicode category breakdown
- `whitespace_analysis`: flag whitespace tokens
- `vocab_path`: provide a reference vocabulary for OOV analysis
- `display_bad_tokens`: print problematic tokens or OOV causes
- `seed`: random seed for reproducibility when sampling

**Output highlights**

- Total and unique token counts
- Average and maximum token lengths
- OOV rate when a vocabulary is supplied
- Artifact breakdown by category
- Top frequent tokens and length distributions
- Unicode category statistics (when enabled)

**Data requirements**

The configured pipeline must expose a `smiles` column; this is the convention used across ChemTorch datasets.

## Hydra target resolver — `resolve_target.py`

Jump from a Hydra YAML `_target_` entry to the corresponding Python definition.

**VS Code task example**

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

**Keybinding example**

```json
{
  "key": "ctrl+f12",
  "command": "workbench.action.tasks.runTask",
  "args": "Hydra: Go to _target_ at cursor",
  "when": "editorTextFocus && editorLangId == yaml"
}
```

**Usage**

1. Open a Hydra YAML config in VS Code.
2. Place the cursor on the `_target_:` line.
3. Trigger the task (for example `Ctrl+F12`) to open the Python definition.

**Shortcuts**

- Resolve the first `_target_` in a file: `python scripts/resolve_target.py --top-target path/to/config.yaml`
- Resolve by fully qualified name: `python scripts/resolve_target.py package.module.ClassName`

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
