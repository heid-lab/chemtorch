# Utility Scripts

Helper utilities that make it easier to debug, reproduce, and extend ChemTorch experiments. Each script is self-contained, and some include deeper walk-throughs in the web docs (see links in the quick reference table when available).

## Quick Reference

| Script | Core purpose |
| --- | --- |
| `install_pyg_deps.py` | Recommend (or install) the matching PyTorch Geometric wheels |
| `generate_data_split_configs.py` | Build OOD benchmark configs for every model/split combo |
| `check_vocab.py` | Inspect token vocabularies, OOV stats, and tokenizer artifacts |
| `resolve_target.py` | Jump from a Hydra `_target_` string to its Python definition |
| `collect_env.py` | Snapshot hardware/CUDA/Python package info for reproducibility (see [docs](https://heid-lab.github.io/chemtorch/advanced_guide/integration_tests.html)) |
| `wandb_to_hydra.py` | Turn a Weights & Biases run into Hydra overrides (see [docs](https://heid-lab.github.io/chemtorch/user_guide/reproducibility.html) for examples) |

## Detailed Guides

### PyG wheel helper — `install_pyg_deps.py`

Inspect the active `torch` install, derive the CUDA tag, and construct the correct PyG wheel index URL.

```bash
# Show the recommended pip command without executing it
python scripts/install_pyg_deps.py

# Install the detected wheel set immediately
python scripts/install_pyg_deps.py --run  # or: -y
```

If the pip command fails because no wheel exists for your platform, follow the official fallback steps: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels

---

### Data split config generator — `generate_data_split_configs.py`

Automates the creation of OOD benchmark configs by cloning a "source" config per model (typically the tuned/baseline config) and stamping in all supported data splitters.

#### Purpose

- Standardize benchmarking across random, scaffold, reaction-core, size, and target ordered splits.
- Keep project-wide conventions (group name, prediction folders, run names) in sync without editing YAML manually.

#### Configuration fields

Configure the script by editing the constants at the top of `generate_data_split_configs.py`:

- `SOURCE_MODEL_CONFIG_DIR` – folder containing the per-model source configs (`<model>.yaml`).
- `OOD_CONFIG_OUTPUT_DIR` – destination directory where split configs will be written (`<model>/<split>.yaml`).
- `PREDICTION_BASE` – base path used when stamping `predictions_save_dir`.
- `CHECKPOINT_BASE` – base path injected into `trainer.checkpoint_callback.dirpath`.
- `GROUP_NAME` – value assigned to `group_name` in every generated config.
- `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` – shared split fractions applied to every splitter definition.

#### Usage

Run the script, optionally filtering by models or splits:

```bash
python scripts/generate_data_split_configs.py              # build every model/split
python scripts/generate_data_split_configs.py --models atom_han drfp_mlp
python scripts/generate_data_split_configs.py --splits random_split size_split_desc
python scripts/generate_data_split_configs.py --models dimereaction --overwrite
```

#### Structural assumptions

- Source configs are named `<model>.yaml` (for example `atom_han.yaml`) and located inside `SOURCE_MODEL_CONFIG_DIR`.
- Each config already contains a valid Hydra `data_module` block; the script only swaps `data_splitter`.
- Output configs are written under `<OOD_CONFIG_OUTPUT_DIR>/<model>/`.
- Prediction and checkpoint paths follow `<PREDICTION_BASE|CHECKPOINT_BASE>/<model>/<split_prefix>/seed_...` and mirror the same structure for checkpoints.

#### Outcome

- For every requested split, a YAML file such as `size_split_desc.yaml` is created with updated `group_name`, `run_name`, `data_module.data_splitter`, predictions directory, and checkpoint directory.
- Existing files are preserved unless `--overwrite` is passed (or the path is removed manually).

#### Extending to new splitters

- Add a new entry to the `SPLIT_SPECS` dictionary inside `generate_data_split_configs.py`. Each entry defines:
  - `file_name`: output YAML name (for example, `index_split.yaml`).
  - `prefix`: label used for `run_name`, prediction paths, and checkpoint paths.
  - `splitter`: the Hydra config snippet that should replace `data_module.data_splitter`.
- Keep the `prefix` unique so run names and directories stay disambiguated.
- When splitters need extra parameters, add them to the `splitter` mapping; the script deep-copies it into each generated config.

---

### Vocabulary analysis — `check_vocab.py`

Provides detailed tokenizer and vocabulary diagnostics using the existing ChemTorch data pipelines and tokenizers.

#### Features

- Reuses configured data pipelines and tokenizers from ChemTorch
- Token frequency statistics and length distributions
- Out-of-vocabulary analysis versus stored vocabularies
- Artifact detection (whitespace, Unicode, malformed chemistry tokens)
- Works with single datasets or pre-split datasets

#### Basic usage

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

#### Key parameters

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

#### Output

- Total and unique token counts
- Average and maximum token lengths
- OOV rate when a vocabulary is supplied
- Artifact breakdown by category
- Top frequent tokens and length distributions
- Unicode category statistics (when enabled)

#### Data requirements

The configured pipeline must expose a `smiles` column; this is the convention used across ChemTorch datasets.

---

### Hydra target resolver — `resolve_target.py`

Jump from a Hydra YAML `_target_` entry to the corresponding Python definition.

#### VS Code task example

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

#### Keybinding example

```json
{
  "key": "ctrl+f12",
  "command": "workbench.action.tasks.runTask",
  "args": "Hydra: Go to _target_ at cursor",
  "when": "editorTextFocus && editorLangId == yaml"
}
```

#### Usage

1. Open a Hydra YAML config in VS Code.
2. Place the cursor on the `_target_:` line.
3. Trigger the task (for example `Ctrl+F12`) to open the Python definition.

#### Shortcuts

- Resolve the first `_target_` in a file: `python scripts/resolve_target.py --top-target path/to/config.yaml`
- Resolve by fully qualified name: `python scripts/resolve_target.py package.module.ClassName`