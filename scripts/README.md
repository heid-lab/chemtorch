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
