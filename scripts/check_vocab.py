import hydra
from omegaconf import DictConfig, OmegaConf
from collections import Counter, defaultdict
import pandas as pd
import unicodedata
from tqdm import tqdm
from typing import List, Dict, Set, Optional, Union
import logging
import numpy as np
from dataclasses import dataclass

# Import ChemTorch components
from chemtorch.components.data_pipeline.simple_data_pipeline import SimpleDataPipeline
from chemtorch.components.preprocessing.tokenizer.abstract_tokenizer import AbstractTokenizer
from chemtorch.utils import DataSplit

log = logging.getLogger(__name__)

@dataclass
class TokenStats:
    """Statistics for a single token."""
    frequency: int
    length: int
    has_whitespace: bool
    has_unicode: bool
    unicode_categories: Set[str]
    examples: List[str]  # Example SMILES containing this token

@dataclass
class VocabAnalysis:
    """Complete vocabulary analysis results."""
    total_tokens: int
    unique_tokens: int
    token_stats: Dict[str, TokenStats]
    oov_tokens: Optional[Set[str]]
    oov_rate: Optional[float]
    artifacts: Dict[str, List[str]]
    length_distribution: Dict[int, int]
    unicode_categories: Dict[str, int]


def load_existing_vocab(vocab_path: str) -> Set[str]:
    """Load existing vocabulary from file."""
    vocab = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                vocab.add(token)
    log.info(f"Loaded {len(vocab)} tokens from {vocab_path}")
    return vocab


def extract_smiles_from_data(data: Union[pd.DataFrame, DataSplit]) -> List[str]:
    """Extract SMILES strings from processed data using standard 'smiles' column."""
    smiles_list = []
    
    def process_dataframe(df: pd.DataFrame, split_name: str = ""):
        nonlocal smiles_list
        log.info(f"Processing {len(df)} samples{' from ' + split_name if split_name else ''}")
        
        if 'smiles' not in df.columns:
            log.error(f"'smiles' column not found in dataframe. Available columns: {list(df.columns)}")
            return
        
        for _, row in df.iterrows():
            if pd.notna(row['smiles']):
                value = str(row['smiles']).strip()
                if value:
                    smiles_list.append(value)
    
    if isinstance(data, DataSplit):
        # Process each split separately
        for split_name in ['train', 'val', 'test']:
            split_data = getattr(data, split_name, None)
            if split_data is not None and isinstance(split_data, pd.DataFrame):
                process_dataframe(split_data, split_name)
    else:
        process_dataframe(data)
    
    # Filter out empty strings
    smiles_list = [s for s in smiles_list if s]
    log.info(f"Extracted {len(smiles_list)} SMILES strings")
    return smiles_list


def analyze_token_artifacts(token: str) -> Dict[str, bool]:
    """Analyze a token for various artifacts."""
    artifacts = {}
    
    # Whitespace analysis
    artifacts['has_leading_whitespace'] = token.startswith(' ')
    artifacts['has_trailing_whitespace'] = token.endswith(' ')
    artifacts['has_internal_whitespace'] = ' ' in token.strip()
    artifacts['has_tabs'] = '\t' in token
    artifacts['has_newlines'] = '\n' in token or '\r' in token
    
    # Unicode analysis
    artifacts['has_non_ascii'] = not token.isascii()
    artifacts['has_control_chars'] = any(unicodedata.category(c).startswith('C') for c in token)
    
    # Chemical notation artifacts
    artifacts['has_multiple_consecutive_dots'] = '..' in token
    artifacts['has_unmatched_brackets'] = token.count('[') != token.count(']')
    # Only flag unmatched parens if token contains BOTH types AND they're unmatched
    # Don't flag single '(' or ')' tokens as unmatched
    artifacts['has_unmatched_parens'] = (
        ('(' in token and ')' in token) and 
        (token.count('(') != token.count(')'))
    )
    
    # Length artifacts
    artifacts['is_very_long'] = len(token) > 50
    artifacts['is_empty'] = len(token) == 0
    
    return artifacts


def get_unicode_categories(token: str) -> Set[str]:
    """Get Unicode categories present in the token."""
    categories = set()
    for char in token:
        categories.add(unicodedata.category(char))
    return categories


def tokenize_and_analyze(smiles_list: List[str], 
                        tokenizer: AbstractTokenizer,
                        existing_vocab: Optional[Set[str]] = None) -> VocabAnalysis:
    """Tokenize SMILES and perform comprehensive analysis."""
    log.info("Starting tokenization and analysis...")
    
    token_counts = Counter()
    token_examples = defaultdict(list)
    all_artifacts = defaultdict(list)
    length_distribution = Counter()
    unicode_categories = Counter()
    
    total_tokens = 0
    failed_tokenizations = 0
    
    # Debug: track tokens that contain parentheses
    tokens_with_parens = []
    
    for smiles_idx, smiles in enumerate(tqdm(smiles_list, desc="Tokenizing SMILES", unit="smiles")):
        
        try:
            tokens = tokenizer.tokenize(smiles)
            if not tokens:
                failed_tokenizations += 1
                continue
                
            for token in tokens:
                token_counts[token] += 1
                total_tokens += 1
                
                # Debug: collect tokens containing parentheses (but aren't just single parens)
                if ('(' in token or ')' in token) and token not in ['(', ')'] and len(tokens_with_parens) < 20:
                    tokens_with_parens.append(f"Token: '{token}' from SMILES: '{smiles}'")
                
                # Store example (limit to avoid memory issues)
                if len(token_examples[token]) < 5:
                    token_examples[token].append(smiles)
                
                # Analyze artifacts
                artifacts = analyze_token_artifacts(token)
                for artifact_type, has_artifact in artifacts.items():
                    if artifact_type != "is_empty" and has_artifact:
                        all_artifacts[artifact_type].append(token)

                # Length distribution
                length_distribution[len(token)] += 1
                
                # Unicode categories
                categories = get_unicode_categories(token)
                for category in categories:
                    unicode_categories[category] += 1
                    
        except Exception as e:
            log.warning(f"Failed to tokenize SMILES {smiles_idx}: {smiles[:50]}... Error: {e}")
            failed_tokenizations += 1
            continue
    
    # Debug output
    if tokens_with_parens:
        log.info("Found tokens containing parentheses (other than standalone '(' or ')'):")
        for example in tokens_with_parens:
            log.info(f"  {example}")
    
    log.info(f"Tokenization complete. Failed: {failed_tokenizations}/{len(smiles_list)}")
    
    # Build token statistics
    token_stats = {}
    for token, count in token_counts.items():
        token_stats[token] = TokenStats(
            frequency=count,
            length=len(token),
            has_whitespace=any(c.isspace() for c in token),
            has_unicode=not token.isascii(),
            unicode_categories=get_unicode_categories(token),
            examples=token_examples[token][:3]  # Limit examples
        )
    
    # OOV analysis
    oov_tokens = None
    oov_rate = None
    if existing_vocab:
        oov_tokens = set(token_counts.keys()) - existing_vocab
        oov_rate = len(oov_tokens) / len(token_counts) if token_counts else 0
        log.info(f"OOV analysis: {len(oov_tokens)} out-of-vocabulary tokens ({oov_rate:.2%})")
    
    return VocabAnalysis(
        total_tokens=total_tokens,
        unique_tokens=len(token_counts),
        token_stats=token_stats,
        oov_tokens=oov_tokens,
        oov_rate=oov_rate,
        artifacts=dict(all_artifacts),
        length_distribution=dict(length_distribution),
        unicode_categories=dict(unicode_categories)
    )


def print_summary(analysis: VocabAnalysis, cfg: DictConfig):
    """Print analysis summary to console."""
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total tokens: {analysis.total_tokens:,}")
    print(f"Unique tokens: {analysis.unique_tokens:,}")
    if analysis.token_stats:
        print(f"Average token length: {np.mean([stats.length for stats in analysis.token_stats.values()]):.2f}")
        print(f"Max token length: {max([stats.length for stats in analysis.token_stats.values()])}")
    
    if analysis.oov_rate is not None:
        oov_count = len(analysis.oov_tokens or set())
        print(f"OOV rate: {analysis.oov_rate:.2%} ({oov_count} tokens)")
        
        # Display OOV tokens if requested
        if cfg.get('display_bad_tokens', False) and analysis.oov_tokens:
            oov_list = sorted(list(analysis.oov_tokens))
            oov_string = ', '.join(f"'{token}'" if token else "''" for token in oov_list)
            print(f"OOV tokens: {oov_string}")
    
    # Artifact summary
    bad_tokens_summary = {}
    if cfg.detect_artifacts and analysis.artifacts:
        print(f"\nArtifacts detected:")
        for artifact_type, tokens in analysis.artifacts.items():
            if tokens:
                unique_tokens = list(set(tokens))
                print(f"  {artifact_type}: {len(unique_tokens)} unique tokens")
                if cfg.get('display_bad_tokens', False):
                    bad_tokens_summary[artifact_type] = unique_tokens
    
    # Top frequent tokens
    print(f"\nTop {min(cfg.report_top_k, 20)} most frequent tokens:")
    sorted_tokens = sorted(analysis.token_stats.items(), key=lambda x: x[1].frequency, reverse=True)
    for i, (token, stats) in enumerate(sorted_tokens[:min(cfg.report_top_k, 20)]):
        # Escape special characters for display
        display_token = repr(token) if any(c.isspace() or not c.isprintable() for c in token) else token
        print(f"  {i+1:2d}. {display_token:<20} ({stats.frequency:,} occurrences)")
    
    # Length distribution summary
    if analysis.length_distribution:
        print(f"\nToken length distribution:")
        for length in sorted(analysis.length_distribution.keys())[:10]:  # Show first 10 lengths
            print(f"  Length {length}: {analysis.length_distribution[length]:,} tokens")
    
    # Unicode category summary
    if analysis.unicode_categories:
        print(f"\nUnicode categories found:")
        sorted_categories = sorted(analysis.unicode_categories.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:5]:  # Show top 5 categories
            print(f"  {category}: {count:,} characters")
    
    # Bad tokens summary (if enabled)
    if cfg.get('display_bad_tokens', False):
        print(f"\n" + "="*60)
        print("BAD TOKENS SUMMARY")
        print("="*60)
        
        # Show OOV tokens
        if analysis.oov_tokens:
            oov_list = sorted(list(analysis.oov_tokens))
            oov_string = ', '.join(f"'{token}'" if token else "''" for token in oov_list)
            print(f"\nOut-of-Vocabulary tokens ({len(oov_list)}):")
            print(f"  {oov_string}")
        
        # Show artifact tokens
        if cfg.detect_artifacts and analysis.artifacts and bad_tokens_summary:
            for artifact_type, tokens in bad_tokens_summary.items():
                if tokens:
                    # Limit to first 20 tokens to avoid overwhelming output
                    display_tokens = sorted(tokens)[:20]
                    tokens_string = ', '.join(f"'{token}'" if token else "''" for token in display_tokens)
                    suffix = f" (showing first 20 of {len(tokens)})" if len(tokens) > 20 else ""
                    print(f"\n{artifact_type.replace('_', ' ').title()} tokens ({len(tokens)}){suffix}:")
                    print(f"  {tokens_string}")


@hydra.main(version_base=None, config_path="../conf", config_name="vocab_check")
def main(cfg: DictConfig) -> None:
    """
    Analyze vocabulary and compute tokenization statistics for chemical data.
    """
    log.info("Starting vocabulary analysis...")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    seed = cfg.get('seed')
    np.random.seed(seed)
    
    # Load existing vocabulary if provided
    existing_vocab = None
    vocab_path = cfg.get('vocab_path')
    if vocab_path:
        existing_vocab = load_existing_vocab(vocab_path)
    
    # Initialize components
    log.info("Initializing components...")
    data_pipeline: SimpleDataPipeline = hydra.utils.instantiate(cfg.data_pipeline)
    tokenizer: AbstractTokenizer = hydra.utils.instantiate(cfg.tokenizer)
    
    # Load and process data through the pipeline
    log.info("Loading and processing data through pipeline...")
    data = data_pipeline()
    
    # Limit samples if specified
    max_samples = cfg.get('max_samples')
    if max_samples:
        if isinstance(data, DataSplit):
            # Limit each split by creating a new DataSplit object
            limited_splits = {}
            for split_name in ['train', 'val', 'test']:
                split_data = getattr(data, split_name, None)
                if split_data is not None and isinstance(split_data, pd.DataFrame):
                    limited_splits[split_name] = split_data.head(max_samples)
                else:
                    limited_splits[split_name] = split_data
            
            # Create new DataSplit with limited data
            data = DataSplit(
                train=limited_splits.get('train'),
                val=limited_splits.get('val'), 
                test=limited_splits.get('test')
            )
        else:
            data = data.head(max_samples)
        log.info(f"Limited to {max_samples} samples per split")
    
    # Extract SMILES strings
    smiles_list = extract_smiles_from_data(data)
    
    if not smiles_list:
        log.error("No SMILES strings found in the data!")
        return
    
    log.info(f"Extracted {len(smiles_list)} SMILES strings")
    
    # Tokenize and analyze
    analysis = tokenize_and_analyze(smiles_list, tokenizer, existing_vocab)
    
    # Print summary
    print_summary(analysis, cfg)
    
    log.info("Analysis complete.")


if __name__ == "__main__":
    main()