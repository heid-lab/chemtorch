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
from chemtorch.components.representation.token.tokenizer.abstract_tokenizer import AbstractTokenizer
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


def analyze_problematic_smiles(smiles: str, tokenizer: AbstractTokenizer) -> Dict[str, Union[str, int, bool, List]]:
    """Analyze a SMILES string that might cause issues."""
    from rdkit import Chem
    
    analysis = {
        'smiles': smiles,
        'length': len(smiles),
        'has_explicit_hydrogens': '[H]' in smiles,
        'has_aromatic_lowercase': any(c.islower() and c.isalpha() for c in smiles),
        'unusual_chars': [c for c in smiles if c not in 'CNOSPFClBrI()[]=#-+123456789@/%\\.:'],
        'rdkit_parseable': False,
        'tokenizer_success': False
    }
    
    # Test RDKit parsing
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            analysis['rdkit_parseable'] = True
            analysis['num_atoms'] = mol.GetNumAtoms()
            analysis['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
    except Exception as e:
        analysis['rdkit_error'] = str(e)
    
    # Test tokenizer
    try:
        tokens = tokenizer.tokenize(smiles)
        if tokens:
            analysis['tokenizer_success'] = True
            analysis['num_tokens'] = len(tokens)
    except Exception as e:
        analysis['tokenizer_error'] = str(e)
    
    return analysis


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
    log.info("Press Ctrl+C at any time to stop and see analysis of processed data")
    
    token_counts = Counter()
    token_examples = defaultdict(list)
    all_artifacts = defaultdict(list)
    length_distribution = Counter()
    unicode_categories = Counter()
    
    total_tokens = 0
    failed_tokenizations = 0
    rdkit_warnings_count = 0
    processed_smiles = 0
    
    # Debug: track tokens that contain parentheses
    tokens_with_parens = []
    problematic_smiles_details = []
    warning_patterns = [
        "not removing hydrogen atom without neighbors",
        "WARNING",
        "ERROR"
    ]
    
    try:
        for smiles_idx, smiles in enumerate(tqdm(smiles_list, desc="Tokenizing SMILES", unit="smiles")):
            processed_smiles = smiles_idx + 1
            
            try:
                # Simple tokenization without trying to capture warnings
                # We'll rely on the user seeing warnings in the terminal output
                tokens = tokenizer.tokenize(smiles)
                
                # For debugging purposes, let's test individual SMILES that might cause issues
                # by checking for patterns that typically cause RDKit warnings
                likely_problematic = (
                    '[H]' in smiles or  # Explicit hydrogens
                    smiles.count('[') != smiles.count(']') or  # Unmatched brackets
                    '..' in smiles or  # Double dots
                    any(c in smiles for c in ['@', '%'])  # Complex stereochemistry
                )
                
                if likely_problematic and len(problematic_smiles_details) < 20:
                    # Collect details for potential problem cases
                    detailed_analysis = analyze_problematic_smiles(smiles, tokenizer)
                    if detailed_analysis.get('warning_details'):
                        rdkit_warnings_count += 1
                        problematic_smiles_details.append(detailed_analysis)
                
                if not tokens:
                    failed_tokenizations += 1
                    continue
                    
                for token in tokens:
                    token_counts[token] += 1
                    total_tokens += 1
                    
                    # Debug: collect tokens with unmatched parentheses
                    if len(tokens_with_parens) < 20:
                        has_unmatched_parens = (
                            ('(' in token and ')' in token) and 
                            (token.count('(') != token.count(')'))
                        ) or (
                            ('(' in token and ')' not in token and token != '(') or
                            (')' in token and '(' not in token and token != ')')
                        )
                        if has_unmatched_parens:
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
                
    except KeyboardInterrupt:
        log.info(f"\nKeyboard interrupt received. Processed {processed_smiles}/{len(smiles_list)} SMILES so far.")
        log.info("Analyzing results from processed data...")
    
    # Debug output
    if tokens_with_parens:
        log.info("Found tokens with unmatched parentheses:")
        for example in tokens_with_parens:
            log.info(f"  {example}")
    
    # Report problematic SMILES
    if problematic_smiles_details:
        log.info(f"Analyzed {len(problematic_smiles_details)} potentially problematic SMILES:")
        for i, analysis in enumerate(problematic_smiles_details[:5], 1):  # Show first 5
            log.info(f"  Example {i}:")
            log.info(f"    SMILES: {analysis['smiles'][:100]}{'...' if len(analysis['smiles']) > 100 else ''}")
            log.info(f"    Length: {analysis['length']}")
            log.info(f"    RDKit parseable: {analysis['rdkit_parseable']}")
            log.info(f"    Has explicit H: {analysis['has_explicit_hydrogens']}")
            log.info(f"    Unusual chars: {analysis['unusual_chars']}")
    
    log.info(f"Tokenization complete. Processed: {processed_smiles}/{len(smiles_list)}, Failed: {failed_tokenizations}")
    
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


def save_vocabulary(vocab: Set[str], path: str, sort_by_frequency: bool = False, 
                   token_stats: Optional[Dict[str, TokenStats]] = None) -> None:
    """Save vocabulary to file, optionally sorted by frequency."""
    log.info(f"Saving vocabulary with {len(vocab)} tokens to {path}")
    
    if sort_by_frequency and token_stats:
        # Sort by frequency (descending), then alphabetically
        sorted_tokens = sorted(vocab, 
                             key=lambda x: (-token_stats.get(x, TokenStats(0, 0, False, False, set(), [])).frequency, x))
    else:
        # Sort alphabetically
        sorted_tokens = sorted(vocab)
    
    with open(path, 'w', encoding='utf-8') as f:
        for token in sorted_tokens:
            f.write(f"{token}\n")
    
    log.info(f"Vocabulary saved to {path}")


def extend_vocabulary(existing_vocab: Set[str], new_tokens: Set[str], 
                     save_path: Optional[str] = None, 
                     original_vocab_path: Optional[str] = None,
                     token_stats: Optional[Dict[str, TokenStats]] = None) -> Set[str]:
    """
    Extend existing vocabulary with new tokens.
    
    Args:
        existing_vocab: Set of existing vocabulary tokens
        new_tokens: Set of new tokens to add
        save_path: Path to save extended vocab (if None, saves in-place)
        original_vocab_path: Original vocabulary file path (for in-place extension)
        token_stats: Token statistics for frequency-based sorting
    
    Returns:
        Extended vocabulary set
    """
    original_size = len(existing_vocab)
    extended_vocab = existing_vocab.union(new_tokens)
    new_size = len(extended_vocab)
    added_tokens = new_size - original_size
    
    log.info(f"Extended vocabulary: {original_size} → {new_size} tokens (+{added_tokens} new tokens)")
    
    if added_tokens > 0:
        # Determine save path
        if save_path is None:
            if original_vocab_path is None:
                log.warning("No save path specified and no original vocab path available. Cannot save extended vocabulary.")
                return extended_vocab
            save_path = original_vocab_path
            log.info(f"Extending vocabulary in-place: {save_path}")
        else:
            log.info(f"Saving extended vocabulary to new path: {save_path}")
        
        # Save extended vocabulary
        save_vocabulary(extended_vocab, save_path, sort_by_frequency=True, token_stats=token_stats)
    else:
        log.info("No new tokens found. Vocabulary unchanged.")
    
    return extended_vocab


def print_summary(analysis: VocabAnalysis, cfg: DictConfig):
    """Print analysis summary to console."""
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS SUMMARY")
    print("="*60)
    
    # Show vocabulary operation mode
    extend_vocab = cfg.get('extend_vocab', False)
    vocab_path = cfg.get('vocab_path')
    save_vocab_path = cfg.get('save_vocab_path')
    
    if extend_vocab:
        if vocab_path:
            print("Mode: Extending existing vocabulary")
        else:
            print("Mode: Creating new vocabulary from data")
    elif vocab_path:
        print("Mode: Analyzing with existing vocabulary")
    else:
        print("Mode: Analysis only (no vocabulary operations)")
    
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
    
    # Show vocabulary save information
    if extend_vocab:
        if save_vocab_path:
            print(f"Vocabulary saved to: {save_vocab_path}")
        elif vocab_path:
            print(f"Vocabulary saved to: {vocab_path} (in-place)")
    
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
    try:
        log.info("Starting vocabulary analysis...")
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        
        # Handle RDKit warning suppression
        if cfg.get('suppress_rdkit_warnings', False):
            log.info("Suppressing RDKit warnings globally")
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')  # type: ignore
        
        # Set random seed
        seed = cfg.get('seed')
        np.random.seed(seed)
        
        # Handle vocabulary loading
        existing_vocab = None
        vocab_path = cfg.get('vocab_path')
        
        if vocab_path:
            existing_vocab = load_existing_vocab(vocab_path)
        else:
            log.info("No vocabulary path specified. Will perform analysis without existing vocabulary.")
            existing_vocab = set()
        
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
        
        # Handle vocabulary extension
        extend_vocab = cfg.get('extend_vocab', False)
        save_vocab_path = cfg.get('save_vocab_path')
        
        if extend_vocab:
            # Get all tokens found in the data
            found_tokens = set(analysis.token_stats.keys())
            
            if existing_vocab is not None:
                # Extend existing vocabulary
                final_vocab = extend_vocabulary(
                    existing_vocab=existing_vocab,
                    new_tokens=found_tokens,
                    save_path=save_vocab_path,
                    original_vocab_path=vocab_path,
                    token_stats=analysis.token_stats
                )
            else:
                # No existing vocab, so create new vocabulary from found tokens
                log.info(f"Creating new vocabulary from scratch with {len(found_tokens)} tokens")
                final_vocab = found_tokens
                
                # Save the new vocabulary if path is specified
                if save_vocab_path:
                    save_vocabulary(final_vocab, save_vocab_path, 
                                  sort_by_frequency=True, token_stats=analysis.token_stats)
                else:
                    log.warning("No save path specified for new vocabulary. Vocabulary not saved.")
        
        # Print summary
        print_summary(analysis, cfg)
        
        log.info("Analysis complete.")
        
    except KeyboardInterrupt:
        log.info("\n" + "="*60)
        log.info("ANALYSIS INTERRUPTED BY USER")
        log.info("="*60)
        log.info("The analysis was stopped early, but you can see the results from processed data above.")
        log.info("To see the full summary, let the analysis complete or use max_samples to limit the dataset size.")
        
    except Exception as e:
        log.error(f"Unexpected error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()