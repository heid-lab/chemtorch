from typing import Callable, List, TYPE_CHECKING
import torch

from chemtorch.components.dataset.abstact_dataset import AbstractDataset
from chemtorch.components.representation.token.abstract_token_representation import AbstractTokenRepresentation

def get_vocab_size(dataset: AbstractDataset[torch.Tensor, AbstractTokenRepresentation]) -> int:
    """
    Get vocabulary size from a dataset with a token representation.
    
    Args:
        dataset: Dataset that uses an AbstractTokenRepresentation
        
    Returns:
        Size of the vocabulary
    """
    return dataset.representation.vocab_size

def extend_vocab(dataset: AbstractDataset[torch.Tensor, AbstractTokenRepresentation], 
                 new_tokens: List[str]) -> None:
    """
    Extend vocabulary of a dataset with a token representation.
    
    Args:
        dataset: Dataset that uses an AbstractTokenRepresentation
        new_tokens: List of new tokens to add to vocabulary
    """
    dataset.representation.extend_vocab(new_tokens)

def save_vocab(dataset: 'AbstractDataset[torch.Tensor, AbstractTokenRepresentation]', 
               vocab_path: str) -> None:
    """
    Save vocabulary from a dataset with a token representation.
    
    Args:
        dataset: Dataset that uses an AbstractTokenRepresentation
        vocab_path: Path where to save the vocabulary file
    """
    dataset.representation.save_vocab(vocab_path)

def extend_vocab_from_strings(dataset: AbstractDataset[torch.Tensor, AbstractTokenRepresentation], 
                             strings: List[str]) -> None:
    """
    Extend vocabulary from a list of strings.
    
    Args:
        dataset: Dataset that uses an AbstractTokenRepresentation
        strings: List of strings to tokenize and add to vocabulary
        
    Returns:
        Number of new tokens added
    """
    new_tokens = set()
    
    for string in strings:
        try:
            tokens = dataset.representation.tokenize(string)
            for token in tokens:
                if token not in dataset.representation.word2id:
                    new_tokens.add(token)
        except Exception:
            # Skip strings that can't be tokenized
            continue
    
    if new_tokens:
        extend_vocab(dataset, list(new_tokens))

def extend_vocab_from_data(dataset: AbstractDataset[torch.Tensor, AbstractTokenRepresentation], 
                          column_name: str = 'smiles') -> None:
    """
    Extend vocabulary by analyzing data in a specific DataFrame column.
    
    Args:
        dataset: Dataset that uses an AbstractTokenRepresentation
        column_name: Name of the DataFrame column to analyze
        
    Returns:
        Number of new tokens added
        
    Raises:
        ValueError: If the specified column doesn't exist in the dataset
    """
    if not hasattr(dataset, 'dataframe'):
        raise ValueError("Dataset must have a 'dataframe' attribute")
    
    if column_name not in dataset.dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in dataset")
    
    strings = dataset.dataframe[column_name].dropna().tolist()
    strings = [s for s in strings if isinstance(s, str)]
    
    extend_vocab_from_strings(dataset, strings)

def extend_vocab_with_extractor(
    dataset: AbstractDataset[torch.Tensor, AbstractTokenRepresentation], 
    extractor: Callable[[AbstractDataset[torch.Tensor, AbstractTokenRepresentation]], List[str]]
) -> None:
    """
    Extend vocabulary using a custom string extraction function.
    
    This is the most flexible version - you provide a function that knows
    how to extract strings from your specific dataset structure.
    
    Args:
        dataset: Dataset that uses an AbstractTokenRepresentation
        extractor: Function that takes a dataset and returns a list of strings
        
    Returns:
        Number of new tokens added
        
    Example:
        >>> def my_extractor(ds):
        ...     return ds.dataframe['smiles'].tolist()
        >>> extend_vocab_from_data_flexible(dataset, my_extractor)
    """
    strings = extractor(dataset)
    extend_vocab_from_strings(dataset, strings)