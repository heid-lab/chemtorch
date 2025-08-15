
import re
from typing import List
from chemtorch.components.preprocessing.tokenizer.abstract_tokenizer import AbstractTokenizer
from chemtorch.components.preprocessing.tokenizer.tokenizer_defaults import DEFAULT_UNK_TOKEN, DEFAULT_MOLECULE_PATTERN


class MoleculeRegexTokenizer(AbstractTokenizer):
    """
    Tokenizes a single molecule SMILES string using a regex pattern.
    """

    def __init__(self, regex_pattern: str = DEFAULT_MOLECULE_PATTERN):
        """
        Args:
            regex_pattern: The regex pattern to use for tokenization.
        """
        self.regex = re.compile(regex_pattern)

    def tokenize(self, molecule_smiles: str) -> List[str]:
        """
        Tokenizes a molecule SMILES string.

        Args:
            molecule_smiles: The molecule SMILES string to tokenize.

        Returns:
            A list of tokens.
        """
        if not molecule_smiles:
            return []

        tokens = self.regex.findall(molecule_smiles)
        if not tokens and molecule_smiles:
            return [DEFAULT_UNK_TOKEN]

        return tokens