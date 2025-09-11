
import re
from typing import List
from chemtorch.components.preprocessing.tokenizer.abstract_tokenizer import AbstractTokenizer
from chemtorch.components.preprocessing.tokenizer.tokenizer_defaults import DEFAULT_UNK_TOKEN


class RegexTokenizer(AbstractTokenizer):
    """
    Tokenizes a SMILES string using a regex pattern.
    """

    def __init__(self, regex_pattern: str):
        """
        Args:
            regex_pattern: The regex pattern to use for tokenization.
        """
        self.regex = re.compile(regex_pattern)

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenizes a SMILES string.

        Args:
            smiles: The SMILES string to tokenize.

        Returns:
            A list of tokens.
        """
        if not smiles:
            return []

        tokens = self.regex.findall(smiles)
        if not tokens and smiles:
            return [DEFAULT_UNK_TOKEN]

        return tokens