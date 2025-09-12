
import re
from typing import List
from chemtorch.components.representation.token.tokenizer.molecule_tokenizer.molecule_tokenizer_base import MoleculeTokenizerBase


class RegexTokenizer(MoleculeTokenizerBase):
    """
    Tokenizes a SMILES string using a regex pattern.
    """

    def __init__(self, regex_pattern: str, vocab_path: str, unk_token: str, pad_token: str):
        """
        Args:
            regex_pattern: The regex pattern to use for tokenization.
        """
        super().__init__(vocab_path=vocab_path, unk_token=unk_token, pad_token=pad_token)
        self.regex = re.compile(regex_pattern)

    @property
    def vocab_path(self) -> str:
        """Path to the vocabulary file."""
        return self._vocab_path

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenizes a SMILES string.

        Args:
            smiles: The SMILES string to tokenize.

        Returns:
            A list of tokens.
        """
        return self.regex.findall(smiles)
    