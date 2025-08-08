import re
from typing import List

SMILES_ATOM_WISE_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
REACTION_SEPARATOR_TOKEN = ">>"
MOLECULE_SEPARATOR_TOKEN = "."
DEFAULT_UNK_TOKEN = "<UNK>"


class MoleculeRegexTokenizer:
    """
    Tokenizes a single molecule SMILES string using a regex pattern.
    """

    def __init__(self, regex_pattern: str = SMILES_ATOM_WISE_PATTERN):
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


class SimpleTokenizer:
    """
    Tokenizes a reaction SMILES string (e.g., "R1.R2>>P1.P2").
    """

    def __init__(
        self,
        unk_token: str = DEFAULT_UNK_TOKEN,
        molecule_tokenizer_pattern: str = SMILES_ATOM_WISE_PATTERN,
    ):
        self.unk_token = unk_token
        self._molecule_tokenizer = MoleculeRegexTokenizer(
            regex_pattern=molecule_tokenizer_pattern
        )

    def _tokenize_side(self, side_smiles: str) -> List[str]:
        """
        Tokenizes one side of a reaction (reactants or products).
        Example: "mol1.mol2.mol3"
        """
        if not side_smiles:
            return []

        side_tokens: List[str] = []
        molecule_smiles_list = side_smiles.split(MOLECULE_SEPARATOR_TOKEN)

        for i, mol_smiles in enumerate(molecule_smiles_list):
            if mol_smiles:
                molecule_tokens = self._molecule_tokenizer.tokenize(mol_smiles)
                side_tokens.extend(molecule_tokens)

            if i < len(molecule_smiles_list) - 1:
                side_tokens.append(MOLECULE_SEPARATOR_TOKEN)

        return side_tokens

    def tokenize(self, reaction_smiles: str) -> List[str]:
        """
        Tokenizes a full reaction SMILES string.

        Args:
            reaction_smiles: The reaction SMILES string (e.g., "R1.R2>>P1.P2").

        Returns:
            A list of tokens representing the reaction.
        """
        if not reaction_smiles:
            return []

        all_tokens: List[str] = []

        parts = reaction_smiles.split(REACTION_SEPARATOR_TOKEN, 1)

        reactants_smiles = parts[0]
        all_tokens.extend(self._tokenize_side(reactants_smiles))

        if len(parts) > 1:
            all_tokens.append(REACTION_SEPARATOR_TOKEN)
            products_smiles = parts[1]
            all_tokens.extend(self._tokenize_side(products_smiles))

        return all_tokens
