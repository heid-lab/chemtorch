import re
from typing import List

from chemtorch.components.preprocessing.tokenizer.abstract_tokenizer import AbstractTokenizer
from chemtorch.components.preprocessing.tokenizer.molecule_tokenizer import MoleculeRegexTokenizer
from chemtorch.components.preprocessing.tokenizer.tokenizer_defaults import DEFAULT_UNK_TOKEN, MOLECULE_SEPARATOR_TOKEN, REACTION_SEPARATOR_TOKEN, DEFAULT_MOLECULE_PATTERN


class ReactionTokenizer(AbstractTokenizer):
    """
    Tokenizes a reaction SMILES string (e.g., "R1.R2>>P1.P2").
    """

    def __init__(
        self,
        unk_token: str = DEFAULT_UNK_TOKEN,
        molecule_tokenizer_pattern: str = DEFAULT_MOLECULE_PATTERN,
    ):
        self.unk_token = unk_token
        self._molecule_tokenizer = MoleculeRegexTokenizer(regex_pattern=molecule_tokenizer_pattern)

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

            if i < len(molecule_smiles_list):
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
