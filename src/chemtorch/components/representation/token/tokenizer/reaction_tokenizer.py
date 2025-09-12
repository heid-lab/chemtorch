from typing import List

try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.representation.token.tokenizer.abstract_tokenizer import AbstractTokenizer
from chemtorch.utils.atom_mapping import remove_atom_mapping


class ReactionTokenizer(AbstractTokenizer):
    """
    Tokenizes a reaction SMILES string (e.g., "R1.R2>>P1.P2") using a specified molecule tokenizer for individual molecules.
    """

    def __init__(
        self,
        molecule_tokenizer: AbstractTokenizer,
    ):
        """
        Args:
            molecule_tokenizer (AbstractTokenizer): Tokenizer for individual molecules.
            unk_token (str): Token to use for unknown tokens.
        """
        self.molecule_tokenizer = molecule_tokenizer

    @property
    @override
    def vocab_path(self) -> str:
        """Path to the vocabulary file."""
        return self.molecule_tokenizer.vocab_path

    @property
    @override
    def unk_token(self) -> str:
        """Token to use for unknown tokens."""
        return self.molecule_tokenizer.unk_token

    @property
    @override
    def pad_token(self) -> str:
        """Token to use for padding."""
        return self.molecule_tokenizer.pad_token

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenizes a full reaction SMILES string.

        Args:
            smiles: The reaction SMILES string (e.g., "R1.R2>>P1.P2").

        Returns:
            A list of tokens representing the reaction.
        """
        # remove atom map numbers
        smiles = remove_atom_mapping(smiles)
        
        # convert reactant>agent>product format to reactant.agent>>product format
        if smiles.count('>') == 2 and '>>' not in smiles:
            parts = smiles.split('>')
            if len(parts) == 3:
                reactant, agent, product = parts
                smiles = f"{reactant}.{agent}>>{product}"

        all_tokens: List[str] = []

        parts = smiles.split(">>", 1)

        if len(parts) < 2:
            raise ValueError(f"Invalid reaction SMILES: '{smiles}'. Must contain '>>' separator.")

        reactants_smiles = parts[0]
        all_tokens.extend(self._tokenize_side(reactants_smiles))
        all_tokens.append(">>")
        products_smiles = parts[1]
        all_tokens.extend(self._tokenize_side(products_smiles))

        return all_tokens


    def _tokenize_side(self, side_smiles: str) -> List[str]:
        """
        Tokenizes one side of a reaction (reactants or products).
        Example: "mol1.mol2.mol3"
        """
        if not side_smiles:
            return []

        side_tokens: List[str] = []
        molecule_smiles_list = side_smiles.split(".")

        for i, mol_smiles in enumerate(molecule_smiles_list):
            if mol_smiles:
                molecule_tokens = self.molecule_tokenizer.tokenize(mol_smiles)
                side_tokens.extend(molecule_tokens)

            if i < len(molecule_smiles_list):
                side_tokens.append(".")

        return side_tokens
