
from typing import List
from rdkit import Chem
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit.Chem.rdmolfiles import MolFragmentToSmiles

from chemtorch.components.preprocessing.tokenizer.molecule_tokenizer.molecule_tokenizer_base import MoleculeTokenizerBase


class SubstructureTokenizer(MoleculeTokenizerBase):
    """
    Tokenizes a SMILES string into substructure tokens based on atom environments of specified radii.

    Inspired by original implementation at https://github.com/BellaCaoyh/hirxn/tree/main#.
    
    Paper: https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c01787
    """

    def __init__(self, radii: List[int], vocab_path: str, unk_token: str, pad_token: str):
        """
        Initialize SubstructureTokenizer.
        
        Args:
            radii: List of radii for atom environments
        """
        super().__init__(vocab_path=vocab_path, unk_token=unk_token, pad_token=pad_token)
        self.radii = radii

    def tokenize(self, smiles: str) -> List[str]:
        token_frequencies_by_r = {r: {} for r in self.radii}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        Chem.SanitizeMol(mol)
        Chem.RemoveHs(mol)  # this might cause warnings for some molecules

        for radius in self.radii:
            #####################################################################################################
            # adapted from https://github.com/BellaCaoyh/hirxn/blob/main/HiRXN/model/rxntokenizer.py lines 177-198
            token_frequencies = dict()

            for i in range(len(mol.GetAtoms())):

                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, i)
                env_atoms=set()

                for bidx in env:
                    env_atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
                    env_atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())

                # If there are no other atoms in the environment (e.g. O in H2O) just use the atom itself
                if len(env_atoms) == 0:
                    env_atoms = {i}

                token = Chem.MolFragmentToSmiles(mol, atomsToUse=list(env_atoms), bondsToUse=env,canonical=True)

                token_frequencies[token] = token_frequencies.get(token, 0) + 1
            #####################################################################################################
            token_frequencies_by_r[radius] = token_frequencies

        # In the original implementation, they return a list of all unique tokens across all radii
        all_tokens = []
        for r in self.radii:
            all_tokens.extend(token_frequencies_by_r[r].keys())
        unique_tokens = list(set(all_tokens))

        return unique_tokens
