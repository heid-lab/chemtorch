# started from code from https://github.com/reymond-group/drfp/blob/main/src/drfp/fingerprint.py, MIT License, Copyright (c) 2021 Daniel Probst

from collections import defaultdict
from hashlib import blake2b
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import torch
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from chemtorch.representation import AbstractRepresentation

RDLogger.DisableLog("rdApp.*")


class DRFP(AbstractRepresentation[torch.Tensor]):
    """
    Stateless class for constructing DRFP (Differential Reaction Fingerprints).

    This class provides a `forward()` method that takes a reaction SMILES string
    and returns a PyTorch Tensor representing the folded DRFP fingerprint.
    It utilizes the DrfpEncoder class for the core fingerprint generation logic.
    """

    def __init__(
        self,
        n_folded_length: int = 2048,
        min_radius: int = 0,
        radius: int = 3,
        rings: bool = True,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
    ):
        """
        Initializes the DRFP representation creator.

        Args:
            n_folded_length (int, optional): The length of the folded fingerprint.
                Default is 2048.
            min_radius (int, optional): The minimum radius for substructure extraction.
                Default is 0 (includes single atoms).
            radius (int, optional): The maximum radius for substructure extraction.
                Default is 3 (corresponds to DRFP6).
            rings (bool, optional): Whether to include full rings as substructures.
                Default is True.
            root_central_atom (bool, optional): Whether to root the central atom
                of substructures when generating SMILES. Default is True.
            include_hydrogens (bool, optional): Whether to include hydrogens in the
                molecular representation. Default is False.
        """
        self.n_folded_length = n_folded_length
        self.min_radius = min_radius
        self.radius = radius
        self.rings = rings
        self.root_central_atom = root_central_atom
        self.include_hydrogens = include_hydrogens

    def construct(self, smiles: str) -> torch.Tensor:
        """
        Generates a DRFP fingerprint for a single reaction SMILES string.

        The method uses `DrfpEncoder.internal_encode` to get the hashed difference
        features of the reaction and then `DrfpEncoder.fold` to create the
        final binary fingerprint vector as a NumPy array, which is then converted
        to a PyTorch Tensor.

        Args:
            smiles: The reaction SMILES string (e.g., "R1.R2>A>P1.P2").

        Returns:
            A PyTorch Tensor (dtype=torch.float32) representing the folded DRFP fingerprint.

        Raises:
            NoReactionError: If the input SMILES is not a valid reaction SMILES
                             (as detected by `DrfpEncoder.internal_encode`).
            RuntimeError: For other errors encountered during fingerprint generation,
                          wrapping the original exception.
        """
        hashed_diff_np, _shingles_diff_bytes = DRFPUtil.internal_encode(
            smiles,
            radius=self.radius,
            min_radius=self.min_radius,
            rings=self.rings,
            get_atom_indices=False,
            root_central_atom=self.root_central_atom,
            include_hydrogens=self.include_hydrogens,
        )

        fingerprint_np, _on_bits = DRFPUtil.fold(
            hashed_diff_np,
            length=self.n_folded_length,
        )
        return torch.from_numpy(fingerprint_np).to(torch.float32)


class DRFPUtil:
    """
    A utility class for encoding SMILES as drfp fingerprints.
    """

    @staticmethod
    def shingling_from_mol(
        in_mol: Mol,
        radius: int = 3,
        rings: bool = True,
        min_radius: int = 0,
        get_atom_indices: bool = False,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
    ) -> Union[List[str], Tuple[List[str], Dict[str, List[Set[int]]]]]:
        """Creates a molecular shingling from a RDKit molecule (rdkit.Chem.rdchem.Mol).

        Arguments:
            in_mol: A RDKit molecule instance
            radius: The drfp radius (a radius of 3 corresponds to drfp6)
            rings: Whether or not to include rings in the shingling
            min_radius: The minimum radius that is used to extract n-grams

        Returns:
            The molecular shingling.
        """

        if include_hydrogens:
            in_mol = AllChem.AddHs(in_mol)

        shingling = []
        atom_indices = defaultdict(list)

        if rings:
            for ring in AllChem.GetSymmSSSR(in_mol):
                bonds = set()
                ring = list(ring)
                indices = set()
                for i in ring:
                    for j in ring:
                        if i != j:
                            indices.add(i)
                            indices.add(j)
                            bond = in_mol.GetBondBetweenAtoms(i, j)
                            if bond is not None:
                                bonds.add(bond.GetIdx())

                ngram = AllChem.MolToSmiles(
                    AllChem.PathToSubmol(in_mol, list(bonds)),
                    canonical=True,
                    allHsExplicit=True,
                ).encode("utf-8")

                shingling.append(ngram)

                if get_atom_indices:
                    atom_indices[ngram].append(indices)

        if min_radius == 0:
            for i, atom in enumerate(in_mol.GetAtoms()):
                ngram = atom.GetSmarts().encode("utf-8")
                shingling.append(ngram)

                if get_atom_indices:
                    atom_indices[ngram].append(set([atom.GetIdx()]))

        for index, _ in enumerate(in_mol.GetAtoms()):
            for i in range(1, radius + 1):
                p = AllChem.FindAtomEnvironmentOfRadiusN(
                    in_mol, i, index, useHs=include_hydrogens
                )
                amap = {}
                submol = AllChem.PathToSubmol(in_mol, p, atomMap=amap)

                if index not in amap:
                    continue

                smiles = ""

                if root_central_atom:
                    smiles = AllChem.MolToSmiles(
                        submol,
                        rootedAtAtom=amap[index],
                        canonical=True,
                        allHsExplicit=True,
                    )
                else:
                    smiles = AllChem.MolToSmiles(
                        submol,
                        canonical=True,
                        allHsExplicit=True,
                    )

                if smiles != "":
                    shingling.append(smiles.encode("utf-8"))
                    if get_atom_indices:
                        atom_indices[smiles.encode("utf-8")].append(set(amap.keys()))

        if not root_central_atom:
            for key in atom_indices:
                atom_indices[key] = list(set([frozenset(s) for s in atom_indices[key]]))

        # Set ensures that the same shingle is not hashed multiple times
        # (which would not change the hash, since there would be no new minima)
        if get_atom_indices:
            return list(set(shingling)), atom_indices
        else:
            return list(set(shingling))

    @staticmethod
    def internal_encode(
        in_smiles: str,
        radius: int = 3,
        min_radius: int = 0,
        rings: bool = True,
        get_atom_indices: bool = False,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, List[Set[int]]]]]],
    ]:
        """Creates an drfp array from a reaction SMILES string.

        Arguments:
            in_smiles: A valid reaction SMILES string
            radius: The drfp radius (a radius of 3 corresponds to drfp6)
            min_radius: The minimum radius that is used to extract n-grams
            rings: Whether or not to include rings in the shingling

        Returns:
            A tuple with two arrays, the first containing the drfp hash values, the second the substructure SMILES
        """

        atom_indices = {}
        atom_indices["reactants"] = []
        atom_indices["products"] = []

        sides = in_smiles.split(">")
        if len(sides) < 3:
            raise ValueError(
                f"The following is not a valid reaction SMILES: '{in_smiles}'"
            )

        if len(sides[1]) > 0:
            sides[0] += "." + sides[1]

        left = sides[0].split(".")
        right = sides[2].split(".")

        left_shingles = set()
        right_shingles = set()

        for l in left:
            mol = AllChem.MolFromSmiles(l)

            if not mol:
                atom_indices["reactants"].append(None)
                continue

            if get_atom_indices:
                sh, ai = DRFPUtil.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
                atom_indices["reactants"].append(ai)
            else:
                sh = DRFPUtil.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            for s in sh:
                right_shingles.add(s)

        for r in right:
            mol = AllChem.MolFromSmiles(r)

            if not mol:
                atom_indices["products"].append(None)
                continue

            if get_atom_indices:
                sh, ai = DRFPUtil.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
                atom_indices["products"].append(ai)
            else:
                sh = DRFPUtil.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            for s in sh:
                left_shingles.add(s)

        s = right_shingles.symmetric_difference(left_shingles)

        if get_atom_indices:
            return DRFPUtil.hash(list(s)), list(s), atom_indices
        else:
            return DRFPUtil.hash(list(s)), list(s)

    @staticmethod
    def hash(shingling: List[str]) -> np.ndarray:
        """Directly hash all the SMILES in a shingling to a 32-bit integer.

        Arguments:
            shingling: A list of n-grams

        Returns:
            A list of hashed n-grams
        """
        hash_values = []

        for t in shingling:
            # Convert to signed 32-bit by taking modulo 2^32 and subtracting 2^32 if >= 2^31
            h = int(blake2b(t, digest_size=4).hexdigest(), 16)
            h = h & 0xFFFFFFFF  # Ensure 32-bit
            if h >= 0x80000000:  # If >= 2^31
                h -= 0x100000000  # Subtract 2^32 to make negative
            hash_values.append(h)

        return np.array(hash_values, dtype=np.int32)

    @staticmethod
    def fold(
        hash_values: np.ndarray, length: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Folds the hash values to a binary vector of a given length.

        Arguments:
            hash_value: An array containing the hash values
            length: The length of the folded fingerprint

        Returns:
            A tuple containing the folded fingerprint and the indices of the on bits
        """

        folded = np.zeros(length, dtype=np.uint8)
        on_bits = hash_values % length
        folded[on_bits] = 1

        return folded, on_bits

    @staticmethod
    def encode(
        X: Union[Iterable, str],
        n_folded_length: int = 2048,
        min_radius: int = 0,
        radius: int = 3,
        rings: bool = True,
        mapping: bool = False,
        atom_index_mapping: bool = False,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
        show_progress_bar: bool = False,
    ) -> Union[
        List[np.ndarray],
        Tuple[List[np.ndarray], Dict[int, Set[str]]],
        Tuple[List[np.ndarray], Dict[int, Set[str]]],
        List[Dict[str, List[Dict[str, List[Set[int]]]]]],
    ]:
        """Encodes a list of reaction SMILES using the drfp fingerprint.

        Args:
            X: An iterable (e.g. List) of reaction SMILES or a single reaction SMILES to be encoded
            n_folded_length: The folded length of the fingerprint (the parameter for the modulo hashing)
            min_radius: The minimum radius of a substructure (0 includes single atoms)
            radius: The maximum radius of a substructure
            rings: Whether to include full rings as substructures
            mapping: Return a feature to substructure mapping in addition to the fingerprints
            atom_index_mapping: Return the atom indices of mapped substructures for each reaction
            root_central_atom: Whether to root the central atom of substructures when generating SMILES
            show_progress_bar: Whether to show a progress bar when encoding reactions

        Returns:
            A list of drfp fingerprints or, if mapping is enabled, a tuple containing a list of drfp fingerprints and a mapping dict.
        """
        if isinstance(X, str):
            X = [X]

        show_progress_bar = not show_progress_bar

        # If mapping is required for atom_index_mapping
        if atom_index_mapping:
            mapping = True

        result = []
        result_map = defaultdict(set)
        atom_index_maps = []

        for _, x in tqdm(enumerate(X), total=len(X), disable=show_progress_bar):
            if atom_index_mapping:
                hashed_diff, smiles_diff, atom_index_map = DRFPUtil.internal_encode(
                    x,
                    min_radius=min_radius,
                    radius=radius,
                    rings=rings,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
            else:
                hashed_diff, smiles_diff = DRFPUtil.internal_encode(
                    x,
                    min_radius=min_radius,
                    radius=radius,
                    rings=rings,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            difference_folded, on_bits = DRFPUtil.fold(
                hashed_diff,
                length=n_folded_length,
            )

            if mapping:
                for unfolded_index, folded_index in enumerate(on_bits):
                    result_map[folded_index].add(
                        smiles_diff[unfolded_index].decode("utf-8")
                    )

            if atom_index_mapping:
                aidx_bit_map = {}
                aidx_bit_map["reactants"] = []
                aidx_bit_map["products"] = []

                for reactant in atom_index_map["reactants"]:
                    r = defaultdict(list)
                    for key, value in reactant.items():
                        if key in smiles_diff:
                            idx = smiles_diff.index(key)
                            r[on_bits[idx]].append(value)
                    aidx_bit_map["reactants"].append(r)

                for product in atom_index_map["products"]:
                    r = defaultdict(list)
                    for key, value in product.items():
                        if key in smiles_diff:
                            idx = smiles_diff.index(key)
                            r[on_bits[idx]].append(value)
                    aidx_bit_map["products"].append(r)

                atom_index_maps.append(aidx_bit_map)

            result.append(difference_folded)

        r = [result]

        if mapping:
            r.append(result_map)

        if atom_index_mapping:
            r.append(atom_index_maps)

        if len(r) == 1:
            return r[0]
        else:
            return tuple(r)
