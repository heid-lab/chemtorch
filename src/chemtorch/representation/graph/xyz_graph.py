import os.path as osp
from typing import List, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Data
from typing_extensions import override

from chemtorch.representation.abstract_representation import (
    AbstractRepresentation,
)


def read_xyz(file_path: str) -> Tuple[List[str], torch.Tensor]:
    """
    Reads a standard XYZ file and returns atomic symbols and coordinates.

    Args:
        file_path (str): The path to the .xyz file.

    Returns:
        A tuple containing a list of atomic symbols (str) and a tensor
        of atomic coordinates ([num_atoms, 3]).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed or cannot be parsed.
    """
    if not osp.exists(file_path):
        raise FileNotFoundError(f"XYZ file not found at: {file_path}")

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        atomic_symbols = []
        coords = []
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 4:
                atomic_symbols.append(parts[0])
                coords.append([float(p) for p in parts[1:4]])

        return atomic_symbols, torch.tensor(coords, dtype=torch.float)
    except (IOError, IndexError, ValueError) as e:
        raise ValueError(f"Error reading or parsing XYZ file {file_path}: {e}")


def symbols_to_atomic_numbers(symbols: List[str]) -> torch.Tensor:
    """
    Converts a list of atomic symbols (e.g., ['C', 'H']) to a tensor of atomic numbers.

    Args:
        symbols (List[str]): A list of atomic symbols.

    Returns:
        A tensor of atomic numbers.
    """
    pt = Chem.GetPeriodicTable()
    try:
        atomic_nums = [pt.GetAtomicNumber(s) for s in symbols]
        return torch.tensor(atomic_nums, dtype=torch.long)
    except Exception as e:
        raise ValueError(f"Error converting symbols to atomic numbers: {e}")


class XYZReactionRepresentation(AbstractRepresentation[Data]):
    """
    Constructs a 3D representation of a reaction from XYZ files.

    This representation reads the 3D structures for a reactant, transition state (ts),
    and product from their respective .xyz files and packages them into a single
    PyTorch Geometric `Data` object. It is designed to be used with `DatasetBase`.

    The resulting `Data` object for each sample will contain:
    - `z_reactant`, `pos_reactant`: Atomic numbers and coordinates for the reactant.
    - `z_ts`, `pos_ts`: Atomic numbers and coordinates for the transition state.
    - `z_product`, `pos_product`: Atomic numbers and coordinates for the product.
    - `smiles`: The reaction SMILES string, for reference.
    """

    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): The root directory where reaction subfolders (e.g., 'reaction_1',
                            'reaction_2') are located.
        """
        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f"The specified root directory does not exist: {root_dir}"
            )
        self.root_dir = root_dir

    # TODO: ORDER MATTERS!!!!!!!!!!!!!!!!!!!!!!!!!!!
    @override
    def construct(self, reaction_dir: str, smiles: str, **kwargs) -> Data:
        """
        Constructs a single reaction graph from its corresponding XYZ files.

        This method is called by `DatasetBase` for each row in the DataFrame.

        Args:
            reaction_dir (str): The name of the subdirectory within `root_dir`
                                containing the XYZ files for this reaction.
            smiles (str): The reaction SMILES string.
            **kwargs: Additional keyword arguments from the DataFrame row are ignored.

        Returns:
            A PyG `Data` object containing the 3D structures.

        Raises:
            FileNotFoundError: If the reaction directory or any required .xyz file is not found.
            ValueError: If the number of atoms is inconsistent across structures.
        """
        reaction_dir = str(reaction_dir).zfill(6)
        folder_path = osp.join(self.root_dir, f"rxn{reaction_dir}")
        if not osp.isdir(folder_path):
            raise FileNotFoundError(
                f"Reaction directory not found: {folder_path}"
            )

        structures = {}
        for state in ["ts"]:
            file_path = osp.join(folder_path, f"{state}{reaction_dir}.xyz")
            symbols, pos = read_xyz(file_path)
            z = symbols_to_atomic_numbers(symbols)
            structures[state] = {"z": z, "pos": pos}

        num_atoms = structures["ts"]["pos"].shape[0]
        if not all(
            s["pos"].shape[0] == num_atoms for s in structures.values()
        ):
            raise ValueError(
                f"Inconsistent number of atoms in reaction {reaction_dir}."
            )

        data = Data(
            z=structures["ts"]["z"],
            pos=structures["ts"]["pos"],
            smiles=smiles,
            num_nodes=num_atoms,
        )

        return data
