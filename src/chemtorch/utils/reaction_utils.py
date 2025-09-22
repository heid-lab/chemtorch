# Code author: Maximilian Peter Kovar
from typing import List, Tuple, Optional, Union
import numpy as np
from rdkit import Chem 

def get_atom_index_by_mapnum(mol: Chem.Mol, mapnum: int) -> Optional[int]:
    """
    Get the atom index for an atom with a specific atom map number.
    
    Args:
        mol (Chem.Mol): The RDKit molecule object to search.
        mapnum (int): The atom map number to search for.
        
    Returns:
        Optional[int]: The atom index if found, None otherwise.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == mapnum:
            return atom.GetIdx()
    return None

def unmap_smarts(smarts: str) -> str:
    """
    Remove atom map numbers from a SMARTS string.
    
    Args:
        smarts (str): The input SMARTS string with atom map numbers.
        
    Returns:
        str: The SMARTS string with atom map numbers removed.
        
    Raises:
        ValueError: If the SMARTS string cannot be parsed.
    """
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError(f"Could not parse the given SMARTS string: {smarts}")
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    unmapped_smarts = Chem.MolToSmarts(mol)
    return unmapped_smarts

def unmap_smiles(smiles: str) -> str:
    """
    Remove atom map numbers from a SMILES string.
    
    Args:
        smiles (str): The input SMILES string with atom map numbers.
        
    Returns:
        str: The SMILES string with atom map numbers removed.
        
    Raises:
        ValueError: If the SMILES string cannot be parsed.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params=params)
    if mol is None:
        raise ValueError(f"Could not parse the given SMILES string: {smiles}")
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    # CHANGED: Use MolToSmiles to return a proper SMILES string
    unmapped_smiles = Chem.MolToSmiles(mol)
    return unmapped_smiles

def smarts2smarts(smarts: str) -> str:
    """
    Parse and reformat a SMARTS string to ensure proper formatting.
    
    Args:
        smarts (str): The input SMARTS string.
        
    Returns:
        str: The reformatted SMARTS string.
        
    Raises:
        ValueError: If the SMARTS string cannot be parsed.
    """
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError(f"Could not parse the given SMARTS string: {smarts}")
    new_smarts_str = Chem.MolToSmarts(mol)
    return new_smarts_str

def smiles2smiles(smiles_str: str) -> str:
    """
    Parse and reformat a SMILES string to ensure proper formatting.
    
    Args:
        smiles_str (str): The input SMILES string.
        
    Returns:
        str: The reformatted SMILES string.
        
    Raises:
        ValueError: If the SMILES string cannot be parsed.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles_str, params=params)
    if mol is None:
        raise ValueError(f"Could not parse the given SMILES string: {smiles_str}")
    # CHANGED: Use MolToSmiles for proper SMILES output
    new_smiles_str = Chem.MolToSmiles(mol)
    return new_smiles_str

def bondtypes(atom: Chem.Atom) -> List[Chem.BondType]:
    """
    Get a sorted list of bond types for all bonds connected to an atom.
    
    Args:
        atom (Chem.Atom): The RDKit atom object.
        
    Returns:
        List[Chem.BondType]: A sorted list of bond types connected to the atom.
    """
    return sorted(b.GetBondType() for b in atom.GetBonds())

def neighbors(atom: Chem.Atom) -> List[int]:
    """
    Get a sorted list of atom map numbers for all neighboring atoms.
    
    Args:
        atom (Chem.Atom): The RDKit atom object.
        
    Returns:
        List[int]: A sorted list of atom map numbers of neighboring atoms.
    """
    return sorted(n.GetAtomMapNum() for n in atom.GetNeighbors())

def neighbors_and_bondtypes(atom: Chem.Atom) -> List[Union[int, Chem.BondType]]:
    """
    Get a combined list of neighboring atom map numbers and bond types.
    
    Args:
        atom (Chem.Atom): The RDKit atom object.
        
    Returns:
        List[Union[int, Chem.BondType]]: A combined list containing neighbor map numbers and bond types.
    """
    return neighbors(atom) + bondtypes(atom)

def remove_atoms_from_rxn(mr: Chem.Mol, mp: Chem.Mol, atoms_to_remove: np.ndarray) -> List[str]:
    """
    Remove specified atoms from reactant and product molecules and return SMILES.
    
    Args:
        mr (Chem.Mol): The reactant molecule.
        mp (Chem.Mol): The product molecule.
        atoms_to_remove (np.ndarray): A 2D numpy array of shape (N, 2) where each row contains
                                     a pair of atom indices [reactant_idx, product_idx] representing
                                     corresponding atoms to be removed from the reactant and product
                                     molecules respectively. The indices are 0-based atom indices
                                     within each molecule.

    Returns:
        List[str]: A list containing the SMILES strings of the modified reactant and product.
    """
    r_p_smiles_new_2 = []
    for mol_i, mol in enumerate([mr, mp]):
        editable_mol = Chem.EditableMol(mol)
        for i in sorted(atoms_to_remove[:, mol_i], reverse=True):
            editable_mol.RemoveAtom(int(i))
        r_p_smiles_new_2.append(Chem.MolToSmiles(editable_mol.GetMol(), allHsExplicit=True))
    return r_p_smiles_new_2

def get_reaction_core(r_smiles: str, p_smiles: str) -> Tuple[str, List[int]]:
    """
    Extract the reaction core by removing atoms that don't change during the reaction.
    
    This function identifies atoms in the reactant and product that have identical
    neighborhoods (same neighbors and bond types) and removes them to extract
    only the reaction core - the atoms that actually participate in the transformation.
    
    Args:
        r_smiles (str): The reactant SMILES string.
        p_smiles (str): The product SMILES string.
        
    Returns:
        Tuple[str, List[int]]: A tuple containing:
            - The reaction SMILES with only the changing atoms (reaction core)
            - A list of atom map numbers that were removed (0-indexed)
            
    Raises:
        ValueError: If SMILES cannot be parsed or processed.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mr = Chem.MolFromSmiles(r_smiles, params=params)
    mp = Chem.MolFromSmiles(p_smiles, params=params)
    
    if mr is None:
        raise ValueError(f"Could not parse reactant SMILES: {r_smiles}")
    if mp is None:
        raise ValueError(f"Could not parse product SMILES: {p_smiles}")

    p_map2i = {a.GetAtomMapNum(): a.GetIdx() for a in mp.GetAtoms()}

    def r2p_atom(ar):
        return mp.GetAtomWithIdx(p_map2i[ar.GetAtomMapNum()])

    atoms_to_remove = np.array([(ar.GetIdx(), p_map2i[ar.GetAtomMapNum()]) for ar in mr.GetAtoms() 
                                      if neighbors_and_bondtypes(ar) == neighbors_and_bondtypes(r2p_atom(ar))
                                     ])
    if len(atoms_to_remove) == 0:
        return ('>>'.join([r_smiles, p_smiles]), [])

    reaction_core = remove_atoms_from_rxn(mr, mp, atoms_to_remove)

    removed_mapnums = sorted([ar.GetAtomMapNum()-1 for ar in mr.GetAtoms() 
                              if neighbors_and_bondtypes(ar) == neighbors_and_bondtypes(r2p_atom(ar))
                             ])

    return ('>>'.join(reaction_core), removed_mapnums)