from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from rdkit.Chem.rdchem import Atom, Bond, BondType, HybridizationType


def make_featurizer(
    featurizer_name: str,
) -> Tuple[Callable, int]:
    """Create function that featurizes atoms or bonds.

    Args:
        featurizer_name: Option for atom or bond features, currently: "atom_atomic_num", "atom_rdkit_base", "bond_rdkit_base".
            To add new featurizers, add an elif statement, and make a list containing tuples of (arbitrary function acting on atom/bond,
            list of possible values) If the list of possible values is empty, the function output will be used directly as feature.

    Returns:
        A function to featurize rdkit atoms.
    """

    if featurizer_name == "atom_atomic_num":
        l = [
            (Atom.GetAtomicNum, list(range(1, 37)) + [53]),
        ]

    elif featurizer_name == "atom_rdkit_base":
        l = [
            (Atom.GetAtomicNum, list(range(1, 37)) + [53]),
            (Atom.GetTotalDegree, list(range(6))),
            (Atom.GetFormalCharge, [-2, -1, 0, 1, 2]),
            (Atom.GetTotalNumHs, list(range(5))),
            (
                Atom.GetHybridization,
                [
                    HybridizationType.S,
                    HybridizationType.SP,
                    HybridizationType.SP2,
                    HybridizationType.SP2D,
                    HybridizationType.SP3,
                    HybridizationType.SP3D,
                    HybridizationType.SP3D2,
                ],
            ),
            (lambda item: Atom.GetIsAromatic(item), []),
            (lambda item: Atom.GetMass(item) * 0.01, []),
        ]

    elif featurizer_name == "atom_rdkit_organic":
        l = [
            (Atom.GetAtomicNum, [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]),
            (Atom.GetTotalDegree, list(range(6))),
            (Atom.GetFormalCharge, [-2, -1, 0, 1, 2]),
            (Atom.GetTotalNumHs, list(range(5))),
            (
                Atom.GetHybridization,
                [
                    HybridizationType.S,
                    HybridizationType.SP,
                    HybridizationType.SP2,
                    HybridizationType.SP3,
                ],
            ),
            (lambda item: Atom.GetIsAromatic(item), []),
            (lambda item: Atom.GetMass(item) * 0.01, []),
            (lambda item: Atom.IsInRingSize(item, 3), []),
            (lambda item: Atom.IsInRingSize(item, 4), []),
            (lambda item: Atom.IsInRingSize(item, 5), []),
            (lambda item: Atom.IsInRingSize(item, 6), []),
            (lambda item: Atom.IsInRingSize(item, 7), []),
        ]

    elif featurizer_name == "bond_rdkit_base":
        l = [
            (
                Bond.GetBondType,
                [
                    BondType.SINGLE,
                    BondType.DOUBLE,
                    BondType.TRIPLE,
                    BondType.AROMATIC,
                ],
            ),
            (lambda item: Bond.GetIsConjugated(item), []),
            # (lambda item: Bond.IsInRing(item), []), # introduced noise in combination with features below
            (lambda item: Bond.IsInRingSize(item, 3), []),
            (lambda item: Bond.IsInRingSize(item, 4), []),
            (lambda item: Bond.IsInRingSize(item, 5), []),
            (lambda item: Bond.IsInRingSize(item, 6), []),
            (lambda item: Bond.IsInRingSize(item, 7), []),
        ]

    else:
        raise NotImplementedError("Option not implemented: " + featurizer_name)

    feature_dim = sum([(len(options) + 1) for _, options in l])

    def featurizer(
        item: Union[Atom, Bond],
    ) -> ArrayLike:
        """Create function that featurizes atoms.

        Args:
            item: Rdkit atom or bond

        Returns:
            Numpy array of features
        """

        features = []
        if item is None:
            return [0] * feature_dim
        for func, options in l:
            if len(options) > 0:
                features.extend(one_hot_unk(item, func, options))
            else:
                features.append(func(item))

        return features

    return featurizer  # , feature_dim


def one_hot_unk(item, func, options):
    # TODO add docstring
    x = [0] * (len(options) + 1)
    option_dict = {j: i for i, j in enumerate(options)}
    x[option_dict.get(func(item), len(option_dict))] = 1

    return x
