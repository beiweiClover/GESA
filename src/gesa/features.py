"""Categorical molecular graph features used by both GESA encoders."""

from __future__ import annotations

from typing import List

from rdkit import Chem


ATOM_FEATURE_DIMS = [119, 11, 12, 8, 2, 2, 4]
BOND_FEATURE_DIMS = [5, 7, 2, 2]

HYBRIDIZATION_TO_INDEX = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
}
CHIRAL_TO_INDEX = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
BOND_TYPE_TO_INDEX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
BOND_STEREO_TO_INDEX = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}


def atom_features(atom: Chem.Atom) -> List[int]:
    charge = atom.GetFormalCharge()
    return [
        min(max(atom.GetAtomicNum(), 0), 118),
        min(max(atom.GetDegree(), 0), 10),
        charge + 5 if -5 <= charge <= 5 else 11,
        HYBRIDIZATION_TO_INDEX.get(atom.GetHybridization(), 7),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        CHIRAL_TO_INDEX.get(atom.GetChiralTag(), 3),
    ]


def bond_features(bond: Chem.Bond) -> List[int]:
    return [
        BOND_TYPE_TO_INDEX.get(bond.GetBondType(), 4),
        BOND_STEREO_TO_INDEX.get(bond.GetStereo(), 6),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]
