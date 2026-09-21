"""Fourteen explicit synthetic-accessibility priors used by GESA."""

from __future__ import annotations

import gzip
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


FEATURE_NAMES = [
    "fragment_contribution", "atom_count", "size_penalty",
    "chiral_center_count", "stereo_penalty", "spiro_atom_count",
    "spiro_penalty", "bridgehead_atom_count", "bridgehead_penalty",
    "macrocycle_count", "macrocycle_penalty", "morgan_feature_count",
    "morgan_nonzero_bits", "fingerprint_density_correction",
]

# Fitted on the immutable target training split only.
SCALER_MEAN = np.asarray([
    0.9634595513343811, 25.784067153930664, 0.431432843208313,
    2.2420156002044678, 0.40315374732017517, 0.05027981474995613,
    0.014490412548184395, 0.24305859208106995, 0.048094648867845535,
    0.04792493209242821, 0.01238580048084259, 72.47859191894531,
    48.977474212646484, 0.00039571174420416355,
], dtype=np.float64)
SCALER_STD = np.asarray([
    0.5807934403419495, 9.485251426696777, 0.20499302446842194,
    2.4027180671691895, 0.30843469500541687, 0.24253346025943756,
    0.0677308514714241, 0.9100112915039062, 0.1610144078731537,
    0.25019633769989014, 0.05975944921374321, 27.409364700317383,
    16.840423583984375, 0.008767370134592056,
], dtype=np.float64)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def _default_score_path() -> Path:
    import rdkit

    candidate = Path(rdkit.__file__).resolve().parent / "Contrib" / "SA_Score" / "fpscores.pkl.gz"
    if not candidate.is_file():
        raise FileNotFoundError(
            "RDKit Contrib SA_Score/fpscores.pkl.gz is required; pass score_path explicitly"
        )
    return candidate


@lru_cache(maxsize=4)
def load_fragment_scores(score_path: str | Path | None = None) -> Dict[int, float]:
    path = Path(score_path) if score_path else _default_score_path()
    with gzip.open(path, "rb") as handle:
        rows = pickle.load(handle)
    scores: Dict[int, float] = {}
    for row in rows:
        for feature_id in row[1:]:
            scores[int(feature_id)] = float(row[0])
    return scores


def raw_sa_priors(molecule: Chem.Mol, fragment_scores: Dict[int, float]) -> np.ndarray:
    fingerprint = _MORGAN.GetSparseCountFingerprint(molecule)
    nonzero = fingerprint.GetNonzeroElements()
    feature_count = int(sum(nonzero.values()))
    fragment_total = sum(
        fragment_scores.get(int(key), -4.0) * int(count) for key, count in nonzero.items()
    )
    fragment_score = fragment_total / max(1, feature_count)
    atom_count = int(molecule.GetNumAtoms())
    chiral_count = len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    bridgehead_count = int(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule))
    spiro_count = int(rdMolDescriptors.CalcNumSpiroAtoms(molecule))
    macrocycle_count = sum(1 for ring in molecule.GetRingInfo().AtomRings() if len(ring) > 8)
    nonzero_bits = len(nonzero)
    density = (
        0.5 * math.log(float(atom_count) / nonzero_bits)
        if atom_count > nonzero_bits and nonzero_bits else 0.0
    )
    return np.asarray([
        fragment_score, atom_count, atom_count ** 1.005 - atom_count,
        chiral_count, math.log10(chiral_count + 1), spiro_count,
        math.log10(spiro_count + 1), bridgehead_count,
        math.log10(bridgehead_count + 1), macrocycle_count,
        math.log10(2.0) if macrocycle_count else 0.0,
        feature_count, nonzero_bits, density,
    ], dtype=np.float32)


def standardized_sa_priors(
    molecules: Sequence[Chem.Mol], score_path: str | Path | None = None
) -> torch.Tensor:
    scores = load_fragment_scores(score_path)
    raw = np.stack([raw_sa_priors(molecule, scores) for molecule in molecules])
    standardized = ((raw - SCALER_MEAN) / SCALER_STD).astype(np.float32)
    return torch.from_numpy(standardized)


class XSAEncoder(torch.nn.Module):
    """Neural projection of the explicit prior vector; name preserves checkpoint keys."""

    def __init__(
        self, input_dim: int = 14, hidden_dim: int = 64,
        output_dim: int = 32, dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)
