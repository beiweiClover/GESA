"""High-level checkpoint loading and batched SMILES inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem

from .graph import collate_graphs, move_batch, smiles_to_graph
from .model import GESA
from .sa_priors import standardized_sa_priors


def load_gesa(
    checkpoint: str | Path,
    device: str | torch.device | None = None,
) -> Tuple[GESA, Dict]:
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = GESA()
    payload = model.load_release_checkpoint(checkpoint, map_location="cpu")
    model.to(selected_device).eval()
    return model, payload


@torch.no_grad()
def predict_smiles(
    model: GESA,
    smiles: Sequence[str],
    *,
    batch_size: int = 128,
    score_path: str | Path | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return `P(HS)` scores and per-row errors for the supplied SMILES.

    HS is label 1 and the positive class. Higher scores indicate greater
    predicted synthetic difficulty; the natural decision rule is score >= 0.5
    for HS.
    """
    device = next(model.parameters()).device
    scores = np.full(len(smiles), np.nan, dtype=np.float32)
    errors = [""] * len(smiles)
    valid = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None or molecule.GetNumAtoms() == 0:
            errors[index] = "RDKit parse failure or empty molecule"
        else:
            valid.append((index, str(value), molecule))
    for start in range(0, len(valid), int(batch_size)):
        rows = valid[start:start + int(batch_size)]
        graphs = [smiles_to_graph(value) for _, value, _ in rows]
        priors = standardized_sa_priors([molecule for _, _, molecule in rows], score_path)
        batch = move_batch(collate_graphs(graphs, priors), device)
        probabilities = torch.sigmoid(model(batch)).detach().float().cpu().numpy()
        for (index, _, _), probability in zip(rows, probabilities):
            scores[index] = float(probability)
    return scores, errors
