"""SMILES-to-graph conversion and batching for GESA inference."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from rdkit import Chem

from .features import BOND_FEATURE_DIMS, atom_features, bond_features


def smiles_to_graph(smiles: str) -> Dict[str, torch.Tensor]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not parse a non-empty molecule: {smiles!r}")
    nodes = [atom_features(atom) for atom in molecule.GetAtoms()]
    edges = []
    attributes = []
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        values = bond_features(bond)
        edges.extend([[first, second], [second, first]])
        attributes.extend([values, values])
    return {
        "x": torch.tensor(nodes, dtype=torch.long),
        "edge_index": (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.empty((2, 0), dtype=torch.long)
        ),
        "edge_attr": (
            torch.tensor(attributes, dtype=torch.long)
            if attributes else torch.empty((0, len(BOND_FEATURE_DIMS)), dtype=torch.long)
        ),
    }


def collate_graphs(
    graphs: Sequence[Dict[str, torch.Tensor]],
    sa_priors: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if not graphs:
        raise ValueError("Cannot collate an empty graph sequence")
    xs, edges, attrs, assignments, positions = [], [], [], [], []
    offset = 0
    for graph_id, graph in enumerate(graphs):
        atoms = int(graph["x"].shape[0])
        xs.append(graph["x"])
        edges.append(graph["edge_index"] + offset)
        attrs.append(graph["edge_attr"])
        assignments.append(torch.full((atoms,), graph_id, dtype=torch.long))
        positions.append(torch.arange(atoms, dtype=torch.long))
        offset += atoms
    batch = {
        "x": torch.cat(xs),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(attrs),
        "graph_index": torch.cat(assignments),
        "atom_position": torch.cat(positions),
        "num_graphs": torch.tensor(len(graphs), dtype=torch.long),
    }
    if sa_priors is not None:
        batch["xsa"] = sa_priors
    return batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
