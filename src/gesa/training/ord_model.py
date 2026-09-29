"""Original shared-GINE reaction pretraining model, data loader, and losses."""

from __future__ import annotations

import gzip
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import IterableDataset, get_worker_info


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
    atomic_number = min(max(atom.GetAtomicNum(), 0), 118)
    degree = min(max(atom.GetDegree(), 0), 10)
    charge = atom.GetFormalCharge()
    charge_index = charge + 5 if -5 <= charge <= 5 else 11
    hybridization = HYBRIDIZATION_TO_INDEX.get(atom.GetHybridization(), 7)
    chirality = CHIRAL_TO_INDEX.get(atom.GetChiralTag(), 3)
    return [
        atomic_number,
        degree,
        charge_index,
        hybridization,
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        chirality,
    ]


def bond_features(bond: Chem.Bond) -> List[int]:
    return [
        BOND_TYPE_TO_INDEX.get(bond.GetBondType(), 4),
        BOND_STEREO_TO_INDEX.get(bond.GetStereo(), 6),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


def normalize_bond_keys(values: Sequence[Sequence[int]]) -> set:
    return {tuple(sorted((int(value[0]), int(value[1])))) for value in values}


def mapped_molecule_to_graph(
    mapped_smiles: str,
    center_atom_maps: Sequence[int],
    center_bonds: Sequence[Sequence[int]],
) -> Dict[str, torch.Tensor]:
    """Converts one mapped molecule to tensors without using map IDs as features."""
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError("RDKit could not parse mapped SMILES: %s" % mapped_smiles)
    center_atom_set = {int(value) for value in center_atom_maps}
    center_bond_set = normalize_bond_keys(center_bonds)

    node_x: List[List[int]] = []
    atom_y: List[float] = []
    for atom in mol.GetAtoms():
        node_x.append(atom_features(atom))
        atom_y.append(float(atom.GetAtomMapNum() in center_atom_set))

    directed_edges: List[List[int]] = []
    directed_attr: List[List[int]] = []
    undirected_edges: List[List[int]] = []
    undirected_attr: List[List[int]] = []
    bond_y: List[float] = []
    for bond in mol.GetBonds():
        first = bond.GetBeginAtomIdx()
        second = bond.GetEndAtomIdx()
        features = bond_features(bond)
        directed_edges.extend([[first, second], [second, first]])
        directed_attr.extend([features, features])
        undirected_edges.append([first, second])
        undirected_attr.append(features)
        first_map = mol.GetAtomWithIdx(first).GetAtomMapNum()
        second_map = mol.GetAtomWithIdx(second).GetAtomMapNum()
        key = tuple(sorted((first_map, second_map)))
        bond_y.append(float(first_map > 0 and second_map > 0 and key in center_bond_set))

    num_atoms = mol.GetNumAtoms()
    return {
        "x": torch.tensor(node_x, dtype=torch.long),
        "edge_index": (
            torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
            if directed_edges
            else torch.empty((2, 0), dtype=torch.long)
        ),
        "edge_attr": (
            torch.tensor(directed_attr, dtype=torch.long)
            if directed_attr
            else torch.empty((0, len(BOND_FEATURE_DIMS)), dtype=torch.long)
        ),
        "bond_index": (
            torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
            if undirected_edges
            else torch.empty((2, 0), dtype=torch.long)
        ),
        "bond_attr": (
            torch.tensor(undirected_attr, dtype=torch.long)
            if undirected_attr
            else torch.empty((0, len(BOND_FEATURE_DIMS)), dtype=torch.long)
        ),
        "atom_y": torch.tensor(atom_y, dtype=torch.float32),
        "bond_y": torch.tensor(bond_y, dtype=torch.float32),
        "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
    }


def record_to_graphs(record: Dict[str, Any]) -> Dict[str, Any]:
    center_atoms = record["center_atom_maps"]
    reactant_bonds = record["reactant_center_bonds"]
    product_bonds = record["product_center_bonds"]
    molecules = []
    roles = []
    for smiles in record["reactants"]:
        molecules.append(mapped_molecule_to_graph(smiles, center_atoms, reactant_bonds))
        roles.append(0)
    molecules.append(
        mapped_molecule_to_graph(record["product"], center_atoms, product_bonds)
    )
    roles.append(1)
    return {
        "sample_id": record["sample_id"],
        "molecules": molecules,
        "roles": roles,
        "weight": float(record.get("weight", 1.0)),
    }


class ReactionJsonlDataset(IterableDataset):
    """Streams task-specific JSONL.GZ shards with shard-aware worker partitioning."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        shuffle: bool = False,
        shuffle_buffer: int = 2048,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.shuffle = shuffle
        self.shuffle_buffer = max(1, int(shuffle_buffer))
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
        self.world_size = (
            int(os.environ.get("WORLD_SIZE", "1"))
            if world_size is None
            else int(world_size)
        )
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Missing manifest: %s" % manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.shards = [self.data_dir / value for value in self.manifest["shards"][split]]
        self.count = int(self.manifest["counts"].get(split, 0))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assigned_shards(self) -> List[Path]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        global_worker = self.rank * num_workers + worker_id
        global_workers = self.world_size * num_workers
        shards = list(self.shards)
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(shards)
        return [path for index, path in enumerate(shards) if index % global_workers == global_worker]

    def _records(self) -> Iterator[Dict[str, Any]]:
        for shard in self._assigned_shards():
            with gzip.open(str(shard), "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        records = self._records()
        if not self.shuffle:
            for record in records:
                yield record_to_graphs(record)
            return
        rng = random.Random(self.seed + self.epoch + self.rank * 100003)
        buffer: List[Dict[str, Any]] = []
        for record in records:
            item = record_to_graphs(record)
            if len(buffer) < self.shuffle_buffer:
                buffer.append(item)
            else:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = item
        rng.shuffle(buffer)
        for item in buffer:
            yield item


def collate_reactions(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    xs: List[torch.Tensor] = []
    edge_indices: List[torch.Tensor] = []
    edge_attrs: List[torch.Tensor] = []
    bond_indices: List[torch.Tensor] = []
    bond_attrs: List[torch.Tensor] = []
    atom_targets: List[torch.Tensor] = []
    bond_targets: List[torch.Tensor] = []
    atom_to_molecule: List[torch.Tensor] = []
    molecule_to_reaction: List[int] = []
    molecule_roles: List[int] = []
    sample_ids: List[str] = []
    weights: List[float] = []
    atom_offset = 0
    molecule_index = 0

    for reaction_index, sample in enumerate(samples):
        sample_ids.append(sample["sample_id"])
        weights.append(float(sample["weight"]))
        for graph, role in zip(sample["molecules"], sample["roles"]):
            num_atoms = int(graph["x"].shape[0])
            xs.append(graph["x"])
            edge_indices.append(graph["edge_index"] + atom_offset)
            edge_attrs.append(graph["edge_attr"])
            bond_indices.append(graph["bond_index"] + atom_offset)
            bond_attrs.append(graph["bond_attr"])
            atom_targets.append(graph["atom_y"])
            bond_targets.append(graph["bond_y"])
            atom_to_molecule.append(
                torch.full((num_atoms,), molecule_index, dtype=torch.long)
            )
            molecule_to_reaction.append(reaction_index)
            molecule_roles.append(int(role))
            atom_offset += num_atoms
            molecule_index += 1

    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "bond_index": torch.cat(bond_indices, dim=1),
        "bond_attr": torch.cat(bond_attrs, dim=0),
        "atom_y": torch.cat(atom_targets, dim=0),
        "bond_y": torch.cat(bond_targets, dim=0),
        "atom_to_molecule": torch.cat(atom_to_molecule, dim=0),
        "molecule_to_reaction": torch.tensor(molecule_to_reaction, dtype=torch.long),
        "molecule_roles": torch.tensor(molecule_roles, dtype=torch.long),
        "reaction_weight": torch.tensor(weights, dtype=torch.float32),
        "num_molecules": molecule_index,
        "num_reactions": len(samples),
        "sample_ids": sample_ids,
    }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class CategoricalEncoder(nn.Module):
    def __init__(self, dimensions: Sequence[int], hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(size), hidden_dim) for size in dimensions]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.embeddings[0](features[:, 0])
        for index in range(1, len(self.embeddings)):
            output = output + self.embeddings[index](features[:, index])
        return output


class GINELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_h: torch.Tensor, edge_index: torch.Tensor, edge_h: torch.Tensor
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(node_h)
        if edge_index.numel() > 0:
            source, target = edge_index[0], edge_index[1]
            messages = F.relu(node_h[source] + edge_h)
            aggregated.index_add_(0, target, messages)
        updated = self.mlp((1.0 + self.eps) * node_h + aggregated)
        return self.norm(node_h + self.dropout(updated))


def mean_pool(values: torch.Tensor, index: torch.Tensor, groups: int) -> torch.Tensor:
    output = values.new_zeros((groups, values.shape[-1]))
    output.index_add_(0, index, values)
    counts = values.new_zeros((groups, 1))
    counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return output / counts.clamp_min(1.0)


class SharedGINEEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.atom_encoder = CategoricalEncoder(ATOM_FEATURE_DIMS, hidden_dim)
        self.bond_encoder = CategoricalEncoder(BOND_FEATURE_DIMS, hidden_dim)
        self.layers = nn.ModuleList(
            [GINELayer(hidden_dim, dropout) for _ in range(int(num_layers))]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_h = self.atom_encoder(x)
        edge_h = self.bond_encoder(edge_attr)
        for layer in self.layers:
            node_h = layer(node_h, edge_index, edge_h)
        return node_h, edge_h


class ReactionPretrainModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
        projection_dim: int = 128,
        temperature: float = 0.1,
        reactant_pool: str = "mean",
    ) -> None:
        super().__init__()
        if reactant_pool not in {"mean", "sum", "atom_mean"}:
            raise ValueError("reactant_pool must be 'mean', 'sum', or 'atom_mean'")
        self.temperature = float(temperature)
        self.reactant_pool = reactant_pool
        self.encoder = SharedGINEEncoder(hidden_dim, num_layers, dropout)
        self.atom_center_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.bond_center_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def _reaction_embeddings(
        self,
        molecule_h: torch.Tensor,
        molecule_to_reaction: torch.Tensor,
        molecule_roles: torch.Tensor,
        num_reactions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reactant_mask = molecule_roles == 0
        product_mask = molecule_roles == 1
        reactant_h = molecule_h.new_zeros((num_reactions, molecule_h.shape[-1]))
        reactant_h.index_add_(
            0, molecule_to_reaction[reactant_mask], molecule_h[reactant_mask]
        )
        if self.reactant_pool == "mean":
            counts = molecule_h.new_zeros((num_reactions, 1))
            counts.index_add_(
                0,
                molecule_to_reaction[reactant_mask],
                molecule_h.new_ones((int(reactant_mask.sum().item()), 1)),
            )
            reactant_h = reactant_h / counts.clamp_min(1.0)
        product_h = molecule_h.new_zeros((num_reactions, molecule_h.shape[-1]))
        product_h.index_add_(
            0, molecule_to_reaction[product_mask], molecule_h[product_mask]
        )
        return reactant_h, product_h

    def _atom_weighted_reaction_embeddings(
        self,
        node_h: torch.Tensor,
        atom_to_molecule: torch.Tensor,
        molecule_to_reaction: torch.Tensor,
        molecule_roles: torch.Tensor,
        num_reactions: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool all atoms on each reaction side without equal molecule weighting."""
        atom_to_reaction = molecule_to_reaction[atom_to_molecule]
        atom_roles = molecule_roles[atom_to_molecule]
        reactant_mask = atom_roles == 0
        product_mask = atom_roles == 1
        reactant_h = mean_pool(
            node_h[reactant_mask], atom_to_reaction[reactant_mask], num_reactions
        )
        product_h = mean_pool(
            node_h[product_mask], atom_to_reaction[product_mask], num_reactions
        )
        return reactant_h, product_h

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        node_h, _directed_edge_h = self.encoder(
            batch["x"], batch["edge_index"], batch["edge_attr"]
        )
        atom_logits = self.atom_center_head(node_h).squeeze(-1)
        if batch["bond_index"].numel() > 0:
            first, second = batch["bond_index"][0], batch["bond_index"][1]
            undirected_edge_h = self.encoder.bond_encoder(batch["bond_attr"])
            bond_repr = torch.cat(
                [
                    node_h[first] + node_h[second],
                    torch.abs(node_h[first] - node_h[second]),
                    undirected_edge_h,
                ],
                dim=-1,
            )
            bond_logits = self.bond_center_head(bond_repr).squeeze(-1)
        else:
            bond_logits = node_h.new_empty((0,))

        molecule_h = mean_pool(
            node_h,
            batch["atom_to_molecule"],
            int(batch["num_molecules"]),
        )
        if self.reactant_pool == "atom_mean":
            reactant_h, product_h = self._atom_weighted_reaction_embeddings(
                node_h,
                batch["atom_to_molecule"],
                batch["molecule_to_reaction"],
                batch["molecule_roles"],
                int(batch["num_reactions"]),
            )
        else:
            reactant_h, product_h = self._reaction_embeddings(
                molecule_h,
                batch["molecule_to_reaction"],
                batch["molecule_roles"],
                int(batch["num_reactions"]),
            )
        reactant_z = F.normalize(self.projector(reactant_h), dim=-1)
        product_z = F.normalize(self.projector(product_h), dim=-1)
        return {
            "atom_logits": atom_logits,
            "bond_logits": bond_logits,
            "reactant_z": reactant_z,
            "product_z": product_z,
            "molecule_h": molecule_h,
        }


def balanced_binary_loss(
    logits: torch.Tensor, targets: torch.Tensor, max_pos_weight: float = 20.0
) -> torch.Tensor:
    if logits.numel() == 0:
        return targets.new_tensor(0.0)
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives.item() > 0:
        ratio = (negatives / positives).clamp(min=1.0, max=max_pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=ratio)
    return F.binary_cross_entropy_with_logits(logits, targets)


def symmetric_info_nce(
    reactant_z: torch.Tensor, product_z: torch.Tensor, temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if reactant_z.shape[0] < 2:
        zero = reactant_z.sum() * 0.0
        return zero, reactant_z.new_tensor(float("nan"))
    logits = torch.matmul(reactant_z, product_z.t()) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )
    accuracy = 0.5 * (
        (logits.argmax(dim=1) == labels).float().mean()
        + (logits.argmax(dim=0) == labels).float().mean()
    )
    return loss, accuracy


def symmetric_recall_at_k(
    reactant_z: torch.Tensor, product_z: torch.Tensor, temperature: float, k: int
) -> torch.Tensor:
    if reactant_z.shape[0] < 2:
        return reactant_z.new_tensor(float("nan"))
    logits = torch.matmul(reactant_z, product_z.t()) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    actual_k = min(int(k), int(logits.shape[0]))
    row_hits = logits.topk(actual_k, dim=1).indices.eq(labels[:, None]).any(dim=1)
    column_hits = (
        logits.t().topk(actual_k, dim=1).indices.eq(labels[:, None]).any(dim=1)
    )
    return 0.5 * (row_hits.float().mean() + column_hits.float().mean())


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if logits.numel() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    true_positive = (predictions & truth).sum().item()
    false_positive = (predictions & ~truth).sum().item()
    false_negative = (~predictions & truth).sum().item()
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_pretrain_loss(
    model: ReactionPretrainModel,
    batch: Dict[str, Any],
    lambda_relation: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    outputs = model(batch)
    atom_loss = balanced_binary_loss(outputs["atom_logits"], batch["atom_y"])
    bond_loss = balanced_binary_loss(outputs["bond_logits"], batch["bond_y"])
    center_loss = 0.5 * (atom_loss + bond_loss)
    relation_loss, contrastive_accuracy = symmetric_info_nce(
        outputs["reactant_z"], outputs["product_z"], model.temperature
    )
    contrastive_recall10 = symmetric_recall_at_k(
        outputs["reactant_z"], outputs["product_z"], model.temperature, 10
    )
    total_loss = center_loss + float(lambda_relation) * relation_loss
    atom_scores = binary_metrics(outputs["atom_logits"].detach(), batch["atom_y"])
    bond_scores = binary_metrics(outputs["bond_logits"].detach(), batch["bond_y"])
    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "center_loss": float(center_loss.detach().cpu()),
        "atom_loss": float(atom_loss.detach().cpu()),
        "bond_loss": float(bond_loss.detach().cpu()),
        "relation_loss": float(relation_loss.detach().cpu()),
        "contrastive_top1": float(contrastive_accuracy.detach().cpu()),
        "contrastive_recall10": float(contrastive_recall10.detach().cpu()),
        "atom_f1": atom_scores["f1"],
        "bond_f1": bond_scores["f1"],
    }
    return total_loss, metrics, outputs


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def feature_schema() -> Dict[str, Any]:
    return {
        "atom_feature_names": [
            "atomic_number",
            "degree",
            "formal_charge",
            "hybridization",
            "aromatic",
            "in_ring",
            "chirality",
        ],
        "atom_feature_dims": ATOM_FEATURE_DIMS,
        "bond_feature_names": ["bond_type", "stereo", "conjugated", "in_ring"],
        "bond_feature_dims": BOND_FEATURE_DIMS,
        "atom_mapping_used_as_feature": False,
    }
