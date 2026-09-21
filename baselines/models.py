"""Standard GNN baselines used in the internal comparison."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.nn.models import AttentiveFP as PyGAttentiveFP

from gesa.features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from gesa.reaction_encoder import CategoricalEncoder, SharedGINEEncoder


class Head(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class GCN(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = dropout
        self.atom_encoder = CategoricalEncoder(ATOM_FEATURE_DIMS, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = Head(hidden, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = self.atom_encoder(batch["x"])
        for convolution, normalization in zip(self.convs, self.norms):
            updated = convolution(node_h, batch["edge_index"])
            node_h = normalization(node_h + F.dropout(F.gelu(updated), self.dropout, self.training))
        return self.head(global_mean_pool(node_h, batch["graph_index"]))


class GAT(nn.Module):
    def __init__(
        self, hidden: int = 256, layers: int = 5, heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.atom_encoder = CategoricalEncoder(ATOM_FEATURE_DIMS, hidden)
        self.bond_encoder = CategoricalEncoder(BOND_FEATURE_DIMS, hidden)
        self.convs = nn.ModuleList([
            GATConv(hidden, hidden // heads, heads=heads, concat=True, dropout=dropout,
                    edge_dim=hidden, add_self_loops=True, fill_value="mean")
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = Head(hidden, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = self.atom_encoder(batch["x"])
        edge_h = self.bond_encoder(batch["edge_attr"])
        for convolution, normalization in zip(self.convs, self.norms):
            updated = convolution(node_h, batch["edge_index"], edge_attr=edge_h)
            node_h = normalization(node_h + F.dropout(F.gelu(updated), self.dropout, self.training))
        return self.head(global_mean_pool(node_h, batch["graph_index"]))


def mean_pool(values: torch.Tensor, index: torch.Tensor, groups: int) -> torch.Tensor:
    output = values.new_zeros((groups, values.shape[-1]))
    output.index_add_(0, index, values)
    counts = values.new_zeros((groups, 1))
    counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return output / counts.clamp_min(1.0)


class GINE(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = SharedGINEEncoder(hidden, layers, dropout)
        self.head = Head(hidden, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h, _ = self.encoder(batch["x"], batch["edge_index"], batch["edge_attr"])
        return self.head(mean_pool(node_h, batch["graph_index"], int(batch["num_graphs"])))


class AttentiveFP(nn.Module):
    def __init__(
        self, hidden: int = 256, layers: int = 5, timesteps: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.atom_encoder = CategoricalEncoder(ATOM_FEATURE_DIMS, hidden)
        self.bond_encoder = CategoricalEncoder(BOND_FEATURE_DIMS, hidden)
        self.encoder = PyGAttentiveFP(
            in_channels=hidden, hidden_channels=hidden, out_channels=hidden,
            edge_dim=hidden, num_layers=layers, num_timesteps=timesteps, dropout=dropout,
        )
        self.head = Head(hidden, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_h = self.atom_encoder(batch["x"])
        edge_h = self.bond_encoder(batch["edge_attr"])
        graph_h = self.encoder(node_h, batch["edge_index"], edge_h, batch["graph_index"])
        return self.head(graph_h)


def build_model(name: str, config: Dict) -> nn.Module:
    common = config.get("model", {})
    hidden = int(common.get("hidden_dim", 256))
    layers = int(common.get("num_layers", 5))
    dropout = float(common.get("dropout", 0.1))
    if name == "GCN":
        return GCN(hidden, layers, dropout)
    if name == "GAT":
        return GAT(hidden, layers, int(common.get("heads", 4)), dropout)
    if name == "GINE":
        return GINE(hidden, layers, dropout)
    if name == "AttentiveFP":
        return AttentiveFP(hidden, layers, int(common.get("timesteps", 2)), dropout)
    raise ValueError(f"Unknown baseline {name!r}")
