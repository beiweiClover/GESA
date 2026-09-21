"""Task-GAT: the task-specific, edge-aware graph attention encoder."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS


class CategoricalEncoder(nn.Module):
    def __init__(self, dimensions: Sequence[int], hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(size), int(hidden_dim)) for size in dimensions]
        )
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.embeddings[0](features[:, 0])
        for index in range(1, len(self.embeddings)):
            output = output + self.embeddings[index](features[:, index])
        return output


class TaskGAT(nn.Module):
    def __init__(
        self, hidden_dim: int = 256, num_layers: int = 5,
        heads: int = 4, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.dropout = float(dropout)
        self.atom_encoder = CategoricalEncoder(ATOM_FEATURE_DIMS, hidden_dim)
        self.bond_encoder = CategoricalEncoder(BOND_FEATURE_DIMS, hidden_dim)
        self.convs = nn.ModuleList([
            GATConv(
                hidden_dim, hidden_dim // heads, heads=heads, concat=True,
                dropout=dropout, edge_dim=hidden_dim, add_self_loops=True,
                fill_value="mean",
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        node_h = self.atom_encoder(x)
        edge_h = self.bond_encoder(edge_attr)
        for convolution, normalization in zip(self.convs, self.norms):
            updated = convolution(node_h, edge_index, edge_attr=edge_h)
            node_h = normalization(
                node_h + F.dropout(F.gelu(updated), p=self.dropout, training=self.training)
            )
        return node_h


# Checkpoint keys use the development-stage class name only through module paths.
LegacyGATEncoder = TaskGAT
