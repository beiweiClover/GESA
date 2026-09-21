"""Reaction-GINE: the reaction-informed message-passing encoder."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class GINELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_h: torch.Tensor, edge_index: torch.Tensor, edge_h: torch.Tensor
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(node_h)
        if edge_index.numel():
            source, target = edge_index[0], edge_index[1]
            aggregated.index_add_(0, target, F.relu(node_h[source] + edge_h))
        updated = self.mlp((1.0 + self.eps) * node_h + aggregated)
        return self.norm(node_h + self.dropout(updated))


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


class ReactionGINE(nn.Module):
    """Adapter name is public; the nested ``encoder`` preserves checkpoint keys."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = SharedGINEEncoder(hidden_dim, num_layers, dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        node_h, _ = self.encoder(x, edge_index, edge_attr)
        return node_h
