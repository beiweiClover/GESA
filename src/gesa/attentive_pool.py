"""Attentive molecular readout."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import softmax


class AttentivePool(nn.Module):
    def __init__(self, hidden_dim: int = 256, attention_dim: int = 128) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(
        self, node_h: torch.Tensor, graph_index: torch.Tensor, num_graphs: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(node_h).squeeze(-1).float()
        alpha = softmax(scores, graph_index, num_nodes=int(num_graphs))
        weighted = alpha.unsqueeze(-1) * node_h.float()
        pooled = weighted.new_zeros((int(num_graphs), node_h.shape[-1]))
        pooled.index_add_(0, graph_index, weighted)
        return pooled, alpha
