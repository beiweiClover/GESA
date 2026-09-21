"""Atom-wise evidence fusion used by GESA-Core and GESA."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SourceGate(nn.Module):
    def __init__(self, hidden_dim: int = 256, gate_hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, gate_hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(gate_hidden_dim, 1),
        )

    def forward(
        self, task_h: torch.Tensor, reaction_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        evidence = torch.cat(
            [task_h, reaction_h, torch.abs(task_h - reaction_h), task_h * reaction_h],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(evidence)).squeeze(-1)
        fused = gate.unsqueeze(-1) * task_h + (1.0 - gate).unsqueeze(-1) * reaction_h
        return fused, gate
