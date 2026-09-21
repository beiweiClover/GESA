"""GESA model with legacy checkpoint-compatible module paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from .attentive_pool import AttentivePool
from .reaction_encoder import ReactionGINE
from .sa_priors import XSAEncoder
from .source_gate import SourceGate
from .task_encoder import TaskGAT


class Projection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GESA(nn.Module):
    """Generative-space Expanded Synthetic Accessibility model."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 5,
        gat_heads: int = 4,
        encoder_dropout: float = 0.1,
        nonencoder_dropout: float = 0.3,
        projection_dim: int = 256,
        gate_hidden_dim: int = 128,
        attention_hidden_dim: int = 128,
        xsa_hidden_dim: int = 64,
        xsa_output_dim: int = 32,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        # Attribute names deliberately match the development checkpoint schema.
        self.task_encoder = TaskGAT(hidden_dim, num_layers, gat_heads, encoder_dropout)
        self.task_projection = Projection(hidden_dim, projection_dim, nonencoder_dropout)
        self.ord_encoder = ReactionGINE(hidden_dim, num_layers, encoder_dropout)
        self.ord_projection = Projection(hidden_dim, projection_dim, nonencoder_dropout)
        self.fusion = SourceGate(projection_dim, gate_hidden_dim, nonencoder_dropout)
        self.pool = AttentivePool(projection_dim, attention_hidden_dim)
        self.xsa_encoder = XSAEncoder(14, xsa_hidden_dim, xsa_output_dim, nonencoder_dropout)
        self.head = nn.Sequential(
            nn.Linear(projection_dim + xsa_output_dim, head_hidden_dim), nn.GELU(),
            nn.Dropout(nonencoder_dropout), nn.Linear(head_hidden_dim, 1),
        )

    @staticmethod
    def _assert_graph(batch: Dict[str, torch.Tensor]) -> None:
        if batch["x"].ndim != 2 or batch["x"].shape[1] != 7:
            raise ValueError("GESA expects seven categorical atom features")
        if batch["edge_index"].ndim != 2 or batch["edge_index"].shape[0] != 2:
            raise ValueError("GESA expects edge_index with shape [2, E]")
        if batch["edge_attr"].ndim != 2 or batch["edge_attr"].shape[1] != 4:
            raise ValueError("GESA expects four categorical bond features")
        if "xsa" not in batch or batch["xsa"].shape[-1] != 14:
            raise ValueError("GESA requires fourteen standardized explicit SA priors")

    def forward(
        self, batch: Dict[str, torch.Tensor], return_explanations: bool = False
    ):
        self._assert_graph(batch)
        task_h = self.task_encoder(batch["x"], batch["edge_index"], batch["edge_attr"])
        reaction_h = self.ord_encoder(batch["x"], batch["edge_index"], batch["edge_attr"])
        if task_h.shape != reaction_h.shape:
            raise RuntimeError("Task-GAT and Reaction-GINE atom representations are not aligned")
        task_view = self.task_projection(task_h)
        reaction_view = self.ord_projection(reaction_h)
        fused, gate = self.fusion(task_view, reaction_view)
        graph_h, alpha = self.pool(
            fused, batch["graph_index"], int(batch["num_graphs"].item())
        )
        graph_h = torch.cat([graph_h, self.xsa_encoder(batch["xsa"])], dim=-1)
        logits = self.head(graph_h).squeeze(-1)
        if return_explanations:
            return {"logits": logits, "alpha": alpha, "gate": gate}
        return logits

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.task_encoder.parameters()
        yield from self.ord_encoder.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        encoder_ids = {id(parameter) for parameter in self.encoder_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in encoder_ids:
                yield parameter

    def load_release_checkpoint(
        self, checkpoint: str | Path, map_location: str | torch.device = "cpu"
    ) -> Dict[str, Any]:
        payload = torch.load(Path(checkpoint), map_location=map_location)
        state = payload.get("model_state_dict", payload)
        incompatible = self.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("GESA checkpoint strict load was not exact")
        return payload

    def describe(self) -> Dict[str, Any]:
        return {
            "model": "GESA",
            "full_name": "Generative-space Expanded Synthetic Accessibility",
            "task_encoder": "Task-GAT (edge-aware)",
            "reaction_encoder": "Reaction-GINE (edge-aware)",
            "fusion": "atom-wise scalar source gating",
            "readout": "attentive molecular pooling",
            "explicit_sa_priors": 14,
            "positive_class": "HS (label=1)",
            "negative_class": "ES (label=0)",
            "inference_score": "P(HS)",
            "default_decision_rule": "P(HS) >= 0.5 => HS",
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }
