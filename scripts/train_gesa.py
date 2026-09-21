#!/usr/bin/env python3
"""Validation-only GESA fine-tuning on user-provided CSV splits.

This release script never opens TS1–TS4. The default initialization is the released
GESA checkpoint; exact from-scratch reproduction additionally requires the documented
Task-GAT and Reaction-GINE pretraining corpora.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gesa.graph import collate_graphs, move_batch, smiles_to_graph  # noqa: E402
from gesa.model import GESA  # noqa: E402
from gesa.sa_priors import (  # noqa: E402
    SCALER_MEAN, SCALER_STD, load_fragment_scores, raw_sa_priors,
)


class MoleculeDataset(Dataset):
    def __init__(self, path: str, smiles_column: str, label_column: str) -> None:
        frame = pd.read_csv(path, usecols=[smiles_column, label_column], keep_default_na=False)
        self.smiles = frame[smiles_column].astype(str).tolist()
        self.labels = frame[label_column].to_numpy(np.float32)
        if not set(np.unique(self.labels)).issubset({0.0, 1.0}):
            raise ValueError("Labels must be binary 0/1")
        self.fragment_scores = load_fragment_scores()

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int):
        molecule = Chem.MolFromSmiles(self.smiles[index])
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid molecule at row {index}")
        raw = raw_sa_priors(molecule, self.fragment_scores)
        standardized = ((raw - SCALER_MEAN) / SCALER_STD).astype(np.float32)
        return smiles_to_graph(self.smiles[index]), torch.from_numpy(standardized), float(self.labels[index])


def collate_training(samples):
    graphs, priors, labels = zip(*samples)
    batch = collate_graphs(graphs, torch.stack(priors))
    batch["labels"] = torch.tensor(labels, dtype=torch.float32)
    return batch


@torch.no_grad()
def validation_auroc(model: GESA, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    labels, scores = [], []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        scores.append(torch.sigmoid(model(batch)).float().cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(scores)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/fine-tune the GESA synthetic accessibility model")
    parser.add_argument("--config", default=str(ROOT / "configs" / "gesa.yaml"))
    parser.add_argument("--initial-checkpoint", default=str(ROOT / "weights" / "GESA.pt"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "GESA_validation_best.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = config["data"]
    if not data.get("train") or not data.get("validation"):
        raise ValueError("Set data.train and data.validation in configs/gesa.yaml")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    settings = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = MoleculeDataset(data["train"], data["smiles_column"], data["label_column"])
    validation_data = MoleculeDataset(
        data["validation"], data["smiles_column"], data["label_column"]
    )
    train_loader = DataLoader(
        train_data, batch_size=int(settings["batch_size"]), shuffle=True,
        num_workers=0, collate_fn=collate_training,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=int(settings["evaluation_batch_size"]), shuffle=False,
        num_workers=0, collate_fn=collate_training,
    )

    model = GESA().to(device)
    if args.initial_checkpoint:
        model.load_release_checkpoint(args.initial_checkpoint, map_location="cpu")
        model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder_parameters()), "lr": float(settings["encoder_learning_rate"])},
        {"params": list(model.head_parameters()), "lr": float(settings["head_learning_rate"])},
    ], weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(settings["scheduler_factor"]),
        patience=int(settings["scheduler_patience"]),
    )
    positive = float(train_data.labels.sum())
    negative = float(len(train_data.labels) - positive)
    positive_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    best, best_epoch, stale = float("-inf"), 0, 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []

    for epoch in range(1, int(settings["max_epochs"]) + 1):
        warmup = epoch <= int(settings["warmup_epochs"])
        for parameter in model.encoder_parameters():
            parameter.requires_grad_(not warmup)
        model.train()
        total_loss, total_rows = 0.0, 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["labels"], pos_weight=positive_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["labels"])
            total_rows += len(batch["labels"])
        auc = validation_auroc(model, validation_loader, device)
        scheduler.step(auc)
        improved = auc > best + float(settings["min_delta_auroc"])
        if improved:
            best, best_epoch, stale = auc, epoch, 0
            torch.save({
                "format": "GESA-release-validation-best-v1", "model": "GESA",
                "seed": args.seed, "epoch": epoch, "validation_auroc": auc,
                "selection_criterion": "internal_validation_auroc_only",
                "external_metrics_consulted": False,
                "model_state_dict": copy.deepcopy(model.state_dict()),
            }, output)
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_bce": total_loss / max(total_rows, 1),
            "validation_auroc": auc, "best_epoch": best_epoch,
            "best_validation_auroc": best, "stale": stale,
        })
        print(json.dumps(history[-1]), flush=True)
        if epoch > int(settings["warmup_epochs"]) and stale >= int(settings["patience"]):
            break
    pd.DataFrame(history).to_csv(output.with_suffix(".history.csv"), index=False)
    print(f"Validation-best checkpoint: {output} (epoch={best_epoch}, AUROC={best:.6f})")


if __name__ == "__main__":
    main()
