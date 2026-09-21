#!/usr/bin/env python3
"""Train one standard baseline with validation-AUROC checkpoint selection."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines"))

from gesa.graph import collate_graphs, move_batch, smiles_to_graph  # noqa: E402
from models import build_model  # noqa: E402


class CSVGraphs(Dataset):
    def __init__(self, path: str, smiles_column: str, label_column: str) -> None:
        frame = pd.read_csv(path, usecols=[smiles_column, label_column], keep_default_na=False)
        self.smiles = frame[smiles_column].astype(str).tolist()
        self.labels = frame[label_column].to_numpy(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        return smiles_to_graph(self.smiles[index]), float(self.labels[index])


def collate(samples):
    graphs, labels = zip(*samples)
    batch = collate_graphs(graphs)
    batch["labels"] = torch.tensor(labels, dtype=torch.float32)
    return batch


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(torch.sigmoid(model(batch)).cpu().tolist())
    return float(roc_auc_score(labels, predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GESA release baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data, training = config["data"], config["training"]
    train = CSVGraphs(data["train"], data["smiles_column"], data["label_column"])
    validation = CSVGraphs(data["validation"], data["smiles_column"], data["label_column"])
    train_loader = DataLoader(train, int(training["batch_size"]), shuffle=True,
                              num_workers=0, collate_fn=collate)
    validation_loader = DataLoader(validation, int(training["evaluation_batch_size"]),
                                   shuffle=False, num_workers=0, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["name"], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    best, stale = float("-inf"), 0
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), batch["labels"])
            loss.backward()
            optimizer.step()
        auc = score(model, validation_loader, device)
        if auc > best + float(training["min_delta_auroc"]):
            best, stale = auc, 0
            torch.save({"model": config["name"], "epoch": epoch,
                        "validation_auroc": auc,
                        "selection_criterion": "internal_validation_auroc_only",
                        "model_state_dict": copy.deepcopy(model.state_dict())}, args.output)
        else:
            stale += 1
        print(f"epoch={epoch} validation_AUROC={auc:.6f} best={best:.6f}", flush=True)
        if stale >= int(training["patience"]):
            break


if __name__ == "__main__":
    main()
