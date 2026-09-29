"""Train the two-task ORD Shared-GINE pretraining model."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .ord_model import (
    ReactionJsonlDataset,
    ReactionPretrainModel,
    binary_metrics,
    collate_reactions,
    compute_pretrain_loss,
    feature_schema,
    model_parameter_count,
    move_batch_to_device,
)


PROJECT_DIR = Path(__file__).resolve().parents[3]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["config_path"] = str(config_path)
    config["data_dir"] = str(resolve_project_path(config["data_dir"]))
    config["output_dir"] = str(resolve_project_path(config["output_dir"]))
    if config.get("resume_from"):
        config["resume_from"] = str(resolve_project_path(config["resume_from"]))
    return config


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON without exposing a partially written progress file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def create_loader(
    config: Dict[str, Any], split: str, shuffle: bool
) -> tuple:
    dataset = ReactionJsonlDataset(
        config["data_dir"],
        split,
        shuffle=shuffle,
        shuffle_buffer=int(config.get("shuffle_buffer", 2048)),
        seed=int(config["seed"]),
    )
    num_workers = int(config.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=num_workers,
        collate_fn=collate_reactions,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(shuffle),
        persistent_workers=bool(config.get("persistent_workers", False))
        and num_workers > 0,
    )
    return dataset, loader


def average_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {
        key: float(np.nanmean([row[key] for row in rows])) for key in keys
    }


def binary_average_precision(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact average precision for one validation split."""
    if logits.numel() == 0:
        return float("nan")
    truth = (targets >= 0.5).float()
    positives = float(truth.sum())
    if positives <= 0:
        return float("nan")
    order = torch.argsort(logits, descending=True)
    sorted_truth = truth[order]
    cumulative_true = sorted_truth.cumsum(0)
    ranks = torch.arange(1, sorted_truth.numel() + 1, dtype=torch.float32)
    precision = cumulative_true / ranks
    return float((precision * sorted_truth).sum() / positives)


def make_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_ratio: float,
    minimum_learning_rate: float,
    base_learning_rate: float,
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = int(round(total_steps * float(warmup_ratio)))
    minimum_ratio = max(
        0.0, min(1.0, float(minimum_learning_rate) / float(base_learning_rate))
    )

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def run_epoch(
    model: ReactionPretrainModel,
    loader: DataLoader,
    device: torch.device,
    lambda_relation: float,
    optimizer: Optional[AdamW] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[LambdaLR] = None,
    max_steps: Optional[int] = None,
    grad_clip: float = 5.0,
    log_every_steps: int = 0,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_rows: List[Dict[str, float]] = []
    atom_logits_all: List[torch.Tensor] = []
    atom_targets_all: List[torch.Tensor] = []
    bond_logits_all: List[torch.Tensor] = []
    bond_targets_all: List[torch.Tensor] = []
    started = time.time()
    reactions_seen = 0
    batches = loader if max_steps is None else itertools.islice(loader, int(max_steps))
    for step, cpu_batch in enumerate(batches, 1):
        batch = move_batch_to_device(cpu_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(scaler is not None and device.type == "cuda")
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss, metrics, outputs = compute_pretrain_loss(
                    model, batch, lambda_relation=lambda_relation
                )
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                atom_logits_all.append(outputs["atom_logits"].detach().float().cpu())
                atom_targets_all.append(batch["atom_y"].detach().float().cpu())
                bond_logits_all.append(outputs["bond_logits"].detach().float().cpu())
                bond_targets_all.append(batch["bond_y"].detach().float().cpu())
        metrics["batch_size"] = float(batch["num_reactions"])
        if training:
            metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        metric_rows.append(metrics)
        reactions_seen += int(batch["num_reactions"])
        if training and log_every_steps > 0 and step % log_every_steps == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "train_step": step,
                        "loss": metrics["loss"],
                        "center_loss": metrics["center_loss"],
                        "relation_loss": metrics["relation_loss"],
                        "learning_rate": metrics["learning_rate"],
                        "reactions_per_second": round(reactions_seen / elapsed, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    averaged = average_metrics(metric_rows)
    elapsed = max(time.time() - started, 1e-6)
    averaged["steps"] = float(len(metric_rows))
    averaged["reactions"] = float(reactions_seen)
    averaged["elapsed_seconds"] = float(elapsed)
    averaged["reactions_per_second"] = float(reactions_seen / elapsed)
    if not training and atom_logits_all:
        atom_logits = torch.cat(atom_logits_all)
        atom_targets = torch.cat(atom_targets_all)
        bond_logits = torch.cat(bond_logits_all)
        bond_targets = torch.cat(bond_targets_all)
        averaged["atom_ap"] = binary_average_precision(atom_logits, atom_targets)
        averaged["bond_ap"] = binary_average_precision(bond_logits, bond_targets)
        averaged["atom_f1"] = binary_metrics(atom_logits, atom_targets)["f1"]
        averaged["bond_f1"] = binary_metrics(bond_logits, bond_targets)["f1"]
    return averaged


def save_checkpoint(
    path: Path,
    model: ReactionPretrainModel,
    optimizer: AdamW,
    epoch: int,
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
    scheduler: Optional[LambdaLR] = None,
    global_step: int = 0,
) -> None:
    payload = {
        "format": "ord-shared-gine-pretrain-v1",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "global_step": int(global_step),
        "model_config": config["model"],
        "feature_schema": feature_schema(),
        "training_config": config,
        "history": history,
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, str(path))


def save_encoder(
    path: Path,
    model: ReactionPretrainModel,
    epoch: int,
    config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "format": "ord-shared-gine-encoder-v1",
            "epoch": int(epoch),
            "encoder_state_dict": model.encoder.state_dict(),
            "encoder_config": {
                "hidden_dim": config["model"]["hidden_dim"],
                "num_layers": config["model"]["num_layers"],
                "dropout": config["model"]["dropout"],
            },
            "feature_schema": feature_schema(),
            "torch_version": str(torch.__version__),
        },
        str(path),
    )


def run_training(config_path: str) -> Dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader = create_loader(config, "train", shuffle=True)
    valid_dataset, valid_loader = create_loader(config, "valid", shuffle=False)
    model = ReactionPretrainModel(**config["model"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    steps_per_epoch = (
        int(config["max_train_steps"])
        if config.get("max_train_steps") is not None
        else math.ceil(train_dataset.count / int(config["batch_size"]))
    )
    epochs = int(config["epochs"])
    total_train_steps = epochs * steps_per_epoch
    history: List[Dict[str, Any]] = []
    best_valid = float("inf")
    global_step = 0
    start_epoch = 1
    resume_from = config.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        checkpoint = torch.load(str(resume_path), map_location=device, weights_only=False)
        if checkpoint.get("model_config") != config["model"]:
            raise ValueError(
                "Resume model_config does not match current config: "
                f"{checkpoint.get('model_config')} != {config['model']}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        completed_epoch = int(checkpoint["epoch"])
        if len(history) != completed_epoch:
            raise ValueError(
                f"Resume history has {len(history)} rows for epoch {completed_epoch}"
            )
        if completed_epoch >= epochs:
            raise ValueError(
                f"Checkpoint epoch {completed_epoch} already reaches target {epochs}"
            )
        start_epoch = completed_epoch + 1
        global_step = int(checkpoint.get("global_step", completed_epoch * steps_per_epoch))
        best_valid = min(
            (float(row["valid"]["loss"]) for row in history),
            default=float("inf"),
        )
        resume_learning_rate = float(
            config.get(
                "resume_learning_rate", optimizer.param_groups[0]["lr"]
            )
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = resume_learning_rate
            parameter_group["initial_lr"] = resume_learning_rate
        remaining_steps = (epochs - completed_epoch) * steps_per_epoch
        scheduler = make_scheduler(
            optimizer,
            total_steps=remaining_steps,
            warmup_ratio=float(config.get("resume_warmup_ratio", 0.0)),
            minimum_learning_rate=float(
                config.get("resume_minimum_learning_rate", resume_learning_rate)
            ),
            base_learning_rate=resume_learning_rate,
        )
    else:
        scheduler = make_scheduler(
            optimizer,
            total_steps=total_train_steps,
            warmup_ratio=float(config.get("warmup_ratio", 0.0)),
            minimum_learning_rate=float(config.get("minimum_learning_rate", 0.0)),
            base_learning_rate=float(config["learning_rate"]),
        )
    use_amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "parameters": model_parameter_count(model),
                "train_samples": train_dataset.count,
                "valid_samples": valid_dataset.count,
                "output_dir": str(output_dir),
                "steps_per_epoch": steps_per_epoch,
                "total_train_steps": total_train_steps,
                "start_epoch": start_epoch,
                "resume_from": resume_from,
                "current_learning_rate": optimizer.param_groups[0]["lr"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if resume_from:
        initial_summary = {
            "status": "running",
            "last_completed_epoch": start_epoch - 1,
            "target_epochs": epochs,
            "device": str(device),
            "parameters": model_parameter_count(model),
            "epochs": epochs,
            "best_valid_loss": best_valid,
            "output_dir": str(output_dir),
            "resume_from": resume_from,
            "history": history,
        }
        write_json_atomic(output_dir / "history.json", history)
        write_json_atomic(output_dir / "summary.json", initial_summary)

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            float(config["lambda_relation"]),
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            max_steps=config.get("max_train_steps"),
            grad_clip=float(config.get("grad_clip", 5.0)),
            log_every_steps=int(config.get("log_every_steps", 0)),
        )
        global_step += int(train_metrics.get("steps", 0))
        with torch.no_grad():
            valid_metrics = run_epoch(
                model,
                valid_loader,
                device,
                float(config["lambda_relation"]),
                optimizer=None,
                scaler=None,
                max_steps=config.get("max_valid_steps"),
            )
        epoch_row = {
            "epoch": epoch,
            "elapsed_seconds": round(time.time() - started, 3),
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False))
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            config,
            history,
            scheduler=scheduler,
            global_step=global_step,
        )
        if bool(config.get("save_each_epoch", False)):
            save_checkpoint(
                output_dir / ("epoch-%03d.pt" % epoch),
                model,
                optimizer,
                epoch,
                config,
                history,
                scheduler=scheduler,
                global_step=global_step,
            )
            save_encoder(
                output_dir / ("encoder-epoch-%03d.pt" % epoch),
                model,
                epoch,
                config,
            )
        valid_loss = float(valid_metrics.get("loss", float("inf")))
        if valid_loss < best_valid:
            best_valid = valid_loss
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                history,
                scheduler=scheduler,
                global_step=global_step,
            )
            save_encoder(output_dir / "encoder.pt", model, epoch, config)
        write_json_atomic(output_dir / ("metrics-epoch-%03d.json" % epoch), epoch_row)
        write_json_atomic(output_dir / "history.json", history)
        summary = {
            "status": "completed" if epoch == epochs else "running",
            "last_completed_epoch": epoch,
            "target_epochs": epochs,
            "device": str(device),
            "parameters": model_parameter_count(model),
            "epochs": epochs,
            "best_valid_loss": best_valid,
            "output_dir": str(output_dir),
            "resume_from": resume_from,
            "history": history,
        }
        write_json_atomic(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_training(args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
