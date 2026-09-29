#!/usr/bin/env python3
"""Pretrain Task-GAT from random initialization on the fixed legacy SA splits."""
from pathlib import Path
import argparse
import sys

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gesa.task_encoder import TaskGAT
from gesa.training.molecular import MoleculeDataset, fit, load_config, resolve, seed_all


class TaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, head = int(config['hidden_dim']), int(config['head_hidden_dim'])
        dropout = float(config['head_dropout'])
        self.encoder = TaskGAT(hidden, int(config['num_layers']), int(config['gat_heads']), float(config['dropout']))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, head), nn.GELU(), nn.Dropout(dropout), nn.Linear(head, 1))

    def forward(self, batch):
        h = self.encoder(batch['x'], batch['edge_index'], batch['edge_attr'])
        return self.head(global_mean_pool(h, batch['graph_index'])).squeeze(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/training/taskgat.yaml'))
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    seed = a.seed if a.seed is not None else int(cfg.get('seed', 123))
    seed_all(seed)
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    data = cfg['data']
    train = MoleculeDataset(resolve(data['train']), cache_dir=data.get('cache_dir'))
    valid = MoleculeDataset(resolve(data['validation']), cache_dir=data.get('cache_dir'))
    model = TaskClassifier(cfg['model'])
    result = fit(model, train, valid, cfg['training'], resolve(a.output or cfg['output'].format(seed=seed)),
                 seed, device, [model.encoder], encoder_only=model.encoder,
                 extras={'stage': 'Task-GAT pretraining', 'configuration': cfg})
    print(result)


if __name__ == '__main__':
    main()
