#!/usr/bin/env python3
"""Train GESA from pretrained encoders, or fine-tune an existing GESA checkpoint."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gesa.model import GESA
from gesa.sa_priors import SCALER_MEAN, SCALER_STD
from gesa.training.molecular import MoleculeDataset, fit, load_config, resolve, seed_all, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/training/gesa.yaml'))
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default=None)
    p.add_argument('--dataset-version', choices=['original', 'single_component', 'custom'])
    p.add_argument('--initialization', choices=['pretrained', 'checkpoint', 'random'])
    p.add_argument('--initial-checkpoint', default=None)
    p.add_argument('--task-checkpoint', default=None)
    p.add_argument('--reaction-checkpoint', default=None)
    p.add_argument('--output', default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    seed = a.seed if a.seed is not None else int(cfg.get('seed', 123))
    seed_all(seed)
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    version = a.dataset_version or cfg.get('dataset_version', 'custom')
    data = cfg['data'].get(version, cfg['data'])
    if not data.get('train') or not data.get('validation'):
        raise ValueError('Set the training and validation paths in the configuration')
    if cfg.get('verify_dataset_hashes', False) and version != 'custom':
        versions = json.loads((ROOT / 'data/retrostar_target/version_patch/dataset_versions.json').read_text())
        key = 'original' if version == 'original' else 'released'
        for split in ['train', 'validation']:
            if sha(resolve(data[split])) != versions['splits'][split][key]['sha256']:
                raise ValueError(f'{split} does not match the declared {version} data version')
    train = MoleculeDataset(resolve(data['train']), cache_dir=cfg.get('cache_dir'), priors=True)
    valid = MoleculeDataset(resolve(data['validation']), cache_dir=cfg.get('cache_dir'), priors=True)
    architecture = {k: v for k, v in cfg.get('architecture', {}).items() if k != 'xsa_input_dim'}
    model = GESA(**architecture)
    mode = a.initialization or ('checkpoint' if a.initial_checkpoint else cfg.get('initialization', 'pretrained'))
    initial = {'mode': mode}
    if mode == 'checkpoint':
        path = resolve(a.initial_checkpoint or 'weights/GESA.pt')
        model.load_release_checkpoint(path, map_location='cpu')
        initial.update(checkpoint=str(path), sha256=sha(path))
    elif mode == 'pretrained':
        task = resolve(a.task_checkpoint or cfg.get('task_checkpoint', 'weights/pretrained/TaskGAT_seed{seed}.pt').format(seed=seed))
        reaction = resolve(a.reaction_checkpoint or cfg.get('reaction_checkpoint', 'weights/pretrained/ReactionGINE_epoch23.pt'))
        if not task.exists():
            raise FileNotFoundError(f'Pretrain Task-GAT for seed {seed} and pass --task-checkpoint, or use the released seed-123 initialization')
        task_payload = torch.load(task, map_location='cpu', weights_only=False)
        reaction_payload = torch.load(reaction, map_location='cpu', weights_only=False)
        model.task_encoder.load_state_dict(task_payload['encoder_state_dict'], strict=True)
        model.ord_encoder.encoder.load_state_dict(reaction_payload['encoder_state_dict'], strict=True)
        initial.update(task_checkpoint=str(task), task_sha256=sha(task),
                       reaction_checkpoint=str(reaction), reaction_sha256=sha(reaction))
    scaling = cfg.get('scaling', {}).get(version, 'fit_training')
    if mode == 'checkpoint':
        mean = getattr(model, 'sa_prior_mean', None)
        std = getattr(model, 'sa_prior_std', None)
        mean, std = (SCALER_MEAN, SCALER_STD) if mean is None else (mean, std)
    elif scaling == 'published_training':
        mean, std = SCALER_MEAN, SCALER_STD
    else:
        raw = np.asarray(train.raw, dtype=np.float64)
        mean, std = raw.mean(0), raw.std(0)
        std[std < 1e-12] = 1.
    train.standardize(mean, std)
    valid.standardize(mean, std)
    model.sa_prior_mean, model.sa_prior_std = np.asarray(mean), np.asarray(std)
    output = resolve(a.output or cfg.get('output', 'runs/gesa/{version}/seed_{seed}').format(version=version, seed=seed))
    result = fit(model, train, valid, cfg['training'], output, seed, device,
                 [model.task_encoder, model.ord_encoder], freeze_epochs=int(cfg['training']['warmup_epochs']),
                 extras={'stage': 'GESA', 'dataset_version': version, 'initialization': initial,
                         'sa_prior_scaler': {'mean': np.asarray(mean).tolist(), 'std': np.asarray(std).tolist()},
                         'configuration': cfg})
    print(result)


if __name__ == '__main__':
    main()
