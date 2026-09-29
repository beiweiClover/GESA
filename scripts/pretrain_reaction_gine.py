#!/usr/bin/env python3
"""Run the original two-phase ORD pretraining recipe from random initialization."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gesa.training.molecular import atomic_json, load_config, resolve, sha
from gesa.training.ord_trainer import run_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/training/reaction_gine.yaml'))
    p.add_argument('--data-dir', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--epochs', type=int, default=None, help='Override total epochs, e.g. for a smoke run')
    a = p.parse_args()
    cfg = load_config(a.config)
    out = resolve(a.output or cfg.pop('output_dir'))
    out.mkdir(parents=True, exist_ok=True)
    cfg['data_dir'] = str(resolve(a.data_dir or cfg['data_dir']))
    if a.device:
        cfg['device'] = a.device
    total = a.epochs or int(cfg['epochs'])
    initial_epochs = min(total, int(cfg.pop('initial_epochs', 10)))
    first = dict(cfg, epochs=initial_epochs, output_dir=str(out / 'initial'))
    atomic_json(out / 'initial_config.json', first)
    run_training(str(out / 'initial_config.json'))
    chosen = out / 'initial'
    if total > initial_epochs:
        continued = dict(cfg, epochs=total, output_dir=str(out / 'continued'),
                         resume_from=str(out / 'initial/last.pt'),
                         resume_learning_rate=1e-5, resume_minimum_learning_rate=1e-5, resume_warmup_ratio=0.)
        atomic_json(out / 'continued_config.json', continued)
        run_training(str(out / 'continued_config.json'))
        first_summary = json.loads((out / 'initial/summary.json').read_text(encoding='utf-8'))
        next_summary = json.loads((out / 'continued/summary.json').read_text(encoding='utf-8'))
        if next_summary['best_valid_loss'] < first_summary['best_valid_loss']:
            chosen = out / 'continued'
    for filename in ['best.pt', 'encoder.pt']:
        shutil.copy2(chosen / filename, out / filename)
    atomic_json(out / 'selected_checkpoint.json', {'selected_phase': chosen.name,
                'encoder_sha256': sha(out / 'encoder.pt'), 'total_epochs': total, 'seed': cfg['seed']})
    print('Selected encoder:', out / 'encoder.pt')


if __name__ == '__main__':
    main()
