"""Shared molecular data and optimization utilities for the two SA stages."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem, rdBase
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

from ..graph import smiles_to_graph, collate_graphs, move_batch
from ..sa_priors import load_fragment_scores, raw_sa_priors

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


class MoleculeDataset(Dataset):
    def __init__(self, path, smiles='canonical_smiles', label='target_hard', cache_dir=None, priors=False):
        self.path = Path(path)
        self.frame = pd.read_csv(path, usecols=[smiles, label], keep_default_na=False)
        self.smiles = self.frame[smiles].astype(str).tolist()
        self.labels = self.frame[label].to_numpy(dtype=np.float32)
        if not set(np.unique(self.labels)).issubset({0., 1.}):
            raise ValueError('Labels must be ES=0 / HS=1')
        if len(np.unique(self.labels)) != 2:
            raise ValueError('Training and validation require both classes')
        self.source_sha256 = sha(self.path)
        self.use_priors = priors
        self.raw = self.normalized = None
        self.store = None
        self.cache = None
        if cache_dir:
            schema = sha(Path(__file__).parent.parent / 'features.py') + sha(Path(__file__).parent.parent / 'sa_priors.py')
            key = hashlib.sha256((self.source_sha256 + rdBase.rdkitVersion + schema + str(priors)).encode()).hexdigest()[:24]
            self.cache = resolve(cache_dir) / key
            if not (self.cache / 'manifest.json').exists():
                self._build_cache()
            self._open_cache()
            if priors:
                self.raw = np.load(self.cache / 'raw.npy', mmap_mode='r')
        elif priors:
            scores = load_fragment_scores()
            rows = []
            for i, s in enumerate(self.smiles):
                m = Chem.MolFromSmiles(s)
                if m is None or m.GetNumAtoms() == 0:
                    raise ValueError(f'Invalid molecule at row {i}: {self.path}')
                rows.append(raw_sa_priors(m, scores))
            self.raw = np.stack(rows)

    def _build_cache(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        xs, edges, attrs, raw = [], [], [], []
        nptr, eptr = [0], [0]
        scores = load_fragment_scores() if self.use_priors else None
        for i, s in enumerate(self.smiles):
            graph = smiles_to_graph(s)
            xs.append(graph['x'].numpy().astype('uint8'))
            edges.append(graph['edge_index'].numpy().astype('int32'))
            attrs.append(graph['edge_attr'].numpy().astype('uint8'))
            nptr.append(nptr[-1] + len(xs[-1]))
            eptr.append(eptr[-1] + edges[-1].shape[1])
            if self.use_priors:
                raw.append(raw_sa_priors(Chem.MolFromSmiles(s), scores))
            if (i + 1) % 50000 == 0:
                print(f'Prepared {i+1}/{len(self.smiles)} molecular graphs', flush=True)
        arrays = {'x': np.concatenate(xs), 'edge_index': np.concatenate(edges, axis=1),
                  'edge_attr': np.concatenate(attrs), 'node_ptr': np.asarray(nptr), 'edge_ptr': np.asarray(eptr)}
        if raw:
            arrays['raw'] = np.stack(raw)
        for name, array in arrays.items():
            np.save(self.cache / (name + '.npy'), array)
        atomic_json(self.cache / 'manifest.json', {'input_sha256': self.source_sha256,
                    'rdkit_version': rdBase.rdkitVersion, 'rows': len(self.labels)})

    def _open_cache(self):
        self.store = {k: np.load(self.cache / (k + '.npy'), mmap_mode='r')
                      for k in ['x', 'edge_index', 'edge_attr', 'node_ptr', 'edge_ptr']}

    def __getstate__(self):
        d = self.__dict__.copy()
        d['store'] = None
        return d

    def standardize(self, mean, std):
        self.normalized = ((self.raw - np.asarray(mean)) / np.asarray(std)).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.cache:
            if self.store is None:
                self._open_cache()
            s = self.store
            a, b = s['node_ptr'][i:i+2]
            c, d = s['edge_ptr'][i:i+2]
            graph = {'x': torch.from_numpy(s['x'][a:b].astype('int64')),
                     'edge_index': torch.from_numpy(s['edge_index'][:, c:d].astype('int64')),
                     'edge_attr': torch.from_numpy(s['edge_attr'][c:d].astype('int64'))}
        else:
            graph = smiles_to_graph(self.smiles[i])
        prior = self.normalized[i] if self.normalized is not None else None
        return graph, prior, float(self.labels[i])


def collate(samples):
    graphs, priors, labels = zip(*samples)
    xsa = torch.from_numpy(np.stack(priors)) if priors[0] is not None else None
    batch = collate_graphs(graphs, xsa)
    batch['labels'] = torch.tensor(labels, dtype=torch.float32)
    return batch


class HistoricalOrder(IterableDataset):
    """Original 4096-row shard order and 16-shard interleaving without duplicate graph files."""
    def __init__(self, dataset, seed, shuffle, shard_size=4096, block_size=16):
        self.dataset, self.seed, self.shuffle = dataset, int(seed), bool(shuffle)
        self.shard_size, self.block_size, self.epoch = int(shard_size), int(block_size), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        starts = list(range(0, len(self.dataset), self.shard_size))
        rng = random.Random(self.seed + self.epoch * 1009)
        if self.shuffle:
            rng.shuffle(starts)
        assigned = starts[worker_id::workers]
        if not self.shuffle:
            for start in assigned:
                for index in range(start, min(start + self.shard_size, len(self.dataset))):
                    yield self.dataset[index]
            return
        for begin in range(0, len(assigned), self.block_size):
            states = []
            for start in assigned[begin:begin + self.block_size]:
                size = min(self.shard_size, len(self.dataset) - start)
                order = np.random.default_rng(self.seed + self.epoch * 1009 + start).permutation(size)
                states.append([start, order, 0])
            active = list(range(len(states)))
            while active:
                pos = rng.randrange(len(active))
                state = states[active[pos]]
                index = state[0] + int(state[1][state[2]])
                state[2] += 1
                yield self.dataset[index]
                if state[2] == len(state[1]):
                    active.pop(pos)


def loader(dataset, settings, train=False, seed=0):
    stream = HistoricalOrder(dataset, seed, train, settings.get('shard_size', 4096), settings.get('interleave_shards', 16))
    return DataLoader(stream, batch_size=int(settings['batch_size'] if train else settings.get('evaluation_batch_size', 512)),
                      num_workers=int(settings.get('loader_workers', 0)), collate_fn=collate,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def validation_auc(model, dataset, settings, device, amp=False):
    model.eval()
    labels, scores = [], []
    for batch in loader(dataset, settings):
        batch = move_batch(batch, device)
        with torch.autocast(device_type='cuda', enabled=True) if amp else nullcontext():
            logits = model(batch)
        labels.append(batch['labels'].cpu().numpy())
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(scores)))


def fit(model, train, valid, settings, output, seed, device, encoders, *, freeze_epochs=0, encoder_only=None, extras=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoders = list(encoders)
    enc_params = [p for e in encoders for p in e.parameters()]
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    encoder_lr = float(settings.get('encoder_learning_rate', settings.get('learning_rate', 3e-4)))
    head_lr = float(settings.get('head_learning_rate', 1e-3))
    opt = torch.optim.AdamW([{'params': enc_params, 'lr': 0. if freeze_epochs else encoder_lr},
                            {'params': other_params, 'lr': head_lr}], weight_decay=float(settings['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=float(settings['scheduler_factor']),
                 patience=int(settings['scheduler_patience']), min_lr=float(settings['minimum_learning_rate']))
    amp = bool(settings.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    positive_weight = float((train.labels == 0).sum() / (train.labels == 1).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    best, best_epoch, stale, history = -float('inf'), 0, 0, []
    best_path = output / 'best.pt'
    record = {'seed': seed, 'training': settings, 'train_sha256': train.source_sha256,
              'validation_sha256': valid.source_sha256, 'positive_weight_HS': positive_weight,
              'train_counts': {'ES': int((train.labels == 0).sum()), 'HS': int((train.labels == 1).sum())},
              **(extras or {})}
    atomic_json(output / 'run_config.json', record)
    train_loader = loader(train, settings, train=True, seed=seed)
    model.to(device)
    for epoch in range(1, int(settings['max_epochs']) + 1):
        if epoch > freeze_epochs and stale >= int(settings['patience']):
            break
        frozen = epoch <= freeze_epochs
        train_loader.dataset.set_epoch(epoch)
        if epoch == freeze_epochs + 1 and freeze_epochs:
            opt.param_groups[0]['lr'] = encoder_lr
            stale = 0
        for p in enc_params:
            p.requires_grad_(not frozen)
        model.train()
        if frozen:
            for e in encoders:
                e.eval()
        total, count = 0., 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=True) if amp else nullcontext():
                loss = loss_fn(model(batch), batch['labels'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings['gradient_clip_norm']))
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(batch['labels'])
            count += len(batch['labels'])
        auc = validation_auc(model, valid, settings, device, amp=amp)
        if not frozen:
            scheduler.step(auc)
        if auc > best + float(settings.get('min_delta_auroc', settings.get('min_delta_auc', .0001))):
            best, best_epoch, stale = auc, epoch, 0
            payload = {'format': 'gesa-training-validation-best-v2', 'seed': seed, 'epoch': epoch,
                       'validation_auroc': auc, 'model_state_dict': model.state_dict(), 'training_metadata': record}
            if extras:
                payload.update(extras)
            torch.save(payload, best_path)
            if encoder_only is not None:
                torch.save({'format': 'taskgat-pretraining-v1', 'seed': seed, 'epoch': epoch,
                            'validation_auroc': auc, 'encoder_state_dict': encoder_only.state_dict()}, output / 'best_encoder.pt')
        elif not frozen:
            stale += 1
        row = {'epoch': epoch, 'train_bce': total / count, 'validation_auroc': auc,
               'best_epoch': best_epoch, 'best_validation_auroc': best, 'stale': stale,
               'encoder_lr': opt.param_groups[0]['lr'], 'head_lr': opt.param_groups[1]['lr']}
        history.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
    result = {'best_epoch': best_epoch, 'best_validation_auroc': best, 'checkpoint': str(best_path),
              'checkpoint_sha256': sha(best_path), 'completed_epochs': len(history)}
    atomic_json(output / 'summary.json', result)
    return result
