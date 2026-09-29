#!/usr/bin/env python3
"""Prepare fixed legacy splits and restore both target-data versions from Zenodo."""
from pathlib import Path
import argparse
import base64
import csv
import gzip
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def prepare(data_root, output_root):
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    patch_root = ROOT / 'data/retrostar_target/version_patch'
    metadata = json.loads((patch_root / 'dataset_versions.json').read_text())
    patch = patch_root / metadata['patch']['file']
    if sha(patch) != metadata['patch']['sha256']:
        raise ValueError('Target-version patch checksum mismatch')
    missing = {s: {} for s in metadata['splits']}
    with gzip.open(patch, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            missing[row['split']][row['original_row']] = base64.b64decode(row['row_b64'])
    manifest = {'source_data_doi': metadata['released_data_doi'], 'target': {}, 'legacy': {}}
    for split, versions in metadata['splits'].items():
        source = data_root / 'retrostar_target' / (split + '.csv')
        if sha(source) != versions['released']['sha256']:
            raise ValueError(f'Released target checksum mismatch: {source}')
        clean = output_root / 'target_single_component' / source.name
        clean.parent.mkdir(exist_ok=True)
        if not clean.exists():
            try:
                os.link(source, clean)
            except OSError:
                shutil.copy2(source, clean)
        elif sha(clean) != versions['released']['sha256']:
            raise ValueError(f'Output already exists with different contents: {clean}')
        target = output_root / 'target_original' / source.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists() or sha(target) != versions['original']['sha256']:
            temporary = target.with_suffix('.csv.tmp')
            with source.open('rb') as src, temporary.open('wb') as dst:
                dst.write(src.readline())
                for index in range(versions['original']['rows']):
                    row = missing[split].get(index)
                    if row is None:
                        row = src.readline()
                    if not row:
                        raise ValueError('Unexpected end of released target CSV')
                    dst.write(row)
                if src.read(1):
                    raise ValueError('Unconsumed released target rows')
            if sha(temporary) != versions['original']['sha256']:
                raise ValueError(f'Original-version reconstruction failed: {split}')
            temporary.replace(target)
        manifest['target'][split] = {'original_sha256': sha(target), 'released_sha256': sha(clean),
                                     'original_version': metadata['original_version'],
                                     'released_version': metadata['released_version']}

    legacy = data_root / 'legacy_sa/legacy_sa_650k.csv'
    if sha(legacy) != metadata['legacy_source_sha256']:
        raise ValueError(f'Legacy source checksum mismatch: {legacy}')
    with (ROOT / 'configs/pretraining/legacy_excluded_32.csv').open(encoding='utf-8') as f:
        excluded = {int(row['source_row_index']) for row in csv.DictReader(f)}
    legacy_out = output_root / 'legacy_sa'
    legacy_out.mkdir(exist_ok=True)
    handles, writers = {}, {}
    counts = {s: {'ES': 0, 'HS': 0} for s in ['train', 'validation', 'test']}
    ordered_hashes = {s: hashlib.sha256() for s in counts}
    try:
        for split in counts:
            handles[split] = (legacy_out / (split + '.csv')).open('w', encoding='utf-8', newline='')
            writers[split] = csv.writer(handles[split])
            writers[split].writerow(['canonical_smiles', 'target_hard'])
        with legacy.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if int(row['source_row_index']) in excluded:
                    continue
                split, label = row['split_reconstructed'], int(row['target_hard'])
                writers[split].writerow([row['canonical_smiles'], label])
                ordered_hashes[split].update((row['canonical_smiles'] + '\t' + str(label) + '\n').encode('utf-8'))
                counts[split]['HS' if label else 'ES'] += 1
    finally:
        for handle in handles.values():
            handle.close()
    expected = {'train': {'ES': 279974, 'HS': 240000},
                'validation': {'ES': 34994, 'HS': 30000}, 'test': {'ES': 35000, 'HS': 30000}}
    if counts != expected:
        raise ValueError(f'Legacy split counts differ from the original records: {counts}')
    for split in counts:
        ordered_sha = ordered_hashes[split].hexdigest()
        if ordered_sha != metadata['legacy_ordered_input_sha256'][split]:
            raise ValueError(f'Legacy model-input order mismatch: {split}')
        manifest['legacy'][split] = {**counts[split], 'sha256': sha(legacy_out / (split + '.csv')),
                                     'ordered_model_input_sha256': ordered_sha}
    manifest['legacy_source_sha256'] = sha(legacy)
    manifest['ord_data_dir'] = str(data_root / 'ord')
    (output_root / 'prepared_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, help='Extracted Zenodo GESA_Data directory')
    p.add_argument('--output-root', default=str(ROOT / 'training_data'))
    a = p.parse_args()
    prepare(a.data_root, a.output_root)
