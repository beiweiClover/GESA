# Training GESA

The workflow has three stages: supervised Task-GAT pretraining on the fixed legacy
SA splits, reaction pretraining on the fixed ORD splits, and GESA training from
the two encoder initializations. The released encoder checkpoints can also be
used directly for the last stage.

## Environment and fixed data

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate gesa
```

Download and extract `GESA_Data.zip` from
[Zenodo](https://doi.org/10.5281/zenodo.22874755). Pass the extracted directory
containing `legacy_sa/`, `ord/`, and `retrostar_target/`:

```bash
python scripts/prepare_training_data.py --data-root /path/to/GESA_Data
```

This prepares fixed legacy splits in `training_data/legacy_sa/` and both target
versions in `training_data/target_original/` and
`training_data/target_single_component/`. Existing split membership is retained.
The 32 excluded legacy source records are identified in
`configs/pretraining/legacy_excluded_32.csv`.

The original target version used for the released model contained 587,021 records.
Its exact CSV files are reconstructed by reinserting the 4,794 removed rows using
the compact lossless patch in `data/retrostar_target/version_patch/`. The preparer
verifies each reconstructed file against its original SHA256. The cleaned version
contains 582,227 records and remains identical to the Zenodo release. Version
identifiers, split counts, and all six target CSV checksums are in
`data/retrostar_target/version_patch/dataset_versions.json`.

## Train GESA from the released encoder initializations

```bash
python scripts/train_gesa.py --dataset-version original --seed 123
```

The default initialization loads `weights/pretrained/TaskGAT_seed123.pt` and
`weights/pretrained/ReactionGINE_epoch23.pt`. The projection, fusion, pooling,
and classifier modules are initialized for training. This command starts GESA
from the two pretrained encoders; it does not load the final `weights/GESA.pt`.

`configs/training/gesa.yaml` specifies the architecture, original target splits,
five-epoch encoder freeze, learning rates, weighted BCE, scheduler, stopping rule,
and validation-AUROC checkpoint selection. The original profile verifies the input
CSV hashes. Its fourteen SA priors use the published statistics fitted on the
original training split. Outputs are written to `runs/gesa/original/seed_123/`.

## Run all three stages from random initialization

Pretrain Task-GAT:

```bash
python scripts/pretrain_taskgat.py --seed 123
```

Pretrain Reaction-GINE:

```bash
python scripts/pretrain_reaction_gine.py --data-dir /path/to/GESA_Data/ord
```

The reaction command executes the original two-phase schedule: ten epochs of
warmup/cosine scheduling, followed by continuation at 1e-5 through epoch 30.
The minimum-validation-loss encoder is exported to
`runs/pretraining/reaction_gine/encoder.pt`.

Train GESA using the resulting encoders:

```bash
python scripts/train_gesa.py --dataset-version original --seed 123 \
  --task-checkpoint runs/pretraining/taskgat/seed_123/best_encoder.pt \
  --reaction-checkpoint runs/pretraining/reaction_gine/encoder.pt
```

Executable configurations are in `configs/training/`. Checked original settings
and historical selection records are in `configs/pretraining/` and
`configs/training_provenance.json`. Each run saves the actual input hashes,
initialization hashes, seed, class weight, selected epoch, and training history.

## Three seeds

Reaction-GINE uses seed 42 and is shared across the downstream seeds. Run the
following after reaction pretraining:

```bash
for seed in 42 123 2026; do
  python scripts/pretrain_taskgat.py --seed "$seed"
  python scripts/train_gesa.py --dataset-version original --seed "$seed" \
    --task-checkpoint "runs/pretraining/taskgat/seed_${seed}/best_encoder.pt" \
    --reaction-checkpoint runs/pretraining/reaction_gine/encoder.pt
done
```

Historical Task-GAT selections were epochs 1, 2, and 1 for seeds 42, 123, and
2026. Each new run applies the configured validation rule and records its own
selected epoch.

## Retraining on the released single-component version

```bash
python scripts/train_gesa.py --dataset-version single_component --seed 123
```

This profile uses the cleaned splits, recalculates the HS positive weight from
that training split, and fits the descriptor means and standard deviations on
training molecules only. The scaler is saved in the new checkpoint and used
automatically by the prediction API.

| Target version | Training ES / HS | HS positive weight |
|---|---:|---:|
| Original model data | 147,851 / 324,784 | 0.4552287058 |
| Released single-component data | 145,285 / 323,793 | 0.4486971615 |

Each version has its own run directory and data hashes. The existing released
checkpoint remains the original seed-123 model.

## Fine-tuning a complete GESA checkpoint

```bash
python scripts/train_gesa.py --initial-checkpoint weights/GESA.pt \
  --dataset-version original --seed 123 --output runs/checkpoint_finetuning
```

For user-provided splits, set training and validation paths in `configs/gesa.yaml`:

```bash
python scripts/train_gesa.py --config configs/gesa.yaml \
  --initial-checkpoint weights/GESA.pt --dataset-version custom
```

## Prediction with a newly trained checkpoint

```bash
python scripts/predict.py --checkpoint runs/gesa/original/seed_123/best.pt \
  --input molecules.csv --output predictions.csv
```

Graphs and raw priors are cached under `runs/molecular_cache/` and reused across
epochs. Training retains the original 4,096-row shard partition and 16-shard
interleaving scheme. Caches and downloaded/prepared data are excluded from Git.
Use `--device cpu` with a training entry point to select CPU execution.
