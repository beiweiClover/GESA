# Legacy SA supervision

The task-specific encoder was pretrained using the immutable legacy SA view.

Its supervised target follows the GESA convention: `target_hard = 0` for ES and
`target_hard = 1` for HS. Some source/audit tables also retain
`reference_easy`, which is the exact inverse (`ES = 1`, `HS = 0`) and is not the
model target.

- Train: 519,974 molecules; source SHA256 `0e447b407244103c61512cb4c0e1d0577ce7ae8fa2f75979f58f8c3dc54d78dc`
- Validation: 64,994; source SHA256 `64da83f58259cdc43c556bf353be3d91e4b2e489b52718bc74df03caa32b986e`
- Test: 65,000; source SHA256 `18684b9a566fc86068dd90d782d9694d58f5f965ada856de661f3f8a04279795`

The legacy SA corpus is available from [Zenodo](https://doi.org/10.5281/zenodo.22874755).
Task-GAT used its existing `split_reconstructed` membership after excluding the
32 `source_row_index` values in
[`legacy_excluded_32.csv`](../../configs/pretraining/legacy_excluded_32.csv).
This recovers the 519,974 / 64,994 / 65,000 train/validation/test counts above.
The seed-123 initialization checkpoint is in
[`weights/pretrained/TaskGAT_seed123.pt`](../../weights/pretrained/TaskGAT_seed123.pt),
and its settings are in
[`configs/pretraining/taskgat.yaml`](../../configs/pretraining/taskgat.yaml).

## Training data

**Companion dataset:** [Zenodo — 10.5281/zenodo.22874755](https://doi.org/10.5281/zenodo.22874755).

Benchmark and case-study source data are included in this repository. The
companion training-data record is linked above. To use the training data,
extract the dataset locally and set the training and validation paths in
`configs/gesa.yaml` at the repository root.

## Executable training

See the [training guide](../../TRAINING.md) for data preparation and the corresponding
pretraining command. Executable configurations are in `configs/training/` at the
repository root.
