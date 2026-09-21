# Legacy SA supervision

The task-specific encoder was pretrained using the immutable legacy SA view.

Its supervised target follows the GESA convention: `target_hard = 0` for ES and
`target_hard = 1` for HS. Some source/audit tables also retain
`reference_easy`, which is the exact inverse (`ES = 1`, `HS = 0`) and is not the
model target.

- Train: 519,974 molecules; source SHA256 `0e447b407244103c61512cb4c0e1d0577ce7ae8fa2f75979f58f8c3dc54d78dc`
- Validation: 64,994; source SHA256 `64da83f58259cdc43c556bf353be3d91e4b2e489b52718bc74df03caa32b986e`
- Test: 65,000; source SHA256 `18684b9a566fc86068dd90d782d9694d58f5f965ada856de661f3f8a04279795`

The large CSVs are not bundled. Configure local paths before retraining.

## Training data

**Companion dataset:** [Zenodo — 10.5281/zenodo.22874755](https://doi.org/10.5281/zenodo.22874755).

Benchmark and case-study source data are included in this repository. The
companion training-data record is linked above. To use the training data,
extract the dataset locally and set the training and validation paths in
`configs/gesa.yaml` at the repository root.
