# Generative-space-expanded target data

The supervised field is `target_hard`: ES is `0`, HS is `1`, and HS is the
positive class used by the GESA loss and evaluation metrics.

| Split | Total | ES | HS | Source SHA256 |
|---|---:|---:|---:|---|
| Train | 469,078 | 145,285 | 323,793 | `32b154d0f13e83b0543e924fc10e49835a1b3343eef742ed30de39e6de8c3826` |
| Validation | 58,394 | 17,923 | 40,471 | `83535a2fc23f3e1d7bb5e368d99e436d72614e18a4691ec4a8622f47b6338265` |
| Internal test | 54,755 | 16,178 | 38,577 | `7d37c396ec931b715172902469103fda6106643a511c0b1e8118e10616e3d045` |

All model-input SMILES represent one connected molecular component. The compact
release does not duplicate these large source files; they are stored in the
companion `GESA_Data/retrostar_target/` directory.

## Label provenance and version accounting

The earlier target-split version contained 587,021 records. Removing 4,794
disconnected, multicomponent TarPass outputs yielded the 582,227 records in the
released single-component splits above. Of these, 432,386 have explicit planning
information and 149,841 retain published GASA-Retro* ES/HS labels.

[`label_audit/`](label_audit/README.md) provides per-source label counts, available
route-length distributions, primary-run settings, split-level removal counts,
and the source-stratified TS4 predictions and metrics.

## Training data

**Companion dataset:** [Zenodo — 10.5281/zenodo.22874755](https://doi.org/10.5281/zenodo.22874755).

Benchmark and case-study source data are included in this repository. The
companion training-data record is linked above. To use the training data,
extract the dataset locally and set the training and validation paths in
`configs/gesa.yaml` at the repository root.

## Original training version

The [lossless version patch](version_patch/README.md) restores the original
587,021-record target splits and verifies their original file hashes. The original
and single-component versions have separate identifiers, manifests, and training
profiles; see the [training guide](../../TRAINING.md).
