# Data layout

Training datasets are distributed through [Zenodo](https://doi.org/10.5281/zenodo.22874755).
This repository provides the model implementation, checkpoints, benchmark predictions,
and case-study source data. Dataset locations, split counts, and checksums are
documented in the corresponding data subdirectories.

Labels use `0 = easy-to-synthesize` and `1 = hard-to-synthesize`; HS is the
positive class. The supervised training field is `target_hard`, and released
benchmark files expose the same label as `y_true`. If a source table contains
`reference_easy`, it is only an inverse convenience field (`ES = 1`, `HS = 0`)
and must not be used to interpret the GESA target or output score.

## Training data

**Companion dataset:** [Zenodo — 10.5281/zenodo.22874755](https://doi.org/10.5281/zenodo.22874755).

Benchmark and case-study source data are included in this repository. The
companion training-data record is linked above. To use the training data,
extract the dataset locally and set the training and validation paths in
`configs/gesa.yaml` at the repository root.

## Generated-molecule validation source data

The three held-out generator cohorts are stored with their evaluation records in
[`benchmarks/external/generator_transfer/`](../benchmarks/external/generator_transfer/README.md).
That directory includes molecular structures, planning labels, all nine methods'
scores, generation seeds and source/version records. It is separate from the
training-data directories above.
