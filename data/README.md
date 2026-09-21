# Data layout

The compact release includes external benchmark records needed to audit prediction
alignment. Large training corpora, cached graph shards, and ORD reaction records are
not duplicated. Their provenance, expected counts, and source hashes are documented
in the subdirectory READMEs.

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
