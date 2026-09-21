# ORD reaction pretraining

Reaction-GINE was initialized from an ORD-pretrained five-layer GINE encoder
(development checkpoint epoch 23, SHA256
`1f1692b0e7b2b11f87757c9e93e1066ec8414685f74460b461cef9e10a160ac0`).
The reaction records are not redistributed here. The released `GESA.pt` contains
the fine-tuned Reaction-GINE parameters needed for inference.

## Training data

**Companion dataset:** [Zenodo — 10.5281/zenodo.22874755](https://doi.org/10.5281/zenodo.22874755).

Benchmark and case-study source data are included in this repository. The
companion training-data record is linked above. To use the training data,
extract the dataset locally and set the training and validation paths in
`configs/gesa.yaml` at the repository root.
