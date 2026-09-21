# GESA checkpoint

`GESA.pt` is the released GESA checkpoint (seed 123, epoch 38).
Load it through the public `gesa.load_gesa` API or `scripts/predict.py`.
The output score is P(HS): larger values indicate greater predicted synthetic
difficulty, with 0.5 as the default classification threshold.

The serialized parameter names are retained for checkpoint compatibility.
The checkpoint SHA256 is recorded in `../RELEASE_MANIFEST.csv`.
