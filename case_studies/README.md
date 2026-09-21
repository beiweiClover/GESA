# Case studies

This directory contains compact, manuscript-facing source tables for two
descriptive GESA case studies. They are separate from the external and internal
benchmarks because neither dataset is used here as an unbiased performance test.

- `stepwise_routes/`: seven selected seven-step literature syntheses, with one
  row per reaction product and its GESA `P(HS)`.
- `synthemol/`: 70 SyntheMol compounds with experimental synthesis annotations,
  four direction-aligned molecular difficulty scores, and fixed-budget Retro*
  planning results.

Across both case studies, larger score values mean greater predicted synthetic
difficulty. Only GESA is a probability (`P(HS)`); the other aligned scores are
not probabilities and are not mutually calibrated with GESA.

The CSV files are retained as the source data for analysis and plotting.
