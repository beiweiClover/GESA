# Matched training-data contribution (Supplementary Figure S5)

This directory contains the compact source data and analysis code for the
controlled GESA training-data comparison on TS1–TS4.

Each of three independently sampled pairs contained two training arms of
12,000 molecules (6,000 ES / 6,000 HS). Both arms shared 6,000 historical
GASA-Retro* molecules. The remaining 6,000 were historical GASA-Retro*
molecules or matched TarPass/MolGenBench molecules. The comparison held the
pretrained initialization and training seed fixed and controlled molecular
properties and scaffold richness. Scaffold identities were allowed to differ.

The **historical** arm represents training on GASA-Retro* molecules alone. Models are trained separately on each arm with the same architecture and seed.

## Files

| Location | Purpose |
|---|---|
| `data/training_subsets.csv.gz` | Exact six training lists, molecular properties and scaffold keys |
| `data/matched_pairs.csv.gz` | Historical/generated replacement pairing |
| `data/predictions_TS1_TS4.csv.gz` | All 257,256 molecule-level predictions for the six models on TS1–TS4 |
| `data/test_nearest_neighbors.csv.gz` | Similarity values used for the equal-proximity analysis |
| `data/historical_reference_scaffolds.csv.gz` | Reference keys for checking historical scaffold coverage |
| `protocol.json` | Sampling/training definitions and model identifiers |
| `results/` | Reproducible statistics and matching audit |
| `figures/Fig_S5.pdf` | Submission figure; bars are means and error bars are sample SDs |
| `figures/caption.md` | Figure legend |

The training lists contain 72,000 rows, with intentional sharing between the
paired arms. `replicate` is the data-sampling seed; `arm` is `historical` or
`generative_replacement`; `target_hard=1` denotes HS; `score` is the continuous
HS score. Molecules are paired between models by `molecule_id` within each
test set. The prediction file contains 42,876 evaluation molecules repeated
for six fitted models.

Scaffold keys use map-free, non-chiral Bemis–Murcko scaffolds. Acyclic molecules
use non-chiral whole-structure keys rather than a single empty scaffold.

## Reproduce

Run from the repository root with Python, NumPy, pandas, scikit-learn,
matplotlib and RDKit installed:

```bash
python benchmarks/internal/matched_data_contribution/code/audit_matching.py
python benchmarks/internal/matched_data_contribution/code/reproduce_results.py
python benchmarks/internal/matched_data_contribution/code/plot_figure.py
```

The analysis reconstructs AUROC/AP and paired 95% percentile intervals using
2,000 class-stratified bootstrap draws. The same test molecules are resampled
for both fitted models. The equal-proximity analysis uses molecules whose
maximum Tanimoto similarities to the two training arms differ by at most 0.02.

Sampling variability across training runs is reported as sample SD, and test-cohort uncertainty is evaluated via bootstrap intervals.

This compact release regenerates the audits, statistics and figure from the
saved training lists and predictions. Training conditions and development
checkpoint hashes are recorded in `protocol.json`.

The independent three-generator benchmark associated with Figure S4 is
documented separately in
[`generator_transfer`](../../external/generator_transfer/README.md).
Source datasets retain their original terms; see the repository's
[training-data information](../../../README.md#training-data).
