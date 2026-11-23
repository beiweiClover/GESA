# External benchmark

This directory contains the TS1--TS4 comparison of eight reference methods and
GESA:

`SAScore`, `SCScore`, `SYBA`, `BR-SAScore`, `RAscore-XGB`, `GASA`, `SynFrag`,
`DeepSA`, and `GESA`.

## Evaluation protocol

The positive class is HS (`target_hard = 1`; ES is `0`). AUROC and AP treat
HS as the positive label. Recall, Precision, and F-score are calculated for HS,
whereas Specificity is the true-negative rate for ES. Thus TP means correctly
predicted HS, FN means HS predicted as ES, TN means correctly predicted ES, and
FP means ES predicted as HS. ROC and PR curves likewise use `pos_label = HS`.

Binary classifiers use the decision boundary defined by their paper or official
implementation:

| Model | HS classification rule | Threshold source |
|---|---|---|
| SYBA | raw score `<= 0` | official definition |
| BR-SAScore | 1--10 score `>= 5` | paper definition |
| RAscore-XGB | ES probability `<= 0.5` | official definition |
| GASA | HS probability `>= 0.5` | official definition |
| DeepSA | HS probability `>= 0.5` | official definition |
| SynFrag | ES probability `<= 0.5` | official definition |
| GESA | synthetic-difficulty score `>= 0.5` | default cutoff |

`GESA_score` is an uncalibrated synthetic-difficulty score. Scores at or
above the default cutoff are classified as HS and lower scores as ES.

SAScore and SCScore have no native binary boundary. For these two methods only,
the threshold is selected on each TS1--TS4 set by maximizing Youden J
(`sensitivity + specificity - 1`), equivalently Balanced Accuracy over exact
ROC candidate thresholds. If thresholds tie, the highest hard-score threshold
is used. The resulting threshold-dependent metrics are test-set-optimized
reference values. AUROC and average precision (AP, stored as `PRAUC` in the
existing CSV) are calculated from continuous scores.

GESA classification metrics use the released public checkpoint (seed 123,
epoch 38). `auc_results.csv` additionally reports GESA AUROC as the mean and
sample SD across seeds 42, 123, and 2026; reference methods have one released
score set each. The seed-123 block in `predictions/GESA.csv` is exactly the
prediction set produced by the current public checkpoint. TS3 contains one
unscored record (`TS3_BIVALIRUDIN`) caused by an RDKit parse failure; all other
GESA records are scored.

## Files

- `full_metrics.csv`: complete TS1--TS4 table with Accuracy, AUROC, Specificity,
  PRAUC, Recall, F-score, Precision, Balanced Accuracy, raw and unified
  thresholds, classification rules, and confusion counts. All positive-class
  metrics and TP/FN counts refer to HS.
- `auc_results.csv`: summary continuous AUROC metrics, including GESA
  three-seed mean and sample SD.
- `auc_results_long.csv`: per-split and per-seed/reference AUROC records.
- `predictions/reference_models.csv.gz`: released hard-class scores for the
  eight reference methods.
- `predictions/GESA.csv`: GESA scores for seeds 42, 123, and 2026.
- `predictions/GESA_public_checkpoint.csv`: predictions from the released checkpoint.

The CSV files are the source data for downstream analysis and plotting.

## Additional generator-transfer evaluation

[`generator_transfer/`](generator_transfer/README.md) contains the separate
NEAT-POCKET, FLOWR, and ResGen evaluation cohorts. The main data file is
[`data/evaluation_molecules.csv`](generator_transfer/data/evaluation_molecules.csv),
with 2,252 molecules, Retro* labels, native scores and HS-direction scores for
all nine methods. The two NEAT-POCKET planning exceptions remain in
`data/all_scored_molecules.csv` with missing labels and are excluded from evaluation.

The directory also provides native and annotated generated SDFs, the fixed
SAScore `[3,4)` subset, stage counts, generator versions/checkpoint hashes,
all 60 generator–pocket seed records, complete metrics, and Supplementary Figure S4.
See [`metadata/SUPPORTING_RECORDS.md`](generator_transfer/metadata/SUPPORTING_RECORDS.md)
for the inclusion and rounding records. These cohorts are not merged into TS1–TS4.

The generator-transfer metrics use non-interpolated AP and HS-positive continuous
scores. No classification threshold is fitted on these cohorts. Its standalone
point-metric verification command, from the repository root, is:

```bash
python benchmarks/external/generator_transfer/code/reproduce_metrics.py
```
