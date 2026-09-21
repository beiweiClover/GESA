# External benchmark

This directory contains the TS1--TS4 comparison of eight reference methods and
GESA:

`SAScore`, `SCScore`, `SYBA`, `BR-SAScore`, `RAscore-XGB`, `GASA`, `SynFrag`,
`DeepSA`, and `GESA`.

## Evaluation protocol

The positive class is HS (`target_hard = 1`; ES is `0`). AUROC and PRAUC treat
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
| GESA | HS probability `>= 0.5` | natural model boundary |

For GESA specifically, `GESA_score = P(HS)`: scores at or above the threshold
are classified as HS, and scores below the threshold are classified as ES.

SAScore and SCScore have no native binary boundary. For these two methods only,
the threshold is selected independently on each TS1--TS4 external set by
maximizing Youden J (`sensitivity + specificity - 1`), equivalently Balanced
Accuracy over exact ROC candidate thresholds. If thresholds tie, the highest
hard-score threshold is used.

This follows the mixed threshold policy commonly used by the reference
benchmark: native boundaries for classifiers and a test-specific best
operating point for continuous heuristic scores. SAScore/SCScore classification
metrics therefore use external labels for threshold selection. AUROC and PRAUC
use continuous hard-class scores and are threshold-independent.

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
- `auc_results.csv`: paper-facing continuous AUROC summary, including GESA
  three-seed mean and sample SD.
- `auc_results_long.csv`: per-split and per-seed/reference AUROC records.
- `predictions/reference_models.csv.gz`: released hard-class scores for the
  eight reference methods.
- `predictions/GESA.csv`: GESA scores for seeds 42, 123, and 2026.
- `predictions/GESA_public_checkpoint.csv`: predictions from the released checkpoint.

The CSV files are the source data for downstream analysis and plotting.
