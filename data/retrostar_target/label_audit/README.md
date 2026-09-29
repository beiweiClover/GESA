# Retro* label provenance and dataset versions

This directory contains source records for the label-consistency analysis. It
summarizes existing data and predictions; no planner was rerun and no model was
retrained for this audit.

## Dataset counts

| Version or component | Molecules |
|---|---:|
| Target splits before single-component cleanup | 587,021 |
| Removed disconnected, multicomponent TarPass records | 4,794 |
| Released single-component target splits | 582,227 |
| Records with explicit generated-source planning information | 432,386 |
| Historical GASA-Retro* labels retained | 149,841 |

Thus, 587,021 = 432,386 + 149,841 + 4,794. The historical GASA-Retro* records
retain their published ES/HS labels.

## Files

- `count_reconciliation.json`: before/after-cleaning totals and label-rule comparison denominator.
- `multicomponent_removals_by_split.csv`: ES/HS counts removed from each split.
- `source_label_counts_disjoint_groups.csv`: mutually exclusive source groups; shared molecules are counted once.
- `source_label_and_route_summary.csv`: source-group labels and available complete-route statistics.
- `complete_route_length_distribution.csv`: full step-count frequencies for complete routes with recorded steps.
- `planning_settings_from_primary_run_logs.csv`: settings recovered from all six primary TarPass/MolGenBench partition logs.
- `TS4_source_predictions.csv`: molecule identifiers, recorded source metadata, labels and frozen GESA predictions for the 3,584 TS4 molecules.
- `TS4_GESA_within_source_metrics.csv`: AUROC/AP for each TS4 source, calculated from the preceding file.
- `summarize_existing_labels.py`: rebuilds the label summaries from the original data directories specified through `--project`; the large training datasets are distributed separately.

The recorded source SMILES in `TS4_source_predictions.csv` preserve the original
metadata, including any atom-map annotations. Source identities are joined by
`molecule_id`; no structure-based rematching or score-based selection is used.
HS is the positive class. AP denotes non-interpolated average precision.

A complete 0-step route in the other groups indicates that the target itself is in the stock library.

The six primary run logs all specify 600 seconds, 150 iterations, top-50 expansion, the same
stock, template and model filenames, and the learned value function.

## Reproduce the source-stratified TS4 metrics

From the repository root:

```bash
python data/retrostar_target/label_audit/verify_source_metrics.py
```

The three unseen-generator cohorts are evaluated separately under
`benchmarks/external/generator_transfer/`; they are not merged into TS4 here.
