# GESA evaluation on three held-out generators

This directory supports the evaluation of NEAT-POCKET, FLOWR and ResGen in the GESA manuscript. It contains generated structures, Retro* labels, all nine methods' predictions, sample audit records and the source data for Supplementary Figure S4. Each generator is evaluated separately without retraining or tuning GESA.

## Primary files

- `data/evaluation_molecules.csv`: all common evaluable molecules; use the `generator` field to evaluate each source separately.
- `data/all_scored_molecules.csv`: all eligible scored molecules, including the two NEAT-POCKET molecules excluded because planning failed or timed out.
- `data/SAScore_3_to_4_evaluation.csv`: fixed 3 <= SAScore < 4 subset of common evaluable molecules.
- `data/*_evaluation.csv`: generator-specific evaluation files.
- `molecules/*_evaluation.sdf`: evaluated molecules with original generated 3D coordinates and joined labels/scores as SDF properties. Coordinates have not been minimized or redocked during packaging.
- `raw_molecules/<generator>/*.sdf`: native SDF records from the final generation attempts, including rejected structures and records beyond the first-50 cap.
- `data/generation_record_index.csv`: maps every native record to a source SDF, 1-based record index and inclusion decision.
- `data/structure_audit.csv`: structure audit for the prespecified first-50-per-pocket cohorts.
- `metadata/sample_flow.csv`: requested/native/selected/scored/evaluable counts, labels and final pocket coverage.
- `metadata/data_dictionary.csv`: score directions, labels and other field definitions.
- `metrics/all_metrics.csv`: full, no-shared-scaffold and fixed-SA-subset AUROC/AP, 95% CIs and paired GESA differences.
- `metrics/full_vs_SA_advantage.csv`: fixed-reference paired changes in relative AUROC differences after range restriction.
- `figures/`: comparison figure and its plotted source data. The three highest AUROC methods are displayed; the CSV metrics contain all nine methods.
- `metadata/model_sources.json`: generator publications, repository commits and checkpoint identities.
- `generation/*_generation_attempts.csv`: generation attempts. Final-attempt SDFs are used; initial input compatibility failures were repaired before scoring, rather than selected using model performance.
- `code/reproduce_metrics.py`: independently reproduce point estimates and optionally pocket-bootstrap intervals from the published scores.

## Design and interpretation

The same 20 pockets were prespecified for generation, with 50 molecules requested per pocket per generator. Actual output counts varied. Sampling used public pretrained weights with no SAScore-based selection, and GESA was not retrained or tuned on these results. Generator sources were absent from the audited GESA target training sources. The design evaluates transfer to these sampled generators under the same Retro* label framework.

Each generator is a separate cohort. Uniqueness is enforced within generator cohorts; possible cross-generator coincidences are retained, and `structure_id` allows them to be identified.

Exact exclusion used the union of historical ZINC15/Nonpher, released target training and validation, Table-1-version target training and validation, and ORD training and validation reactants/products. Exact matching used atom-map-free canonical isomeric SMILES. Scaffold sensitivity uses the released target training keys.

Retro* used the original fixed stock, single-step model and value function, 150 iterations, top-50 templates, and 600 seconds per molecule. A complete route of at most 10 reaction steps yields ES=0; a complete route longer than 10 or a normally completed search without a complete route yields HS=1. Timeout and execution-error records have no label and are excluded from evaluation. A 0-step complete route indicates that the target is itself in stock.

`HS_score_*` columns are the direction-aligned continuous scores used for all reported metrics, with higher scores indicating HS. The native `SYBA`, `RAscore_XGB`, and `SynFrag_raw` have the reverse direction; do not feed them directly into an HS-positive AUROC calculation. SynFrag evaluation uses its unclipped native output, not `SynFrag_clipped`. AP denotes non-interpolated average precision, not trapezoidal PR-AUC. AP depends on cohort class prevalence.

Confidence intervals use 2,000 percentile pocket-level bootstrap replicates, with paired sampling across methods. The full-versus-restricted difference analysis uses the same pocket resamples for both cohorts.

## Publication sources

- NEAT-POCKET: https://arxiv.org/abs/2609.05097 (preprint).
- FLOWR: https://doi.org/10.1038/s43588-026-00998-8 (Nature Computational Science, 2026).
- ResGen: https://doi.org/10.1038/s42256-023-00712-7 (Nature Machine Intelligence, 2023).

## Reuse and provenance

This package contains newly generated molecular outputs and computed scores. Third-party checkpoints, upstream source-code archives and protein/ligand input structures are not redistributed here. Their source links and identifiers are recorded instead. The parent GESA repository's software license and upstream third-party terms remain applicable; no separate data-specific license is assigned by this directory.

CSV files use UTF-8, boolean flags use True/False, and missing values are empty fields. Checksums cover all distributed payload files. Predictor and planner file identities were verified against the preceding experiment; full source snapshots remain in the author's experiment archive.

## Detailed sampling and rounding records

See `metadata/SUPPORTING_RECORDS.md` for per-stage counts, all 60 pocket seeds, exact generator/checkpoint versions, and the AUROC difference rounding audit.
