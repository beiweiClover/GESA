# Stepwise literature-route case study

## Files

- `stepwise_routes_gesa.csv`: 49 reaction products from seven complete,
  graph-valid linear routes. The reaction SMILES, product SMILES, GESA score,
  and fixed-threshold ES/HS call are retained for every step.
- `route_summary.csv`: one row per route with publication metadata, the Spearman
  correlation between step position and GESA score, and the ordered score
  sequence as JSON.

`reaction_smiles` uses `reactants>>product`; `product_smiles` is the exact right
side of that source reaction SMILES. `gesa_p_hs` stores the uncalibrated GESA score on a 0 to 1 scale, with larger
values indicating greater predicted difficulty. `gesa_class` is `HS` when
`gesa_p_hs >= 0.5` and `ES` otherwise.

## Selection and interpretation boundary

These are seven-step members selected from a prespecified route subset covering
publication years 2020–2024 with valid linear graphs and 5–15 total steps.
They are representative illustrations of GESA score progression along synthesis
routes.

GESA is a molecule-level difficulty model. Its score is not a reaction yield,
reaction-feasibility estimate, or retrosynthetic action score.

## Provenance

- Selection table: `seven_step_routes_sorted_by_publication_year.csv`,
  SHA256 `89ab89347eb65e59e66f9b46089e5d0efca9757df8fd12f4d67734524356c878`.
- Step table: `linear_route_step_trajectories.csv.gz`, SHA256
  `3294d9f3d9620a8643ab60752cb50a769054dea27276b12b25f2c4c14493ad15`.
- Source analysis: `Case study/analysis/stepwise_gesa/` in the enclosing project.
