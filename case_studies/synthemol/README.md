# SyntheMol case study

`synthemol_70_four_models_retrostar.csv` contains one row for each of 70 unique,
RDKit-canonicalized molecules. It retains only identifiers, the generation-model
stratum used by the analysis, experimental annotations, four molecular scores,
and the fixed-budget Retro* result.

## Experimental annotations

- `synthesis_status`: 58 `success` and 12 `failed` experimental syntheses.
- `active`: six compounds listed in the supplied active table; all six are also
  synthesis successes.
- `experimental_group`: mutually exclusive reporting groups:
  `active_success` (6), `synthesis_success_other` (52), and
  `synthesis_failed` (12). `synthesis_success_other` means only “not present in
  the supplied active table”; it must not be interpreted as experimentally
  inactive.

Synthesis annotations reflect the experimental outcomes reported in the SyntheMol study.

## Score semantics

All four retained scores use `0 = easier` and `1 = harder`:

- SCScore difficulty: `(SCScore_raw_1to5 - 1) / 4`.
- BR-SAScore difficulty: `(BR-SAScore_raw_1to10 - 1) / 9`.
- SynFrag difficulty: `1 - clip(SynFrag_native_ease, 0, 1)`.
- `gesa_p_hs`: GESA `P(HS)`; `gesa_class` uses the fixed 0.5 boundary.

The first three are direction/range transformations, not probability
calibrations. Their absolute magnitudes should not be read as directly
interchangeable with GESA probabilities.

## Retro* semantics

Retro* used the prior reference setting: at most 150 iterations per molecule,
expansion top-k 50, and a 600-second per-molecule timeout. It found a route for
66 molecules and returned `no_route` for four, indicating that no complete route
was identified within the search budget.

For solved cases, `retrostar_best_route` encodes each reaction as
`product>one_step_score>reactants`, with multiple reactions separated by `|`.
`retrostar_reaction_nodes` counts all reaction nodes in the best AND tree;
`retrostar_route_depth` is the maximum root-to-leaf reaction depth. Route fields
are intentionally empty for `no_route` rows, while the exhausted iteration
budget remains recorded.

## Provenance

The compact table is derived without rescoring from
`synthemol_70_four_models_retrostar_full.csv`, SHA256
`f4c15f0867dde5cadf4f3db9bdd18b46fe1c01b16f5431870704ad94433ed6a1`,
under `Case study/analysis/synthemol_four_model_retrostar/tables/` in the
enclosing project.
