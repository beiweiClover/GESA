# Supporting records for the three-generator evaluation

This Source Data record accompanies Supplementary Figure S4. It adds no new experiments or molecular analyses.

- `sample_flow_detailed.csv`: source-specific counts at each generation, structure audit, planning and evaluation stage, including the fixed SAScore subset.
- `per_pocket_generation_manifest.csv`: final generation seed and stage counts for each of the 60 generator–pocket combinations.
- `generator_versions_weights_and_seeds.csv`: exact upstream code commits, checkpoint filenames and SHA-256 identities, auxiliary NEAT bond-predictor identity, and seed ranges.
- `sampling_protocol.json`: fixed inference settings and analysis seeds. ResGen used beam_size=100, with the original probability thresholds and maximum 50 atom-generation steps; this compute setting was fixed before SA/planning outcomes. FLOWR used the noHs checkpoint, 100 integration steps and uniform-sample categorical sampling. NEAT-POCKET used the SPINDR checkpoint and paired bond predictor, 60 integration steps and guidance=0.5.
- `../generation/*_generation_attempts.csv`: all recorded attempts, including initial compatibility failures. Final-attempt SDF records are the ones included in the structure audit.
- `../metrics/AUROC_margin_rounding_audit.csv`: independent recalculation from the released per-molecule scores. AUROC differences are calculated before rounding. Exact differences and metrics are provided in the audit table.

All three generators were requested to produce 50 molecules PER POCKET for each of 20 pockets, i.e. 1,000 requested per generator. ResGen saved 531 native records; limiting each pocket to its first 50 records retained 515, and excluding three empty structures and 40 exact development matches retained 472. Native yield variations across pockets reflect generator output characteristics.

NEAT-POCKET generation seeds were 6621–6640, FLOWR 48210–48229, and ResGen 47210–47229. Exact seeds are mapped to pocket identifiers in the manifest. Metric confidence intervals used 2,000 pocket-level bootstrap replicates with seed 6621; the exploratory paired full-versus-SA contrast used seed 20261119.

Version and resource identities are preserved in the accompanying JSON/CSV records. No checkpoints or large upstream source datasets are redistributed in this directory.
