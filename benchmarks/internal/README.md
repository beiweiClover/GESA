# Internal model comparison

The paper-facing comparison is fixed to:

**Standard GNN Backbones:** GCN, GAT, GINE, AttentiveFP  
**GESA Framework:** Task-GAT, Reaction-GINE, GESA-Core, GESA

`per_seed_results.csv` retains both `paper_label` and `development_label` to preserve source-data provenance.
`summary_results.csv` contains the aggregated results for downstream analysis.

GESA-Core denotes Task-GAT + Reaction-GINE + atom-wise source gating + attentive
readout, without the fourteen explicit SA priors. GESA is the complete framework.
