# Standard GNN baselines

The internal benchmark uses GCN, edge-aware GAT, GINE, and AttentiveFP with the
same seven atom-feature fields, four bond-feature fields where supported, and
validation-AUROC checkpoint selection. `models.py` contains the public
implementations and `train.py` provides a validation-only training entry point.

Configure local train/validation paths in the selected YAML before running, e.g.:

```bash
PYTHONPATH=src python baselines/train.py \
  --config baselines/configs/gat.yaml \
  --output runs/GAT_validation_best.pt
```
