# GESA checkpoints

`GESA.pt` is the released GESA checkpoint (trial_011, seed 123, epoch 38).
Load it through the public `gesa.load_gesa` API or `scripts/predict.py`.
Larger scores indicate greater predicted synthetic difficulty, with 0.5 as the
default classification threshold.

The two encoder initialization checkpoints are:

| File | Initialization | Selected checkpoint |
|---|---|---|
| `pretrained/TaskGAT_seed123.pt` | Legacy SA supervision | Seed 123, epoch 2 |
| `pretrained/ReactionGINE_epoch23.pt` | ORD reaction pretraining | Seed 42, epoch 23 |

The initialization files retain their original `encoder_state_dict` payloads.
To initialize the two branches of a new GESA model:

```python
import torch
from gesa.model import GESA

torch.manual_seed(123)
model = GESA()
task = torch.load("weights/pretrained/TaskGAT_seed123.pt", map_location="cpu", weights_only=False)
reaction = torch.load("weights/pretrained/ReactionGINE_epoch23.pt", map_location="cpu", weights_only=False)
model.task_encoder.load_state_dict(task["encoder_state_dict"], strict=True)
model.ord_encoder.encoder.load_state_dict(reaction["encoder_state_dict"], strict=True)
```

[`checkpoint_metadata.json`](checkpoint_metadata.json) records identifiers,
selected epochs, file sizes, and SHA256 values for all three checkpoints.
All distributed files are also covered by [`RELEASE_MANIFEST.csv`](../RELEASE_MANIFEST.csv).

Pretraining settings are in [`configs/pretraining/`](../configs/pretraining/),
and the original downstream settings and data-version records are in
[`configs/training_provenance.json`](../configs/training_provenance.json).
The original released `GESA.pt` is unchanged.
