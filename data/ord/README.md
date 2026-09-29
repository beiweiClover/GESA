# ORD reaction pretraining

Reaction-GINE was initialized from the five-layer ORD-pretrained encoder at epoch 23.
The initialization checkpoint is in
[`weights/pretrained/ReactionGINE_epoch23.pt`](../../weights/pretrained/ReactionGINE_epoch23.pt).
Its training settings are in
[`configs/pretraining/reaction_gine.yaml`](../../configs/pretraining/reaction_gine.yaml),
and its SHA256 is recorded in
[`weights/checkpoint_metadata.json`](../../weights/checkpoint_metadata.json).

The cleaned ORD data are available from
[Zenodo](https://doi.org/10.5281/zenodo.22874755):

| Split | Reactions |
|---|---:|
| Train | 828,786 |
| Validation | 6,129 |
| Test | 6,352 |

The archive's `ord/manifest.json` lists the fixed split shards and their SHA256
checksums. Individual reaction records retain their split membership. Pretraining
used the train split, and the checkpoint was selected by minimum validation loss.
The InfoNCE temperature was 0.1, and the total objective was
`0.5 * atom_BCE + 0.5 * bond_BCE + 0.5 * bidirectional_InfoNCE`.
Atom and bond positive weights were calculated separately within each minibatch
as `clamp(N_negative / N_positive, 1, 20)`, with unweighted BCE if no positives
were present. The complete settings are in the configuration linked above.

## Executable training

See the [training guide](../../TRAINING.md) for data preparation and the corresponding
pretraining command. Executable configurations are in `configs/training/` at the
repository root.
