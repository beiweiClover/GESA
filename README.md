# GESA

[Web server](https://lmmd.ecust.edu.cn/GESA) · [Dataset](https://doi.org/10.5281/zenodo.22874755)

Official code and data repository for:

**GESA: Expanding Synthetic Accessibility Prediction into Generative Chemical Space**

Xinmin Li, Le Xiong, Wenxiang Song, Hongbo Yu, Meiling Zhan, Guixia Liu, Weihua Li, Yun Tang*

**Affiliations**

1. Shanghai Frontiers Science Center of Optogenetic Techniques for Cell Metabolism
2. Shanghai Key Laboratory of New Drug Design, School of Pharmacy, East China University of Science and Technology, Shanghai 200237, China

**Corresponding author:** Yun Tang — [ytang234@ecust.edu.cn](mailto:ytang234@ecust.edu.cn)

**Manuscript status:** Under submission.

GESA is a molecular synthetic-accessibility prediction framework combining
supervision expanded into generative chemical space with task-specific,
reaction-informed, and explicit synthetic-accessibility evidence.

![Overview of GESA: generative chemical space, Task-GAT, Reaction-GINE, SA priors, and the web server](assets/GESA_cover.png)

## Web server

Access the GESA web server at **[https://lmmd.ecust.edu.cn/GESA](https://lmmd.ecust.edu.cn/GESA)**.

## Architecture

```text
Molecule
   |
   +-- Task-GAT -------- task-specific SA representation --+
   |                                                       |
   +-- Reaction-GINE --- reaction-informed representation -+
                                                           v
                                                Atom-wise Source Gating
                                                           |
                                                Attentive Molecular Readout
                                                           |
                                                + Explicit SA Priors
                                                           |
                                                           v
                                                          GESA
```

Both graph encoders use atom, bond, and connectivity information. Task-GAT uses
edge-aware graph attention; Reaction-GINE uses bond-conditioned message passing.
The aligned atom representations are fused before attentive molecular pooling and
concatenation with fourteen explicit SA priors.

## Quick start

Create the environment and expose the source package:

```bash
conda env create -f environment.yml
conda activate gesa
export PYTHONPATH="$PWD/src:${PYTHONPATH}"
```

Python API:

```python
from gesa import load_gesa, predict_smiles

model, metadata = load_gesa("weights/GESA.pt")
scores, errors = predict_smiles(model, ["CCO", "c1ccccc1"])
```

Command line:

```bash
python scripts/predict.py \
    --checkpoint weights/GESA.pt \
    --input molecules.csv \
    --output predictions.csv
```

## Label and score semantics

GESA uses `0 = ES` (easy to synthesize) and `1 = HS` (hard to synthesize).
HS is the positive class throughout training and evaluation. `GESA_score` is
`P(HS)` rather than an easy-to-synthesize probability, so larger values indicate
greater predicted synthetic difficulty. At the natural probability boundary,
`GESA_score >= 0.5` is classified as HS and `GESA_score < 0.5` as ES.

Accordingly, AUROC and AP use HS as the positive label; Recall, Precision,
and F1-score are HS-class metrics; Specificity is the true-negative rate for ES.
In confusion matrices, TP is a correctly predicted HS molecule and TN is a
correctly predicted ES molecule. For GESA, molecules with scores at or above
the threshold are classified as HS.

## External evaluation

Across seeds 42, 123, and 2026, AUROC was:

| Dataset | GESA AUROC, mean ± sample SD |
|---|---:|
| TS1 | 0.983047 ± 0.003802 |
| TS2 | 0.941277 ± 0.001076 |
| TS3 | 0.884411 ± 0.002281 |
| TS4 | 0.900972 ± 0.002585 |
| Equal-weight TS1–TS4 macro-average | 0.927427 ± 0.000451 |

These metrics can be recalculated from the released `y_true` and `y_score` arrays
using `scripts/evaluate_ts1_ts4.py`.

### Evaluation on held-out generators

The [generator-transfer source data](benchmarks/external/generator_transfer/README.md)
contain newly sampled NEAT-POCKET, FLOWR, and ResGen molecules, Retro* labels,
and predictions from GESA and all eight reference methods. GESA was not
retrained or tuned for these evaluations. Each generator is evaluated as a
separate cohort before and after restriction to SAScore `[3,4)`.

| Generator | Full cohort, N (ES/HS) | GESA AUROC | SAScore [3,4), N (ES/HS) | GESA AUROC |
|---|---:|---:|---:|---:|
| NEAT-POCKET | 835 (171/664) | 0.978 | 197 (70/127) | 0.916 |
| FLOWR | 945 (272/673) | 0.978 | 198 (96/102) | 0.926 |
| ResGen | 472 (78/394) | 0.975 | 115 (27/88) | 0.902 |

Start with [`evaluation_molecules.csv`](benchmarks/external/generator_transfer/data/evaluation_molecules.csv)
for the 2,252 evaluable molecules and nine-method scores. Original generated
structures, exclusion records, generator/checkpoint versions, per-pocket seeds,
confidence intervals, and Supplementary Figure S4 are included in the same directory.

Recalculate all AUROC/AP point estimates from the saved predictions:

```bash
python benchmarks/external/generator_transfer/code/reproduce_metrics.py
```

## Training and reproduction

Recalculate the released benchmark metrics:

```bash
python scripts/evaluate_ts1_ts4.py
```

Prepare the fixed data splits from the extracted Zenodo archive, then train GESA
from the released Task-GAT and Reaction-GINE encoder initializations:

```bash
python scripts/prepare_training_data.py --data-root /path/to/GESA_Data
python scripts/train_gesa.py --dataset-version original --seed 123
```

The [complete training guide](TRAINING.md) also provides commands for Task-GAT
pretraining, two-phase ORD pretraining, GESA training from newly pretrained
encoders, the three training seeds, cleaned-data retraining, and checkpoint
fine-tuning. Executable configurations are in
[`configs/training/`](configs/training/); historical settings and selection
records are in [`configs/pretraining/`](configs/pretraining/) and
[`training_provenance.json`](configs/training_provenance.json).

The released initialization checkpoints and SHA256 values are documented in
[`weights/checkpoint_metadata.json`](weights/checkpoint_metadata.json).

## Training data

**Companion dataset:** [Zenodo — 10.5281/zenodo.22874755](https://doi.org/10.5281/zenodo.22874755).

Benchmark and case-study source data are included in this repository. The
companion training-data record is linked above. To use the training data,
extract the dataset locally and set the training and validation paths in
`configs/gesa.yaml` at the repository root.

[Label provenance and dataset-version records](data/retrostar_target/label_audit/README.md)
document the 587,021-to-582,227 single-component cleanup, label provenance,
planning settings, route-length distributions, and within-source TS4 evaluation.
The [lossless version patch](data/retrostar_target/version_patch/README.md) restores
the original 587,021-record target splits from the released single-component data
and verifies all three original SHA256 values.

### Controlled training-data comparison

Source data and reproducible analyses for Supplementary Figure S5 are in
[`matched_data_contribution`](benchmarks/internal/matched_data_contribution/README.md).
This analysis compares three independently sampled pairs of class-balanced
12,000-molecule training subsets on TS1–TS4, with upstream pretraining held fixed.

## Repository map

- `assets/`: graphical overview of GESA

- `src/gesa/`: public model and inference API
- `src/gesa/training/`: molecular and reaction pretraining/training implementation
- `scripts/`: data preparation, three-stage training, prediction, and evaluation
- `TRAINING.md`: complete training procedure and dataset-version selection
- `configs/training/`: executable training configurations
- `weights/`: GESA checkpoint, pretrained encoder initializations, and checkpoint metadata
- `configs/pretraining/`: Task-GAT and Reaction-GINE settings and legacy split exclusions
- `benchmarks/external/`: nine-model external evaluation and source predictions
- `benchmarks/external/generator_transfer/`: three held-out generator cohorts,
  molecular structures, nine-method predictions, provenance and metric reproduction
- `benchmarks/internal/`: eight-model framework comparison
- `benchmarks/internal/matched_data_contribution/`: controlled training-data comparison and Figure S5 source data
- `case_studies/`: compact stepwise-route and SyntheMol manuscript source data
- `baselines/`: standard GNN baseline implementations
- `RELEASE_MANIFEST.csv`: file sizes and SHA256 checksums

Benchmark and case-study tables are retained as source data. The generator-transfer
analysis also includes Supplementary Figure S4 and its plotting source data.

## License

The original GESA source code is released under the [MIT License](LICENSE).
Third-party datasets and software retain their respective licenses and terms
of use.
