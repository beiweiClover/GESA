# Target dataset versions

`dataset_versions.json` identifies both target versions and records split counts,
class counts, byte sizes, and SHA256 values for all six CSV files.

| Split | Original rows | Released single-component rows | Removed rows |
|---|---:|---:|---:|
| Train | 472,635 | 469,078 | 3,557 |
| Validation | 59,084 | 58,394 | 690 |
| Internal test | 55,302 | 54,755 | 547 |
| Total | 587,021 | 582,227 | 4,794 |

The original version corresponds to the existing released GESA checkpoint.
The single-component version is distributed in the
[Zenodo data archive](https://doi.org/10.5281/zenodo.22874755).

`removed_rows.jsonl.gz` is a lossless restoration patch containing original row
positions and base64-encoded CSV lines. Retained lines and their order are identical
between versions. The approximately 0.43 MB patch restores the original files
byte for byte; their original SHA256 values are checked automatically.

From the repository root:

```bash
python scripts/prepare_training_data.py --data-root /path/to/GESA_Data
```

Both versions are prepared under separate directories in `training_data/`.
The full procedure is described in [`TRAINING.md`](../../../TRAINING.md).
