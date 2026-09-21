# External benchmark data

`TS1.csv.gz` through `TS4.csv.gz` preserve benchmark row order, molecule identifier,
SMILES, and binary label. They are generated from the immutable benchmark prediction
table and checked against each GESA prediction array before AUROC calculation.

The binary label is `y_true`, with ES=`0` and HS=`1`. ROC and PR calculations
therefore use HS as the positive label.

| Split | Molecules | Uncompressed CSV SHA256 |
|---|---:|---|
| TS1 | 7,162 | `c1083c998fe271afbbc7a593ffef0203bc887f0b9231405e830e3ee2f8de3ab0` |
| TS2 | 30,346 | `44a4b0fa1eff2d7f19d7161d7855e7eb42a71e7860bc4672cb63016c896c2e33` |
| TS3 | 1,785 | `f4d001d427d118b7536a82019bca0f09ac5654cf2dd82410848a0753d7ecddf0` |
| TS4 | 3,584 | `f5bfcf0e5725ad9ac8eb33165ebe70ec003a4fd16e4b1c8df702d0ae0bd3c9fe` |

Every released benchmark SMILES represents one connected molecular component.
