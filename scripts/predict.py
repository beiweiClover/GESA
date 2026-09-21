#!/usr/bin/env python3
"""Predict synthetic-accessibility scores from a CSV of SMILES."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gesa import load_gesa, predict_smiles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="GESA synthetic accessibility model")
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "GESA.pt"))
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, keep_default_na=False)
    if args.smiles_column not in frame:
        raise KeyError(f"Missing SMILES column: {args.smiles_column}")
    model, _ = load_gesa(args.checkpoint, args.device)
    scores, errors = predict_smiles(
        model, frame[args.smiles_column].astype(str).tolist(), batch_size=args.batch_size
    )
    frame["GESA_score"] = scores
    frame["GESA_error"] = errors
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
