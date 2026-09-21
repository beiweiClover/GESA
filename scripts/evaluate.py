#!/usr/bin/env python3
"""Evaluate GESA scores with HS=1 as the positive class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GESA prediction scores")
    parser.add_argument("--input", required=True)
    parser.add_argument("--label-column", default="y_true")
    parser.add_argument("--score-column", default="GESA_score")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    label = frame[args.label_column].to_numpy(dtype=np.int64)
    score = frame[args.score_column].to_numpy(dtype=np.float64)
    valid = np.isfinite(score)
    if not set(np.unique(label)).issubset({0, 1}):
        raise ValueError("Labels must use ES=0 and HS=1")
    if len(np.unique(label[valid])) != 2:
        raise ValueError("AUROC requires both binary classes after score filtering")
    result = {
        "positive_class": "HS (label=1)",
        "negative_class": "ES (label=0)",
        "score_semantics": "GESA_score=P(HS); higher means harder to synthesize",
        "rows": int(len(frame)),
        "scored": int(valid.sum()),
        "auroc": float(roc_auc_score(label[valid], score[valid])),
        "auprc": float(average_precision_score(label[valid], score[valid])),
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
