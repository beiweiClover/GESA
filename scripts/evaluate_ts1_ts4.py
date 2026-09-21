#!/usr/bin/env python3
"""Recompute all released external metrics from aligned labels and scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "benchmarks" / "external" / "predictions"
OUTPUT = ROOT / "benchmarks" / "external"
SPLITS = ["TS1", "TS2", "TS3", "TS4"]
SEEDS = [42, 123, 2026]
REFERENCE_MODELS = [
    "SAScore", "SCScore", "SYBA", "BR-SAScore", "RAscore-XGB",
    "GASA", "SynFrag", "DeepSA",
]
MODEL_ORDER = REFERENCE_MODELS + ["GESA"]
FIXED_HARD_THRESHOLDS = {
    "SYBA": 0.0,
    "BR-SAScore": 4.0 / 9.0,
    "RAscore-XGB": 0.5,
    "GASA": 0.5,
    "SynFrag": 0.5,
    "DeepSA": 0.5,
    "GESA": 0.5,
}
THRESHOLD_SOURCES = {
    "SYBA": "official_definition",
    "BR-SAScore": "paper_definition",
    "RAscore-XGB": "official_definition",
    "GASA": "official_definition",
    "SynFrag": "official_definition",
    "DeepSA": "official_definition",
    "GESA": "natural_probability_boundary",
}


def optimal_threshold(y_true: np.ndarray, hard_score: np.ndarray) -> float:
    """Return the highest observed threshold tied for maximum Youden J."""
    fpr, tpr, thresholds = roc_curve(
        y_true, hard_score, drop_intermediate=False
    )
    youden = tpr - fpr
    maximum = np.nanmax(youden)
    tied = np.flatnonzero(np.isclose(youden, maximum, rtol=0.0, atol=1e-15))
    return float(thresholds[int(tied[0])])


def raw_threshold_and_rule(model: str, hard_threshold: float) -> tuple[float, str, str]:
    if model in {"SAScore", "SCScore"}:
        return hard_threshold, model, f"{model} >= {hard_threshold:.8g}"
    if model == "SYBA":
        raw = -hard_threshold
        return raw, "raw SYBA score", f"SYBA <= {raw:.8g}"
    if model == "BR-SAScore":
        raw = 1.0 + 9.0 * hard_threshold
        return raw, "BR-SAScore 1--10", f"BR-SAScore >= {raw:.8g}"
    if model in {"RAscore-XGB", "SynFrag"}:
        raw = 1.0 - hard_threshold
        return raw, f"{model} easy score", f"{model} <= {raw:.8g}"
    return hard_threshold, f"{model} HS probability", f"{model} >= {hard_threshold:.8g}"


def evaluate_operating_point(
    dataset: str,
    model: str,
    y_true: np.ndarray,
    hard_score: np.ndarray,
    evaluation_unit: str,
) -> dict[str, object]:
    n_total = len(y_true)
    valid = np.isfinite(y_true) & np.isfinite(hard_score)
    y = y_true[valid].astype(np.int64)
    score = hard_score[valid].astype(np.float64)
    if model in {"SAScore", "SCScore"}:
        threshold = optimal_threshold(y, score)
        threshold_source = "maximum_Youden_J_on_same_external_test"
        tie_rule = "highest_hard_score_threshold"
    else:
        threshold = FIXED_HARD_THRESHOLDS[model]
        threshold_source = THRESHOLD_SOURCES[model]
        tie_rule = "not_applicable"
    prediction = (score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    raw_threshold, scale, rule = raw_threshold_and_rule(model, threshold)
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    return {
        "dataset": dataset,
        "model": model,
        "evaluation_unit": evaluation_unit,
        "n_total": int(n_total),
        "n_scored": int(valid.sum()),
        "n_failed": int(n_total - valid.sum()),
        "n_ES": int((y == 0).sum()),
        "n_HS": int((y == 1).sum()),
        "threshold": raw_threshold,
        "threshold_scale": scale,
        "hard_score_threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_tie_rule": tie_rule,
        "classification_rule": rule,
        "accuracy": accuracy_score(y, prediction),
        "AUROC": roc_auc_score(y, score),
        "specificity": specificity,
        "PRAUC": average_precision_score(y, score),
        "recall": recall,
        "F_score": f1_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "Youden_J": recall + specificity - 1.0,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def validate_alignment(
    reference: pd.DataFrame,
    gesa: pd.DataFrame,
    public: pd.DataFrame,
) -> None:
    for split in SPLITS:
        expected = reference.loc[reference.split.eq(split)].sort_values("row_index")
        expected_keys = expected[["molecule_id", "y_true"]].reset_index(drop=True)
        selected_public = public.loc[public.split.eq(split)].sort_values("row_index")
        if not expected_keys.equals(
            selected_public[["molecule_id", "y_true"]].reset_index(drop=True)
        ):
            raise RuntimeError(f"Public-checkpoint label/order mismatch for {split}")
        for seed in SEEDS:
            selected = gesa.loc[
                gesa.split.eq(split) & gesa.seed.eq(seed)
            ].sort_values("row_index")
            if not expected_keys.equals(
                selected[["molecule_id", "y_true"]].reset_index(drop=True)
            ):
                raise RuntimeError(f"GESA label/order mismatch for {split}, seed {seed}")
            if seed == 123 and not np.allclose(
                selected.y_score.to_numpy(np.float64),
                selected_public.y_score.to_numpy(np.float64),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise RuntimeError(f"Seed-123 scores do not match public checkpoint for {split}")


def build_auc_tables(
    reference: pd.DataFrame,
    gesa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for split in SPLITS:
        ref_split = reference.loc[reference.split.eq(split)].sort_values("row_index")
        for model in REFERENCE_MODELS:
            label = ref_split.y_true.to_numpy(np.int64)
            score = ref_split[model].to_numpy(np.float64)
            valid = np.isfinite(score)
            rows.append({
                "Model": model,
                "Split": split,
                "Seed": "reference",
                "AUROC": roc_auc_score(label[valid], score[valid]),
                "N": int(len(label)),
                "N_scored": int(valid.sum()),
            })
        for seed in SEEDS:
            selected = gesa.loc[
                gesa.split.eq(split) & gesa.seed.eq(seed)
            ].sort_values("row_index")
            label = selected.y_true.to_numpy(np.int64)
            score = selected.y_score.to_numpy(np.float64)
            valid = np.isfinite(score)
            rows.append({
                "Model": "GESA",
                "Split": split,
                "Seed": str(seed),
                "AUROC": roc_auc_score(label[valid], score[valid]),
                "N": int(len(label)),
                "N_scored": int(valid.sum()),
            })
    long = pd.DataFrame(rows)

    summary_rows = []
    for model in MODEL_ORDER:
        row = {"Model": model}
        split_means = []
        for split in SPLITS:
            values = long.loc[
                long.Model.eq(model) & long.Split.eq(split), "AUROC"
            ]
            row[split] = values.mean()
            row[f"{split}_SD"] = values.std(ddof=1) if len(values) > 1 else np.nan
            split_means.append(values.mean())
        row["TS1_TS4_Macro"] = float(np.mean(split_means))
        if model == "GESA":
            seed_macro = (
                long.loc[long.Model.eq(model)]
                .pivot(index="Seed", columns="Split", values="AUROC")[SPLITS]
                .mean(axis=1)
            )
            row["TS1_TS4_Macro_SD"] = seed_macro.std(ddof=1)
        else:
            row["TS1_TS4_Macro_SD"] = np.nan
        summary_rows.append(row)
    return long, pd.DataFrame(summary_rows)


def build_full_metrics(
    reference: pd.DataFrame,
    public: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        ref = reference.loc[reference.split.eq(split)].sort_values("row_index")
        y_ref = ref.y_true.to_numpy(np.float64)
        for model in REFERENCE_MODELS:
            rows.append(evaluate_operating_point(
                split,
                model,
                y_ref,
                ref[model].to_numpy(np.float64),
                "released_reference_prediction",
            ))
        selected = public.loc[public.split.eq(split)].sort_values("row_index")
        rows.append(evaluate_operating_point(
            split,
            "GESA",
            selected.y_true.to_numpy(np.float64),
            selected.y_score.to_numpy(np.float64),
            "public_checkpoint_seed_123",
        ))
    result = pd.DataFrame(rows)
    result["dataset"] = pd.Categorical(result.dataset, SPLITS, ordered=True)
    result["model"] = pd.Categorical(result.model, MODEL_ORDER, ordered=True)
    result = result.sort_values(["dataset", "model"]).reset_index(drop=True)
    result["dataset"] = result.dataset.astype(str)
    result["model"] = result.model.astype(str)
    return result


def main() -> None:
    reference = pd.read_csv(PREDICTIONS / "reference_models.csv.gz")
    gesa = pd.read_csv(PREDICTIONS / "GESA.csv")
    public = pd.read_csv(PREDICTIONS / "GESA_public_checkpoint.csv")
    validate_alignment(reference, gesa, public)

    long, summary = build_auc_tables(reference, gesa)
    long.to_csv(OUTPUT / "auc_results_long.csv", index=False)
    summary.to_csv(OUTPUT / "auc_results.csv", index=False)
    build_full_metrics(reference, public).to_csv(
        OUTPUT / "full_metrics.csv", index=False
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
