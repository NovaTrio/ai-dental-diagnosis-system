"""Predict continuous first-binding-file size, then map to nearest ISO size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

from src.canalwidth.training.prepare_binding_file_dataset import (
    DEFAULT_LABELS,
    DEFAULT_WIDTHS,
    DEFAULT_WL_LOOCV,
    prepare_binding_file_dataset,
)


ISO_FILE_SIZES = np.asarray(
    [6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    dtype=float,
)
DEFAULT_DATASET = Path("data/canalwidth/training/binding_file_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("src/canalwidth/models/binding_file")
FEATURES = ["predicted_wl_mm", "apical_diameter_px"]


def nearest_iso_size(
    values: np.ndarray | list[float] | float,
) -> np.ndarray:
    array = np.atleast_1d(np.asarray(values, dtype=float))
    indices = np.abs(array[:, None] - ISO_FILE_SIZES[None, :]).argmin(axis=1)
    return ISO_FILE_SIZES[indices].astype(int)


def train_binding_file_model(
    dataset: pd.DataFrame,
) -> tuple[LinearRegression, pd.DataFrame, dict[str, object]]:
    if len(dataset) < 5:
        raise ValueError("At least five matched samples are required")
    x = dataset[FEATURES].to_numpy(dtype=float)
    y = dataset["first_binding_file_size_k"].to_numpy(dtype=float)
    continuous = np.empty(len(dataset), dtype=float)
    folds = np.empty(len(dataset), dtype=int)
    for fold, (train_indices, test_indices) in enumerate(
        LeaveOneOut().split(x), start=1
    ):
        model = LinearRegression().fit(x[train_indices], y[train_indices])
        continuous[test_indices] = model.predict(x[test_indices])
        folds[test_indices] = fold
    rounded = nearest_iso_size(continuous)
    actual_iso = nearest_iso_size(y)
    results = dataset.copy()
    results.insert(0, "fold", folds)
    results["continuous_prediction_k"] = continuous
    results["recommended_iso_size_k"] = rounded
    results["absolute_error_k"] = np.abs(y - continuous)
    final_model = LinearRegression().fit(x, y)
    metrics: dict[str, object] = {
        "model": "Ordinary Linear Regression with nearest-ISO postprocessing",
        "features": FEATURES,
        "target": "first_binding_file_size_k",
        "validation": "Leave-One-Out Cross-Validation",
        "n_samples": int(len(dataset)),
        "continuous_mae_k": float(mean_absolute_error(y, continuous)),
        "continuous_rmse_k": float(np.sqrt(mean_squared_error(y, continuous))),
        "continuous_r2": float(r2_score(y, continuous)),
        "exact_iso_accuracy": float(np.mean(rounded == actual_iso)),
        "within_5_size_units_accuracy": float(np.mean(np.abs(rounded - actual_iso) <= 5)),
        "intercept": float(final_model.intercept_),
        "coefficients": {
            feature: float(coefficient)
            for feature, coefficient in zip(FEATURES, final_model.coef_)
        },
    }
    return final_model, results, metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=Path, default=DEFAULT_WIDTHS)
    parser.add_argument("--wl-predictions", type=Path, default=DEFAULT_WL_LOOCV)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = prepare_binding_file_dataset(
        args.widths,
        args.wl_predictions,
        args.labels,
    )
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.dataset_output, index=False)
    model, predictions, metrics = train_binding_file_model(dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "linear_regression.joblib")
    predictions.to_csv(args.output_dir / "loocv_predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Samples: {metrics['n_samples']}")
    print(f"Continuous MAE: {metrics['continuous_mae_k']:.3f} size units")
    print(f"Continuous RMSE: {metrics['continuous_rmse_k']:.3f} size units")
    print(f"Continuous R2: {metrics['continuous_r2']:.4f}")
    print(f"Exact ISO accuracy: {100 * metrics['exact_iso_accuracy']:.1f}%")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
