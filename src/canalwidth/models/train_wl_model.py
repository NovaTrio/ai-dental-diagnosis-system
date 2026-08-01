"""Train canal-centreline-length linear regression with LOOCV."""

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

from src.canalwidth.training.prepare_wl_dataset import (
    DEFAULT_LABELS,
    DEFAULT_MIDLINES,
    prepare_canal_wl_dataset,
)


DEFAULT_OUTPUT_DIR = Path("src/canalwidth/models/working_length")
DEFAULT_DATASET = Path("data/canalwidth/training/canal_wl_dataset.csv")


def train_canal_wl_model(
    dataset: pd.DataFrame,
) -> tuple[LinearRegression, pd.DataFrame, dict[str, float | int | str]]:
    if len(dataset) < 3:
        raise ValueError("At least three matched samples are required")
    x = dataset[["midline_length_px"]].to_numpy(dtype=float)
    y = dataset["actual_wl_mm"].to_numpy(dtype=float)
    predictions = np.empty(len(dataset), dtype=float)
    folds = np.empty(len(dataset), dtype=int)
    for fold, (train_indices, test_indices) in enumerate(
        LeaveOneOut().split(x), start=1
    ):
        model = LinearRegression().fit(x[train_indices], y[train_indices])
        predictions[test_indices] = model.predict(x[test_indices])
        folds[test_indices] = fold

    results = dataset.copy()
    results.insert(0, "fold", folds)
    results["predicted_wl_mm"] = predictions
    results["residual_mm"] = y - predictions
    results["absolute_error_mm"] = np.abs(results["residual_mm"])
    final_model = LinearRegression().fit(x, y)
    metrics: dict[str, float | int | str] = {
        "model": "Ordinary Linear Regression",
        "feature": "canal_midline_length_px",
        "target": "actual_wl_mm",
        "validation": "Leave-One-Out Cross-Validation",
        "n_samples": int(len(dataset)),
        "mae_mm": float(mean_absolute_error(y, predictions)),
        "rmse_mm": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
        "intercept_beta_0": float(final_model.intercept_),
        "slope_beta_1": float(final_model.coef_[0]),
        "equation": (
            f"WL_mm = {final_model.intercept_:.6f} + "
            f"{final_model.coef_[0]:.6f} * canal_midline_length_px"
        ),
    }
    return final_model, results, metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midlines", type=Path, default=DEFAULT_MIDLINES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = prepare_canal_wl_dataset(args.midlines, args.labels)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.dataset_output, index=False)
    model, predictions, metrics = train_canal_wl_model(dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "linear_regression.joblib")
    predictions.to_csv(args.output_dir / "loocv_predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(metrics["equation"])
    print(f"Samples:    {metrics['n_samples']}")
    print(f"LOOCV MAE:  {metrics['mae_mm']:.4f} mm")
    print(f"LOOCV RMSE: {metrics['rmse_mm']:.4f} mm")
    print(f"LOOCV R2:   {metrics['r2']:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
