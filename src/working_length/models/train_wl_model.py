"""Train and evaluate the midline-length linear-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from training.prepare_dataset import load_and_prepare_dataset


def train_loocv_baseline(csv_path: str | Path) -> tuple[LinearRegression, object, dict]:
    """Evaluate with LOOCV and fit a final model on all available samples."""
    dataset = load_and_prepare_dataset(csv_path)
    if len(dataset) < 3:
        raise ValueError("At least three samples are required for LOOCV regression")

    x = dataset[["midline_length_px"]].to_numpy(dtype=float)
    y = dataset["actual_wl_mm"].to_numpy(dtype=float)
    predictions = np.empty(len(dataset), dtype=float)
    fold_numbers = np.empty(len(dataset), dtype=int)

    loo = LeaveOneOut()
    for fold, (train_indices, test_indices) in enumerate(loo.split(x), start=1):
        fold_model = LinearRegression()
        fold_model.fit(x[train_indices], y[train_indices])
        predictions[test_indices] = fold_model.predict(x[test_indices])
        fold_numbers[test_indices] = fold

    residuals = y - predictions
    results = dataset.copy()
    results.insert(0, "fold", fold_numbers)
    results["predicted_wl_mm"] = predictions
    results["residual_mm"] = residuals
    results["absolute_error_mm"] = np.abs(residuals)

    final_model = LinearRegression().fit(x, y)
    metrics = {
        "validation": "Leave-One-Out Cross-Validation",
        "n_samples": int(len(dataset)),
        "n_folds": int(len(dataset)),
        "training_samples_per_fold": int(len(dataset) - 1),
        "test_samples_per_fold": 1,
        "feature": "midline_length_px",
        "target": "actual_wl_mm",
        "mae_mm": float(mean_absolute_error(y, predictions)),
        "rmse_mm": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
        "mean_error_mm": float(np.mean(residuals)),
        "residual_sd_mm": float(np.std(residuals, ddof=1)),
        "final_intercept_beta_0": float(final_model.intercept_),
        "final_slope_beta_1": float(final_model.coef_[0]),
        "equation": (
            f"WL_mm = {final_model.intercept_:.6f} + "
            f"{final_model.coef_[0]:.6f} * midline_length_px"
        ),
    }
    return final_model, results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=MODULE_ROOT / "training" / "wl_labels.csv",
        help="Validated label CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_a",
        help="Directory for the model, metrics, and LOOCV predictions",
    )
    args = parser.parse_args()

    model, predictions, metrics = train_loocv_baseline(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "linear_regression.joblib")
    predictions.to_csv(args.output_dir / "loocv_predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(metrics["equation"])
    print(f"LOOCV MAE:  {metrics['mae_mm']:.4f} mm")
    print(f"LOOCV RMSE: {metrics['rmse_mm']:.4f} mm")
    print(f"LOOCV R²:   {metrics['r2']:.4f}")
    print(f"Artifacts saved: {args.output_dir}")


if __name__ == "__main__":
    main()
