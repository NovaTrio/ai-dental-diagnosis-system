"""Evaluate a locally weighted linear-regression working-length baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut


MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from training.prepare_dataset import load_and_prepare_dataset


def gaussian_weights(train_lengths: np.ndarray, test_length: float, tau: float) -> np.ndarray:
    """Return Gaussian similarity weights for one held-out tooth."""
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number")
    distances = train_lengths.reshape(-1) - test_length
    return np.exp(-(distances**2) / (2.0 * tau**2))


def evaluate_lwlr(csv_path: str | Path, tau: float = 30.0) -> tuple[pd.DataFrame, dict]:
    """Fit a separate weighted regression for every LOOCV test sample."""
    dataset = load_and_prepare_dataset(csv_path)
    if len(dataset) < 3:
        raise ValueError("At least three samples are required for LOOCV regression")

    x = dataset[["midline_length_px"]].to_numpy(dtype=float)
    y = dataset["actual_wl_mm"].to_numpy(dtype=float)
    rows: list[dict] = []

    for fold, (train_indices, test_indices) in enumerate(LeaveOneOut().split(x), start=1):
        test_index = int(test_indices[0])
        test_length = float(x[test_index, 0])
        weights = gaussian_weights(x[train_indices, 0], test_length, tau)
        if np.count_nonzero(weights > np.finfo(float).eps) < 2:
            raise ValueError(f"tau={tau} leaves fewer than two effective samples in fold {fold}")

        model = LinearRegression()
        model.fit(x[train_indices], y[train_indices], sample_weight=weights)
        prediction = float(model.predict(x[test_indices])[0])
        residual = float(y[test_index] - prediction)
        effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
        rows.append(
            {
                "fold": fold,
                "image_name": dataset.loc[test_index, "image_name"],
                "midline_length_px": test_length,
                "actual_wl_mm": float(y[test_index]),
                "predicted_wl_mm": prediction,
                "residual_mm": residual,
                "absolute_error_mm": abs(residual),
                "local_intercept_beta_0": float(model.intercept_),
                "local_slope_beta_1": float(model.coef_[0]),
                "maximum_training_weight": float(weights.max()),
                "effective_training_samples": effective_n,
            }
        )

    predictions = pd.DataFrame(rows)
    actual = predictions["actual_wl_mm"].to_numpy()
    predicted = predictions["predicted_wl_mm"].to_numpy()
    residual = actual - predicted
    percentage_errors = residual / actual * 100.0
    metrics = {
        "model": "Locally Weighted Linear Regression",
        "validation": "Leave-One-Out Cross-Validation",
        "bandwidth_tau_px": float(tau),
        "n_samples": int(len(dataset)),
        "n_folds": int(len(dataset)),
        "training_samples_per_fold": int(len(dataset) - 1),
        "test_samples_per_fold": 1,
        "mae_mm": float(mean_absolute_error(actual, predicted)),
        "rmse_mm": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mean_error_mm": float(np.mean(residual)),
        "mean_percentage_error_pct": float(np.mean(percentage_errors)),
        "mean_absolute_percentage_error_pct": float(np.mean(np.abs(percentage_errors))),
        "residual_sd_mm": float(np.std(residual, ddof=1)),
        "mean_effective_training_samples": float(
            predictions["effective_training_samples"].mean()
        ),
    }
    return predictions, metrics


def save_residual_plots(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """Save residual-versus-prediction and residual-distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(predictions["predicted_wl_mm"], predictions["residual_mm"], alpha=0.8)
    axes[0].axhline(0, color="tab:red", linestyle="--", linewidth=1.5)
    axes[0].set(xlabel="Predicted WL (mm)", ylabel="Residual: actual - predicted (mm)")
    axes[0].grid(alpha=0.25)

    axes[1].hist(predictions["residual_mm"], bins="auto", edgecolor="white")
    axes[1].axvline(0, color="tab:red", linestyle="--", linewidth=1.5)
    axes[1].set(xlabel="Residual (mm)", ylabel="Count")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Locally Weighted Regression LOOCV Residuals")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(weighted_metrics: dict, baseline_metrics_path: Path, output: Path) -> None:
    """Write a compact Baseline A versus Baseline B metrics table."""
    rows = []
    if baseline_metrics_path.is_file():
        with baseline_metrics_path.open(encoding="utf-8") as file:
            baseline = json.load(file)
        rows.append(
            {
                "model": "Ordinary Linear Regression",
                "mae_mm": baseline["mae_mm"],
                "rmse_mm": baseline["rmse_mm"],
                "r2": baseline["r2"],
            }
        )
    rows.append(
        {
            "model": weighted_metrics["model"],
            "mae_mm": weighted_metrics["mae_mm"],
            "rmse_mm": weighted_metrics["rmse_mm"],
            "r2": weighted_metrics["r2"],
        }
    )
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=MODULE_ROOT / "training" / "wl_labels.csv",
    )
    parser.add_argument("--tau", type=float, default=30.0, help="Gaussian bandwidth in pixels")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    predictions, metrics = evaluate_lwlr(args.input, args.tau)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "loocv_predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_residual_plots(predictions, args.output_dir / "residual_plots.png")
    save_model_comparison(
        metrics,
        Path(__file__).resolve().parents[1] / "baseline_a" / "metrics.json",
        args.output_dir / "model_comparison.csv",
    )

    print(f"Bandwidth tau: {metrics['bandwidth_tau_px']:.2f} px")
    print(f"LOOCV MAE:  {metrics['mae_mm']:.4f} mm")
    print(f"LOOCV RMSE: {metrics['rmse_mm']:.4f} mm")
    print(f"LOOCV R²:   {metrics['r2']:.4f}")
    print(f"Artifacts saved: {args.output_dir}")


if __name__ == "__main__":
    main()
