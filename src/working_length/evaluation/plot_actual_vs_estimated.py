"""Plot labelled clinical values against leave-one-out system estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_WL_PREDICTIONS = Path(
    "src/canalwidth/models/working_length/loocv_predictions.csv"
)
DEFAULT_FILE_PREDICTIONS = Path(
    "src/canalwidth/models/binding_file/loocv_predictions.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/working_length/evaluation_plots")


def _identity_limits(actual: np.ndarray, *estimates: np.ndarray) -> tuple[float, float]:
    values = np.concatenate((actual, *estimates)).astype(float)
    span = float(values.max() - values.min())
    padding = max(0.5, span * 0.08)
    return float(values.min() - padding), float(values.max() + padding)


def plot_working_length(predictions: pd.DataFrame, output_path: Path) -> None:
    actual = predictions["actual_wl_mm"].to_numpy(dtype=float)
    estimated = predictions["predicted_wl_mm"].to_numpy(dtype=float)
    lower, upper = _identity_limits(actual, estimated)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(actual, estimated, s=55, color="#2563eb", edgecolor="white")
    axis.plot([lower, upper], [lower, upper], "--", color="#dc2626", label="Perfect agreement")
    axis.set(
        title="Working Length: Actual vs Estimated (LOOCV)",
        xlabel="Actual labelled working length (mm)",
        ylabel="System-estimated working length (mm)",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_file_recommendation(predictions: pd.DataFrame, output_path: Path) -> None:
    actual = predictions["first_binding_file_size_k"].to_numpy(dtype=float)
    continuous = predictions["continuous_prediction_k"].to_numpy(dtype=float)
    recommended = predictions["recommended_iso_size_k"].to_numpy(dtype=float)
    lower, upper = _identity_limits(actual, continuous, recommended)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        actual,
        continuous,
        s=50,
        color="#f59e0b",
        marker="o",
        label="Continuous estimate",
        alpha=0.85,
    )
    axis.scatter(
        actual,
        recommended,
        s=60,
        color="#7c3aed",
        marker="x",
        linewidth=1.8,
        label="Nearest-ISO recommendation",
    )
    axis.plot([lower, upper], [lower, upper], "--", color="#dc2626", label="Perfect agreement")
    axis.set(
        title="Binding File/GP Size: Actual vs Estimated (LOOCV)",
        xlabel="Actual labelled first-binding file size (K)",
        ylabel="System-estimated file size (K)",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wl-predictions", type=Path, default=DEFAULT_WL_PREDICTIONS)
    parser.add_argument("--file-predictions", type=Path, default=DEFAULT_FILE_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.wl_predictions.is_file():
        raise FileNotFoundError(f"Working-length predictions not found: {args.wl_predictions}")
    if not args.file_predictions.is_file():
        raise FileNotFoundError(f"Binding-file predictions not found: {args.file_predictions}")

    wl_output = args.output_dir / "working_length_actual_vs_estimated.png"
    file_output = args.output_dir / "gp_file_size_actual_vs_estimated.png"
    plot_working_length(pd.read_csv(args.wl_predictions), wl_output)
    plot_file_recommendation(pd.read_csv(args.file_predictions), file_output)
    print(f"Saved: {wl_output}")
    print(f"Saved: {file_output}")


if __name__ == "__main__":
    main()
