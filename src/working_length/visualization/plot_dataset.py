"""Visualize the baseline relationship between tooth length and EAL WL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from training.prepare_dataset import load_and_prepare_dataset


def plot_midline_vs_working_length(
    csv_path: str | Path, output_path: str | Path
) -> tuple[float, float]:
    """Create the baseline scatter plot and return Pearson r and R-squared."""
    dataset = load_and_prepare_dataset(csv_path)
    x = dataset["midline_length_px"].to_numpy(dtype=float)
    y = dataset["actual_wl_mm"].to_numpy(dtype=float)

    if len(dataset) < 2 or np.ptp(x) == 0:
        raise ValueError("At least two samples with different midline lengths are required")

    slope, intercept = np.polyfit(x, y, 1)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    r_squared = pearson_r**2
    line_x = np.linspace(x.min(), x.max(), 200)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=55, alpha=0.8, edgecolor="white", linewidth=0.7)
    ax.plot(
        line_x,
        slope * line_x + intercept,
        color="tab:red",
        linewidth=2,
        label=f"Linear trend (r = {pearson_r:.3f}, R² = {r_squared:.3f})",
    )
    ax.set(
        title="Radiographic Midline Length vs Actual Working Length",
        xlabel="Midline Length (px)",
        ylabel="Actual WL (mm)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pearson_r, r_squared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=MODULE_ROOT / "training" / "wl_labels.csv",
        help="CSV containing image_name, midline_length_px, and actual_wl_mm",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/working_length/visualizations/midline_vs_actual_wl.png"),
        help="Destination PNG path",
    )
    args = parser.parse_args()

    pearson_r, r_squared = plot_midline_vs_working_length(args.input, args.output)
    print(f"Plot saved: {args.output}")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"R-squared (single linear feature): {r_squared:.4f}")


if __name__ == "__main__":
    main()
