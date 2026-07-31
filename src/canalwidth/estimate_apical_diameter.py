"""Create radiographic apical-diameter estimates from median canal widths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WIDTHS = Path("data/canalwidth/widths/canal_widths.csv")
DEFAULT_OUTPUT = Path("data/canalwidth/apical_diameter/estimates.csv")


def estimate_apical_diameters(
    widths_csv: Path,
    mm_per_pixel: float | None = None,
) -> pd.DataFrame:
    if not widths_csv.is_file():
        raise FileNotFoundError(f"Canal-width CSV not found: {widths_csv}")
    if mm_per_pixel is not None and (
        not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0
    ):
        raise ValueError("mm_per_pixel must be a positive finite value")
    table = pd.read_csv(widths_csv)
    required = {
        "image_name",
        "predicted_wl_mm",
        "median_canal_width_px",
        "width_valid",
    }
    if missing := required - set(table.columns):
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")
    valid = table["width_valid"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    estimates = table.loc[
        valid,
        ["image_name", "predicted_wl_mm", "median_canal_width_px"],
    ].copy()
    estimates = estimates.rename(
        columns={"median_canal_width_px": "apical_diameter_px"}
    )
    estimates["apical_diameter_px"] = pd.to_numeric(
        estimates["apical_diameter_px"], errors="coerce"
    )
    estimates = estimates.dropna(subset=["apical_diameter_px"])
    if mm_per_pixel is not None:
        estimates["mm_per_pixel"] = mm_per_pixel
        estimates["apical_diameter_mm"] = (
            estimates["apical_diameter_px"] * mm_per_pixel
        )
    return estimates.reset_index(drop=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=Path, default=DEFAULT_WIDTHS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=None,
        help="Optional radiographic calibration; omit to retain pixel units",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    estimates = estimate_apical_diameters(args.widths, args.mm_per_pixel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(args.output, index=False)
    unit = "px and mm" if args.mm_per_pixel is not None else "px"
    print(f"Saved {len(estimates)} apical-diameter estimate(s) in {unit}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
