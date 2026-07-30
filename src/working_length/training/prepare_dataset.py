"""Validate and prepare the baseline working-length regression dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("image_name", "midline_length_px", "actual_wl_mm")


def load_and_prepare_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a label CSV and return the validated baseline modeling columns.

    The source file may contain additional clinical metadata. It is deliberately
    excluded from this baseline so that the only predictor is the radiographic
    tooth midline length and the target is the EAL working length.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    dataset = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    prepared = dataset.loc[:, REQUIRED_COLUMNS].copy()
    prepared["image_name"] = prepared["image_name"].astype("string").str.strip()
    for column in ("midline_length_px", "actual_wl_mm"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    invalid = (
        prepared["image_name"].isna()
        | prepared["image_name"].eq("")
        | prepared[["midline_length_px", "actual_wl_mm"]].isna().any(axis=1)
        | ~np.isfinite(prepared["midline_length_px"])
        | ~np.isfinite(prepared["actual_wl_mm"])
        | prepared["midline_length_px"].le(0)
        | prepared["actual_wl_mm"].le(0)
    )
    if invalid.any():
        rows = ", ".join(str(index + 2) for index in prepared.index[invalid])
        raise ValueError(f"Invalid or non-positive values on CSV row(s): {rows}")

    return prepared.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("wl_labels.csv"),
        help="Input label CSV (default: wl_labels.csv beside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/working_length/prepared/baseline_dataset.csv"),
        help="Destination for the validated three-column dataset",
    )
    args = parser.parse_args()

    prepared = load_and_prepare_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(f"Prepared {len(prepared)} samples: {args.output}")


if __name__ == "__main__":
    main()
