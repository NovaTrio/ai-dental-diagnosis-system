"""Join valid canal-centreline lengths to measured working-length labels."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MIDLINES = Path("data/canalwidth/midlines/canal_midlines.csv")
DEFAULT_LABELS = Path("data/working_length/labels/wl_labels.csv")
DEFAULT_OUTPUT = Path("data/canalwidth/training/canal_wl_dataset.csv")
TOOTH_NAME = re.compile(r"^(?P<image>.+)_tooth(?:_(?P<root>\d+))?$")


def measurement_key(filename: str) -> tuple[str, str]:
    """Convert `<image>_tooth[_N]` into the clinical `(image, root)` key."""
    stem = Path(str(filename)).stem
    match = TOOTH_NAME.match(stem)
    if not match:
        return stem, "R1"
    return match.group("image"), f"R{match.group('root') or '1'}"


def prepare_canal_wl_dataset(
    midline_csv: str | Path,
    labels_csv: str | Path,
) -> pd.DataFrame:
    midline_csv = Path(midline_csv)
    labels_csv = Path(labels_csv)
    if not midline_csv.is_file():
        raise FileNotFoundError(f"Canal midline CSV not found: {midline_csv}")
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Working-length labels not found: {labels_csv}")

    midlines = pd.read_csv(midline_csv)
    labels = pd.read_csv(labels_csv)
    required_midlines = {"image_name", "midline_detected", "midline_length_px"}
    required_labels = {"image_name", "root_id", "actual_wl_mm"}
    if missing := required_midlines - set(midlines.columns):
        raise ValueError(f"Missing midline column(s): {', '.join(sorted(missing))}")
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"Missing label column(s): {', '.join(sorted(missing))}")

    detected = midlines["midline_detected"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    midlines = midlines.loc[detected].copy()
    keys = midlines["image_name"].map(measurement_key)
    midlines["image_key"] = keys.map(lambda value: value[0])
    midlines["root_id"] = keys.map(lambda value: value[1])
    midlines["midline_length_px"] = pd.to_numeric(
        midlines["midline_length_px"], errors="coerce"
    )

    labels = labels.copy()
    labels["image_key"] = labels["image_name"].map(
        lambda name: Path(str(name)).stem
    )
    labels["root_id"] = labels["root_id"].fillna("").astype(str).str.strip()
    labels["root_id"] = labels["root_id"].mask(labels["root_id"].eq(""), "R1")
    labels["actual_wl_mm"] = pd.to_numeric(labels["actual_wl_mm"], errors="coerce")

    duplicate_midlines = midlines.duplicated(["image_key", "root_id"], keep=False)
    duplicate_labels = labels.duplicated(["image_key", "root_id"], keep=False)
    if duplicate_midlines.any():
        raise ValueError("Duplicate detected canal midline key(s)")
    if duplicate_labels.any():
        ambiguous_keys = labels.loc[
            duplicate_labels, ["image_key", "root_id"]
        ].drop_duplicates()
        formatted = ", ".join(
            f"{row.image_key}/{row.root_id}"
            for row in ambiguous_keys.itertuples(index=False)
        )
        warnings.warn(
            f"Excluding ambiguous duplicate working-length key(s): {formatted}",
            stacklevel=2,
        )
        labels = labels.loc[~duplicate_labels].copy()

    joined = midlines.merge(
        labels[["image_key", "root_id", "actual_wl_mm"]],
        on=["image_key", "root_id"],
        how="inner",
        validate="one_to_one",
    )
    prepared = joined[
        ["image_name", "root_id", "midline_length_px", "actual_wl_mm"]
    ].copy()
    valid = (
        np.isfinite(prepared["midline_length_px"])
        & np.isfinite(prepared["actual_wl_mm"])
        & prepared["midline_length_px"].gt(0)
        & prepared["actual_wl_mm"].gt(0)
    )
    prepared = prepared.loc[valid].reset_index(drop=True)
    if prepared.empty:
        raise ValueError("No valid canal-midline/working-length matches")
    return prepared


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midlines", type=Path, default=DEFAULT_MIDLINES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset = prepare_canal_wl_dataset(args.midlines, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Prepared {len(dataset)} canal working-length samples: {args.output}")


if __name__ == "__main__":
    main()
