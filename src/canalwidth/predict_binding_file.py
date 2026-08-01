"""Predict continuous binding-file size and recommend the nearest ISO size."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.canalwidth.models.train_binding_file_model import (
    FEATURES,
    nearest_iso_size,
)


DEFAULT_MODEL = Path(
    "src/canalwidth/models/binding_file/linear_regression.joblib"
)
DEFAULT_DIAMETERS = Path("data/canalwidth/apical_diameter/estimates.csv")
DEFAULT_OUTPUT = Path("data/canalwidth/predictions/binding_files.csv")


def predict_binding_files(model_path: Path, estimates_csv: Path) -> pd.DataFrame:
    if not model_path.is_file():
        raise FileNotFoundError(f"Binding-file model not found: {model_path}")
    if not estimates_csv.is_file():
        raise FileNotFoundError(f"Apical estimates not found: {estimates_csv}")
    estimates = pd.read_csv(estimates_csv)
    required = {"image_name", *FEATURES}
    if missing := required - set(estimates.columns):
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")
    model = joblib.load(model_path)
    continuous = model.predict(estimates[FEATURES].to_numpy(dtype=float))
    estimates["continuous_file_size_k"] = continuous
    estimates["recommended_iso_size_k"] = nearest_iso_size(continuous)
    return estimates[
        [
            "image_name",
            "predicted_wl_mm",
            "apical_diameter_px",
            "continuous_file_size_k",
            "recommended_iso_size_k",
        ]
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--estimates", type=Path, default=DEFAULT_DIAMETERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    predictions = predict_binding_files(args.model, args.estimates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} binding-file recommendation(s)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
