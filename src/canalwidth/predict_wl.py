"""Predict working length from valid canal-centreline measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL = Path(
    "src/canalwidth/models/working_length/linear_regression.joblib"
)
DEFAULT_MIDLINES = Path("data/canalwidth/midlines/canal_midlines.csv")
DEFAULT_OUTPUT = Path("data/canalwidth/predictions/working_lengths.csv")


def predict_working_lengths(
    model_path: Path,
    midline_csv: Path,
) -> pd.DataFrame:
    if not model_path.is_file():
        raise FileNotFoundError(f"Working-length model not found: {model_path}")
    if not midline_csv.is_file():
        raise FileNotFoundError(f"Canal midline CSV not found: {midline_csv}")
    measurements = pd.read_csv(midline_csv)
    required = {"image_name", "midline_detected", "midline_length_px"}
    if missing := required - set(measurements.columns):
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")
    valid = measurements["midline_detected"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    measurements = measurements.loc[valid].copy()
    measurements["midline_length_px"] = pd.to_numeric(
        measurements["midline_length_px"], errors="coerce"
    )
    measurements = measurements.dropna(subset=["midline_length_px"])
    measurements = measurements.loc[measurements["midline_length_px"] > 0]
    if measurements.empty:
        raise ValueError("No valid canal midlines available for prediction")
    model = joblib.load(model_path)
    measurements["predicted_wl_mm"] = model.predict(
        measurements[["midline_length_px"]].to_numpy(dtype=float)
    )
    return measurements[
        ["image_name", "midline_length_px", "predicted_wl_mm"]
    ].reset_index(drop=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--midlines", type=Path, default=DEFAULT_MIDLINES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    predictions = predict_working_lengths(args.model, args.midlines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Predicted working length for {len(predictions)} image(s)")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
