"""Measure median apical canal width at 95–99% of the centreline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

from src.canalwidth.features.canal_midline import extract_canal_midline
from src.canalwidth.features.canal_width import (
    create_width_overlay,
    measure_apical_canal_widths,
)


DEFAULT_IMAGE_DIR = Path("data/working_length/segmentation/isolated_teeth")
DEFAULT_CANAL_MASK_DIR = Path("data/canalwidth/segmentation/masks")
DEFAULT_PREDICTIONS = Path("data/canalwidth/predictions/working_lengths.csv")
DEFAULT_OUTPUT_DIR = Path("data/canalwidth/widths")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _predictions_by_name(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    table = pd.read_csv(path)
    required = {"image_name", "predicted_wl_mm"}
    if not required.issubset(table.columns):
        raise ValueError(f"Invalid prediction CSV: {path}")
    return dict(
        zip(
            table["image_name"].astype(str),
            pd.to_numeric(table["predicted_wl_mm"], errors="coerce"),
        )
    )


def _write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def measure_directory(
    image_dir: Path,
    canal_mask_dir: Path,
    output_dir: Path,
    predictions_csv: Path | None = DEFAULT_PREDICTIONS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not canal_mask_dir.is_dir():
        raise FileNotFoundError(f"Canal-mask directory not found: {canal_mask_dir}")
    predictions = _predictions_by_name(predictions_csv)
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        mask_path = canal_mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            print(f"Missing canal mask: {image_path.name}")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        dark_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or dark_mask is None:
            continue
        midline = extract_canal_midline(image, dark_mask)
        if not midline["detected"]:
            result = {
                "measurements": [],
                "valid_measurement_count": 0,
                "median_width_px": None,
                "valid": False,
                "reason": "invalid_canal_midline",
            }
        else:
            result = measure_apical_canal_widths(
                midline["combined_canal_mask"],
                midline["midline_path"],
            )
            overlay = create_width_overlay(
                image,
                midline["midline_path"],
                result["measurements"],
            )
            _write_image(output_dir / "overlays" / f"{image_path.stem}.png", overlay)

        summary_rows.append(
            {
                "image_name": image_path.name,
                "predicted_wl_mm": predictions.get(image_path.name),
                "median_canal_width_px": result["median_width_px"],
                "valid_width_measurements": result["valid_measurement_count"],
                "width_valid": result["valid"],
                "reason": result["reason"],
            }
        )
        for measurement in result["measurements"]:
            detail_rows.append(
                {
                    "image_name": image_path.name,
                    "fraction": measurement["fraction"],
                    "width_px": measurement.get("width_px"),
                    "valid": measurement["valid"],
                    "reason": measurement["reason"],
                    "left_wall_x": (
                        measurement["left_wall"][0] if measurement["valid"] else None
                    ),
                    "left_wall_y": (
                        measurement["left_wall"][1] if measurement["valid"] else None
                    ),
                    "centre_x": (
                        measurement["centre"][0] if measurement["valid"] else None
                    ),
                    "centre_y": (
                        measurement["centre"][1] if measurement["valid"] else None
                    ),
                    "right_wall_x": (
                        measurement["right_wall"][0] if measurement["valid"] else None
                    ),
                    "right_wall_y": (
                        measurement["right_wall"][1] if measurement["valid"] else None
                    ),
                }
            )
        print(
            f"{image_path.name}: median={result['median_width_px']} "
            f"({result['valid_measurement_count']}/5 valid)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "canal_widths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "image_name",
                "predicted_wl_mm",
                "median_canal_width_px",
                "valid_width_measurements",
                "width_valid",
                "reason",
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    detail_fields = (
        "image_name",
        "fraction",
        "width_px",
        "valid",
        "reason",
        "left_wall_x",
        "left_wall_y",
        "centre_x",
        "centre_y",
        "right_wall_x",
        "right_wall_y",
    )
    with (output_dir / "canal_width_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    return summary_rows, detail_rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--canal-mask-dir", type=Path, default=DEFAULT_CANAL_MASK_DIR
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    summary, _ = measure_directory(
        args.image_dir,
        args.canal_mask_dir,
        args.output_dir,
        args.predictions,
    )
    valid = sum(bool(row["width_valid"]) for row in summary)
    print(f"Valid median widths: {valid}/{len(summary)}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
