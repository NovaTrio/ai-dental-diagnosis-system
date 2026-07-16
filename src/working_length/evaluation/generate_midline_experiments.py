"""Extract comparable predicted midlines from manual and predicted masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.working_length.evaluation.midline_metrics import (
    orient_path_coronal_to_apex,
    polyline_length,
)
from src.working_length.features.extract_features import (
    crop_tooth_region,
    extract_rowwise_midline,
)


DEFAULT_GT_MASK_DIR = Path(
    "data/working_length/segmentation_evaluation/ground_truth_masks"
)
DEFAULT_PREDICTED_MASK_DIR = Path("data/working_length/segmentation/masks")
DEFAULT_OUTPUT_ROOT = Path("data/working_length/midline_evaluation/predicted")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract_midline_from_mask(mask_path: Path) -> dict:
    """Apply the production rowwise midline algorithm to one binary mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, cropped_mask, (offset_x, offset_y) = crop_tooth_region(
        binary,
        binary,
        padding_ratio=0.03,
    )
    cropped_path, _ = extract_rowwise_midline(
        cropped_mask,
        smooth_factor=60,
        min_row_pixels=5,
    )
    cropped_path = orient_path_coronal_to_apex(cropped_path, cropped_mask)
    full_path = [
        (int(x + offset_x), int(y + offset_y))
        for x, y in cropped_path
    ]
    if len(full_path) < 2:
        raise ValueError(f"Could not extract a valid midline from: {mask_path}")
    return {
        "image_name": f"{mask_path.stem}.png",
        "point_order": "coronal_to_apex",
        "points": [[x, y] for x, y in full_path],
        "length_px": polyline_length(full_path),
        "source_mask": str(mask_path.as_posix()),
    }


def generate_from_mask_directory(mask_dir: Path, output_dir: Path) -> int:
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    mask_paths = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not mask_paths:
        raise ValueError(f"No mask images found in: {mask_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for mask_path in mask_paths:
        try:
            prediction = extract_midline_from_mask(mask_path)
        except ValueError as error:
            print(f"Skipped {mask_path.name}: {error}")
            continue
        output_path = output_dir / f"{mask_path.stem}.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(prediction, output_file, indent=2)
        generated += 1
    return generated


def generate_both_experiments(
    ground_truth_mask_dir: Path,
    predicted_mask_dir: Path,
    output_root: Path,
) -> tuple[int, int]:
    ground_truth_count = generate_from_mask_directory(
        ground_truth_mask_dir,
        output_root / "ground_truth_mask",
    )
    predicted_count = generate_from_mask_directory(
        predicted_mask_dir,
        output_root / "predicted_mask",
    )
    return ground_truth_count, predicted_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract midlines separately from manual and predicted masks"
    )
    parser.add_argument("--ground-truth-mask-dir", type=Path, default=DEFAULT_GT_MASK_DIR)
    parser.add_argument(
        "--predicted-mask-dir",
        type=Path,
        default=DEFAULT_PREDICTED_MASK_DIR,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ground_truth_count, predicted_count = generate_both_experiments(
        args.ground_truth_mask_dir,
        args.predicted_mask_dir,
        args.output_root,
    )
    print(f"Generated ground-truth-mask midlines: {ground_truth_count}")
    print(f"Generated predicted-mask midlines: {predicted_count}")


if __name__ == "__main__":
    main()
