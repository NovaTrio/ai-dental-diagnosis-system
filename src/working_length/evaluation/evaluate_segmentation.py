"""Evaluate predicted tooth masks against manually annotated ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_PREDICTED_DIR = Path("data/working_length/segmentation/masks")
DEFAULT_GROUND_TRUTH_DIR = Path(
    "data/working_length/segmentation_evaluation/ground_truth_masks"
)
DEFAULT_RESULT_DIR = Path("data/working_length/segmentation_evaluation/results")
DEFAULT_OVERLAY_DIR = Path("data/working_length/segmentation_evaluation/overlays")
RESULT_COLUMNS = ["Image", "Dice", "IoU", "Precision", "Recall", "Accuracy"]
EPSILON = 1e-8


def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Calculate binary segmentation metrics for one prediction/GT pair."""
    predicted_positive = prediction > 0
    actual_positive = ground_truth > 0

    true_positive = np.logical_and(predicted_positive, actual_positive).sum()
    false_positive = np.logical_and(predicted_positive, ~actual_positive).sum()
    false_negative = np.logical_and(~predicted_positive, actual_positive).sum()
    true_negative = np.logical_and(~predicted_positive, ~actual_positive).sum()

    dice = (2 * true_positive) / (
        2 * true_positive + false_positive + false_negative + EPSILON
    )
    iou = true_positive / (
        true_positive + false_positive + false_negative + EPSILON
    )
    precision = true_positive / (true_positive + false_positive + EPSILON)
    recall = true_positive / (true_positive + false_negative + EPSILON)
    pixel_accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative
    )

    return (
        float(dice),
        float(iou),
        float(precision),
        float(recall),
        float(pixel_accuracy),
    )


def read_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def create_agreement_overlay(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Color agreement and errors: green=TP, red=FP, blue=FN."""
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground-truth masks must have equal shapes")

    predicted_positive = prediction > 0
    actual_positive = ground_truth > 0
    true_positive = np.logical_and(predicted_positive, actual_positive)
    false_positive = np.logical_and(predicted_positive, ~actual_positive)
    false_negative = np.logical_and(~predicted_positive, actual_positive)

    overlay = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    overlay[true_positive] = (0, 255, 0)  # Green in BGR.
    overlay[false_positive] = (0, 0, 255)  # Red in BGR.
    overlay[false_negative] = (255, 0, 0)  # Blue in BGR.
    return overlay


def evaluate_segmentation(
    predicted_dir: Path,
    ground_truth_dir: Path,
    overlay_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate all PNG ground-truth masks that have matching predictions."""
    if not predicted_dir.is_dir():
        raise FileNotFoundError(f"Predicted-mask directory not found: {predicted_dir}")
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {ground_truth_dir}"
        )

    ground_truth_paths = sorted(ground_truth_dir.glob("*.png"))
    if not ground_truth_paths:
        raise ValueError(f"No PNG ground-truth masks found in: {ground_truth_dir}")

    results = []
    resized_count = 0
    missing_count = 0

    for ground_truth_path in ground_truth_paths:
        prediction_path = predicted_dir / ground_truth_path.name
        if not prediction_path.exists():
            print(f"Missing prediction: {ground_truth_path.name}")
            missing_count += 1
            continue

        ground_truth = read_grayscale_image(ground_truth_path)
        prediction = read_grayscale_image(prediction_path)

        if ground_truth.shape != prediction.shape:
            print(
                f"Resizing prediction: {ground_truth_path.name} "
                f"{prediction.shape} -> {ground_truth.shape}"
            )
            prediction = cv2.resize(
                prediction,
                (ground_truth.shape[1], ground_truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            resized_count += 1

        dice, iou, precision, recall, accuracy = calculate_metrics(
            prediction,
            ground_truth,
        )
        if overlay_dir is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay = create_agreement_overlay(prediction, ground_truth)
            overlay_path = overlay_dir / ground_truth_path.name
            if not cv2.imwrite(str(overlay_path), overlay):
                raise ValueError(f"Could not save overlay: {overlay_path}")
        results.append(
            [
                ground_truth_path.name,
                dice,
                iou,
                precision,
                recall,
                accuracy,
            ]
        )

    if not results:
        raise ValueError("No matching prediction and ground-truth mask pairs found.")

    print(
        f"Evaluated {len(results)} pair(s); "
        f"missing={missing_count}, resized={resized_count}."
    )
    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted tooth masks against ground-truth masks."
    )
    parser.add_argument(
        "--predicted-dir",
        type=Path,
        default=DEFAULT_PREDICTED_DIR,
        help=f"Predicted mask directory (default: {DEFAULT_PREDICTED_DIR})",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help=f"Ground-truth mask directory (default: {DEFAULT_GROUND_TRUTH_DIR})",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Evaluation result directory (default: {DEFAULT_RESULT_DIR})",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=DEFAULT_OVERLAY_DIR,
        help=f"Agreement overlay directory (default: {DEFAULT_OVERLAY_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    results = evaluate_segmentation(
        predicted_dir=args.predicted_dir,
        ground_truth_dir=args.ground_truth_dir,
        overlay_dir=args.overlay_dir,
    )

    print("\nPer-image results\n")
    print(results.to_string(index=False))

    print("\nMean results\n")
    print(results.mean(numeric_only=True).to_string())

    args.result_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.result_dir / "segmentation_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Saved overlays to: {args.overlay_dir}")


if __name__ == "__main__":
    main()
