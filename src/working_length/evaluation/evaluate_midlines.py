"""Evaluate predicted midline lengths against manual polyline annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.working_length.evaluation.midline_metrics import (
    average_path_error,
    endpoint_errors,
    hausdorff_distance,
    polyline_length,
    symmetric_average_path_error,
)


DEFAULT_GROUND_TRUTH_DIR = Path(
    "data/working_length/midline_evaluation/ground_truth"
)
DEFAULT_PREDICTED_DIR = Path(
    "data/working_length/midline_evaluation/predicted/predicted_mask"
)
DEFAULT_RESULT_DIR = Path(
    "data/working_length/midline_evaluation/results/predicted_mask"
)
DEFAULT_IMAGE_DIR = Path("data/working_length/segmentation/images")
DEFAULT_OVERLAY_DIR = Path(
    "data/working_length/midline_evaluation/overlays/predicted_mask"
)

RESULT_COLUMNS = [
    "Image",
    "GroundTruthLengthPx",
    "PredictedLengthPx",
    "SignedLengthErrorPx",
    "AbsoluteLengthErrorPx",
    "PercentageLengthError",
    "AveragePathErrorPx",
    "SymmetricAveragePathErrorPx",
    "HausdorffDistancePx",
    "CoronalReferenceErrorPx",
    "ApexLocalizationErrorPx",
]


def load_midline(path: Path) -> tuple[str, np.ndarray]:
    """Load and validate one [x, y] midline annotation."""
    with path.open("r", encoding="utf-8") as annotation_file:
        annotation = json.load(annotation_file)

    image_name = annotation.get("image_name")
    if not isinstance(image_name, str) or not image_name:
        raise ValueError(f"Missing image_name in: {path}")

    points = np.asarray(annotation.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError(f"At least two [x, y] points are required in: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite point coordinate found in: {path}")
    return image_name, points


def calculate_length_errors(
    predicted_length: float,
    ground_truth_length: float,
) -> tuple[float, float, float]:
    """Return signed, absolute, and percentage length errors."""
    if ground_truth_length <= 0:
        raise ValueError("Ground-truth length must be greater than zero")
    signed_error = predicted_length - ground_truth_length
    absolute_error = abs(signed_error)
    percentage_error = absolute_error / ground_truth_length * 100.0
    return float(signed_error), float(absolute_error), float(percentage_error)


def create_recommended_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return the compact per-image table used in the research report."""
    compact = results[
        [
            "Image",
            "GroundTruthLengthPx",
            "PredictedLengthPx",
            "AbsoluteLengthErrorPx",
            "PercentageLengthError",
            "AveragePathErrorPx",
            "ApexLocalizationErrorPx",
        ]
    ].copy()
    return compact.rename(
        columns={
            "GroundTruthLengthPx": "GT Length px",
            "PredictedLengthPx": "Predicted Length px",
            "AbsoluteLengthErrorPx": "Absolute Error px",
            "PercentageLengthError": "Error %",
            "AveragePathErrorPx": "Mean Path Error px",
            "ApexLocalizationErrorPx": "Apex Error px",
        }
    )


def create_recommended_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Create the requested long-form summary table."""
    metrics = [
        ("Length absolute error", "AbsoluteLengthErrorPx"),
        ("Percentage error", "PercentageLengthError"),
        ("Average path error", "AveragePathErrorPx"),
        ("Apex error", "ApexLocalizationErrorPx"),
    ]
    rows = []
    for metric_name, column_name in metrics:
        values = results[column_name].to_numpy(dtype=float)
        rows.append(
            {
                "Metric": metric_name,
                "Mean": float(np.mean(values)),
                "Median": float(np.median(values)),
                "Standard deviation": float(np.std(values, ddof=0)),
                "Maximum": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def create_midline_overlay(
    image: np.ndarray,
    ground_truth_points: np.ndarray,
    predicted_points: np.ndarray,
) -> np.ndarray:
    """Draw manual path/apex in green and predicted path/apex in red."""
    if image.ndim == 2:
        visual = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        visual = image.copy()
    else:
        raise ValueError("Overlay image must be grayscale or BGR")

    ground_truth = np.rint(ground_truth_points).astype(np.int32)
    predicted = np.rint(predicted_points).astype(np.int32)
    thickness = max(1, int(round(min(visual.shape[:2]) / 120)))
    radius = max(3, 2 * thickness + 1)
    cv2.polylines(
        visual,
        [ground_truth],
        False,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    cv2.polylines(
        visual,
        [predicted],
        False,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.circle(visual, tuple(ground_truth[-1]), radius, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(visual, tuple(predicted[-1]), radius, (0, 0, 255), -1, cv2.LINE_AA)
    return visual


def generate_overlays(
    image_dir: Path,
    ground_truth_dir: Path,
    predicted_dir: Path,
    overlay_dir: Path,
) -> int:
    """Save one path-comparison overlay for every matched annotation pair."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Midline overlay image directory not found: {image_dir}")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for ground_truth_path in sorted(ground_truth_dir.glob("*.json")):
        predicted_path = predicted_dir / ground_truth_path.name
        if not predicted_path.exists():
            continue
        image_name, ground_truth_points = load_midline(ground_truth_path)
        predicted_image_name, predicted_points = load_midline(predicted_path)
        if image_name != predicted_image_name:
            continue
        image_path = image_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Missing overlay image: {image_path}")
            continue
        overlay = create_midline_overlay(image, ground_truth_points, predicted_points)
        output_path = overlay_dir / f"{Path(image_name).stem}.png"
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f"Could not save overlay: {output_path}")
        saved_count += 1
    return saved_count


def save_evaluation_outputs(
    results: pd.DataFrame,
    extended_summary: pd.DataFrame,
    result_dir: Path,
    image_dir: Path,
    ground_truth_dir: Path,
    predicted_dir: Path,
    overlay_dir: Path,
) -> dict[str, Path | int]:
    """Save compact/detailed tables and visual overlays for one experiment."""
    compact_results = create_recommended_results(results)
    recommended_summary = create_recommended_summary(results)
    result_dir.mkdir(parents=True, exist_ok=True)
    results_path = result_dir / "midline_length_results.csv"
    summary_path = result_dir / "midline_length_summary.csv"
    detailed_path = result_dir / "midline_detailed_results.csv"
    extended_summary_path = result_dir / "midline_extended_summary.csv"
    compact_results.to_csv(results_path, index=False)
    recommended_summary.to_csv(summary_path, index=False)
    results.to_csv(detailed_path, index=False)
    extended_summary.to_csv(extended_summary_path, index=False)
    overlay_count = generate_overlays(
        image_dir=image_dir,
        ground_truth_dir=ground_truth_dir,
        predicted_dir=predicted_dir,
        overlay_dir=overlay_dir,
    )
    return {
        "results": results_path,
        "summary": summary_path,
        "detailed": detailed_path,
        "extended_summary": extended_summary_path,
        "overlay_count": overlay_count,
    }


def evaluate_midline_lengths(
    ground_truth_dir: Path,
    predicted_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every ground-truth JSON that has a matching prediction JSON."""
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth midline directory not found: {ground_truth_dir}"
        )
    if not predicted_dir.is_dir():
        raise FileNotFoundError(
            f"Predicted midline directory not found: {predicted_dir}"
        )

    ground_truth_paths = sorted(ground_truth_dir.glob("*.json"))
    if not ground_truth_paths:
        raise ValueError(f"No ground-truth JSON files found in: {ground_truth_dir}")

    rows = []
    missing_count = 0
    for ground_truth_path in ground_truth_paths:
        predicted_path = predicted_dir / ground_truth_path.name
        if not predicted_path.exists():
            print(f"Missing prediction: {ground_truth_path.name}")
            missing_count += 1
            continue

        ground_truth_image, ground_truth_points = load_midline(ground_truth_path)
        predicted_image, predicted_points = load_midline(predicted_path)
        if ground_truth_image != predicted_image:
            raise ValueError(
                f"image_name mismatch for {ground_truth_path.name}: "
                f"{ground_truth_image!r} != {predicted_image!r}"
            )

        ground_truth_length = polyline_length(ground_truth_points)
        predicted_length = polyline_length(predicted_points)
        signed_error, absolute_error, percentage_error = calculate_length_errors(
            predicted_length,
            ground_truth_length,
        )
        path_error = average_path_error(predicted_points, ground_truth_points)
        symmetric_path_error = symmetric_average_path_error(
            predicted_points,
            ground_truth_points,
        )
        hausdorff_error = hausdorff_distance(
            predicted_points,
            ground_truth_points,
        )
        coronal_error, apex_error = endpoint_errors(
            predicted_points,
            ground_truth_points,
        )
        rows.append(
            [
                ground_truth_image,
                ground_truth_length,
                predicted_length,
                signed_error,
                absolute_error,
                percentage_error,
                path_error,
                symmetric_path_error,
                hausdorff_error,
                coronal_error,
                apex_error,
            ]
        )

    if not rows:
        raise ValueError("No matching ground-truth and predicted midlines found")

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    absolute_errors = results["AbsoluteLengthErrorPx"].to_numpy(dtype=float)
    signed_errors = results["SignedLengthErrorPx"].to_numpy(dtype=float)
    percentage_errors = results["PercentageLengthError"].to_numpy(dtype=float)
    path_errors = results["AveragePathErrorPx"].to_numpy(dtype=float)
    symmetric_path_errors = results[
        "SymmetricAveragePathErrorPx"
    ].to_numpy(dtype=float)
    hausdorff_errors = results["HausdorffDistancePx"].to_numpy(dtype=float)
    coronal_errors = results["CoronalReferenceErrorPx"].to_numpy(dtype=float)
    apex_errors = results["ApexLocalizationErrorPx"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "EvaluatedImages": len(results),
                "MissingPredictions": missing_count,
                "MAE_Px": float(np.mean(absolute_errors)),
                "RMSE_Px": float(np.sqrt(np.mean(np.square(signed_errors)))),
                "MeanPercentageError": float(np.mean(percentage_errors)),
                "MedianAbsoluteErrorPx": float(np.median(absolute_errors)),
                "MinimumAbsoluteErrorPx": float(np.min(absolute_errors)),
                "MaximumAbsoluteErrorPx": float(np.max(absolute_errors)),
                "MeanAveragePathErrorPx": float(np.mean(path_errors)),
                "MedianAveragePathErrorPx": float(np.median(path_errors)),
                "MeanSymmetricPathErrorPx": float(np.mean(symmetric_path_errors)),
                "MeanHausdorffDistancePx": float(np.mean(hausdorff_errors)),
                "MedianHausdorffDistancePx": float(np.median(hausdorff_errors)),
                "MaximumHausdorffDistancePx": float(np.max(hausdorff_errors)),
                "MeanApexErrorPx": float(np.mean(apex_errors)),
                "MedianApexErrorPx": float(np.median(apex_errors)),
                "MaximumApexErrorPx": float(np.max(apex_errors)),
                "MeanCoronalReferenceErrorPx": float(np.mean(coronal_errors)),
                "MedianCoronalReferenceErrorPx": float(np.median(coronal_errors)),
                "MaximumCoronalReferenceErrorPx": float(np.max(coronal_errors)),
            }
        ]
    )
    return results, summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted and manual midline polyline lengths"
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
    )
    parser.add_argument(
        "--predicted-dir",
        type=Path,
        default=DEFAULT_PREDICTED_DIR,
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    results, summary = evaluate_midline_lengths(
        ground_truth_dir=args.ground_truth_dir,
        predicted_dir=args.predicted_dir,
    )
    compact_results = create_recommended_results(results)
    recommended_summary = create_recommended_summary(results)

    print("\nPer-image midline results\n")
    print(compact_results.to_string(index=False))
    print("\nSummary\n")
    print(recommended_summary.to_string(index=False))

    saved = save_evaluation_outputs(
        results=results,
        extended_summary=summary,
        result_dir=args.result_dir,
        image_dir=args.image_dir,
        ground_truth_dir=args.ground_truth_dir,
        predicted_dir=args.predicted_dir,
        overlay_dir=args.overlay_dir,
    )
    print(f"\nSaved: {saved['results']}")
    print(f"Saved: {saved['summary']}")
    print(f"Saved: {saved['detailed']}")
    print(f"Saved: {saved['extended_summary']}")
    print(f"Saved {saved['overlay_count']} overlay(s) to: {args.overlay_dir}")


if __name__ == "__main__":
    main()
