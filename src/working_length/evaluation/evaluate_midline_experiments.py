"""Run and compare the two mask-to-midline evaluation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.working_length.evaluation.evaluate_midlines import (
    evaluate_midline_lengths,
    save_evaluation_outputs,
)
from src.working_length.evaluation.generate_midline_experiments import (
    DEFAULT_GT_MASK_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREDICTED_MASK_DIR,
    generate_both_experiments,
)


DEFAULT_MANUAL_MIDLINE_DIR = Path(
    "data/working_length/midline_evaluation/ground_truth"
)
DEFAULT_IMAGE_DIR = Path("data/working_length/segmentation/images")
DEFAULT_RESULT_ROOT = Path("data/working_length/midline_evaluation/results")
DEFAULT_OVERLAY_ROOT = Path("data/working_length/midline_evaluation/overlays")


def comparison_row(experiment: str, results: pd.DataFrame) -> dict:
    absolute_errors = results["AbsoluteLengthErrorPx"].to_numpy(dtype=float)
    return {
        "Experiment": experiment,
        "EvaluatedImages": len(results),
        "LengthMAE_Px": float(np.mean(absolute_errors)),
        "LengthRMSE_Px": float(np.sqrt(np.mean(np.square(absolute_errors)))),
        "MeanPercentageError": float(np.mean(results["PercentageLengthError"])),
        "MeanAveragePathErrorPx": float(np.mean(results["AveragePathErrorPx"])),
        "MeanHausdorffDistancePx": float(np.mean(results["HausdorffDistancePx"])),
        "MeanApexErrorPx": float(np.mean(results["ApexLocalizationErrorPx"])),
        "MeanCoronalReferenceErrorPx": float(
            np.mean(results["CoronalReferenceErrorPx"])
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare midlines extracted from manual versus predicted masks"
    )
    parser.add_argument("--manual-midline-dir", type=Path, default=DEFAULT_MANUAL_MIDLINE_DIR)
    parser.add_argument("--ground-truth-mask-dir", type=Path, default=DEFAULT_GT_MASK_DIR)
    parser.add_argument("--predicted-mask-dir", type=Path, default=DEFAULT_PREDICTED_MASK_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--overlay-root", type=Path, default=DEFAULT_OVERLAY_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    generated_counts = generate_both_experiments(
        args.ground_truth_mask_dir,
        args.predicted_mask_dir,
        args.prediction_root,
    )
    experiment_names = ("ground_truth_mask", "predicted_mask")
    comparison_rows = []
    for experiment_name, generated_count in zip(experiment_names, generated_counts):
        predicted_dir = args.prediction_root / experiment_name
        results, extended_summary = evaluate_midline_lengths(
            args.manual_midline_dir,
            predicted_dir,
        )
        save_evaluation_outputs(
            results=results,
            extended_summary=extended_summary,
            result_dir=args.result_root / experiment_name,
            image_dir=args.image_dir,
            ground_truth_dir=args.manual_midline_dir,
            predicted_dir=predicted_dir,
            overlay_dir=args.overlay_root / experiment_name,
        )
        comparison_rows.append(comparison_row(experiment_name, results))
        print(
            f"{experiment_name}: generated={generated_count}, "
            f"evaluated={len(results)}"
        )

    comparison = pd.DataFrame(comparison_rows)
    args.result_root.mkdir(parents=True, exist_ok=True)
    comparison_path = args.result_root / "experiment_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\nExperiment comparison\n")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {comparison_path}")


if __name__ == "__main__":
    main()
