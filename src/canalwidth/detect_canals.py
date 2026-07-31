"""Run canal detection over isolated working-length tooth images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from src.canalwidth.segmentation.canal_detection import detect_canal


DEFAULT_INPUT_DIR = Path("data/working_length/segmentation/isolated_teeth")
DEFAULT_OUTPUT_DIR = Path("data/canalwidth/segmentation")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    save_debug: bool = False,
    blackhat_kernel_size: int = 15,
    adaptive_block_size: int = 31,
    adaptive_c: float = -2.0,
) -> list[dict[str, object]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported tooth images found in: {input_dir}")

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        result = detect_canal(
            image,
            blackhat_kernel_size=blackhat_kernel_size,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
        )
        stem = image_path.stem
        write_image(output_dir / "masks" / f"{stem}.png", result["canal_mask"])
        write_image(output_dir / "overlays" / f"{stem}.png", result["overlay"])
        if save_debug:
            for stage in (
                "tooth_interior_mask",
                "clahe",
                "blackhat",
                "blackhat_normalized",
                "thresholded",
                "cleaned_candidates",
            ):
                write_image(
                    output_dir / "debug" / f"{stem}_{stage}.png",
                    result[stage],
                )
            report_path = output_dir / "debug" / f"{stem}_components.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as file:
                json.dump(result["component_reports"], file, indent=2)

        canal_pixels = cv2.countNonZero(result["canal_mask"])
        rows.append(
            {
                "image_name": image_path.name,
                "canal_detected": bool(result["detected"]),
                "canal_area_px": canal_pixels,
            }
        )
        status = f"{canal_pixels} px" if canal_pixels else "not detected"
        print(f"{image_path.name}: {status}")

    feature_path = output_dir / "canal_detection.csv"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("image_name", "canal_detected", "canal_area_px"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blackhat-kernel-size", type=int, default=15)
    parser.add_argument("--adaptive-block-size", type=int, default=31)
    parser.add_argument("--adaptive-c", type=float, default=-2.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows = process_directory(
        args.input_dir,
        args.output_dir,
        save_debug=args.debug,
        blackhat_kernel_size=args.blackhat_kernel_size,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
    )
    detected = sum(bool(row["canal_detected"]) for row in rows)
    print(f"Detected canals in {detected}/{len(rows)} image(s)")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
