"""Extract canal centrelines from isolated teeth and canal masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from src.canalwidth.features.canal_midline import extract_canal_midline


DEFAULT_IMAGE_DIR = Path("data/working_length/segmentation/isolated_teeth")
DEFAULT_CANAL_MASK_DIR = Path("data/canalwidth/segmentation/masks")
DEFAULT_OUTPUT_DIR = Path("data/canalwidth/midlines")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def extract_directory(
    image_dir: Path,
    canal_mask_dir: Path,
    output_dir: Path,
    *,
    smoothing: float = 3.0,
    save_debug: bool = False,
) -> list[dict[str, object]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not canal_mask_dir.is_dir():
        raise FileNotFoundError(f"Canal-mask directory not found: {canal_mask_dir}")
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    rows: list[dict[str, object]] = []
    coordinate_rows: list[dict[str, object]] = []
    for image_path in images:
        mask_path = canal_mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            print(f"Missing canal mask: {image_path.name}")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        canal_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or canal_mask is None:
            print(f"Skipping unreadable pair: {image_path.name}")
            continue
        if canal_mask.shape != image.shape:
            raise ValueError(f"Shape mismatch for {image_path.name}")

        result = extract_canal_midline(image, canal_mask, smoothing=smoothing)
        stem = image_path.stem
        write_image(output_dir / "masks" / f"{stem}.png", result["midline_mask"])
        write_image(output_dir / "overlays" / f"{stem}.png", result["overlay"])
        if save_debug:
            for stage in (
                "bright_material_mask",
                "combined_canal_mask",
                "skeleton",
            ):
                write_image(
                    output_dir / "debug" / f"{stem}_{stage}.png",
                    result[stage],
                )
        rows.append(
            {
                "image_name": image_path.name,
                "midline_detected": result["detected"],
                "midline_length_px": f"{result['midline_length_px']:.4f}",
                "point_count": len(result["midline_path"]),
            }
        )
        for index, (x, y) in enumerate(result["midline_path"]):
            coordinate_rows.append(
                {
                    "image_name": image_path.name,
                    "point_index": index,
                    "x": f"{x:.4f}",
                    "y": f"{y:.4f}",
                }
            )
        print(
            f"{image_path.name}: {len(result['midline_path'])} points, "
            f"{result['midline_length_px']:.2f} px"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "canal_midlines.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "image_name",
                "midline_detected",
                "midline_length_px",
                "point_count",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "canal_midline_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("image_name", "point_index", "x", "y"),
        )
        writer.writeheader()
        writer.writerows(coordinate_rows)
    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--canal-mask-dir", type=Path, default=DEFAULT_CANAL_MASK_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoothing", type=float, default=3.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows = extract_directory(
        args.image_dir,
        args.canal_mask_dir,
        args.output_dir,
        smoothing=args.smoothing,
        save_debug=args.debug,
    )
    detected = sum(bool(row["midline_detected"]) for row in rows)
    print(f"Extracted {detected}/{len(rows)} canal midline(s)")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
