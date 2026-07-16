"""Annotate ordered ground-truth tooth midlines as JSON polylines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.working_length.evaluation.midline_metrics import polyline_length


DEFAULT_INPUT_DIR = Path("data/working_length/segmentation/images")
DEFAULT_OUTPUT_DIR = Path("data/working_length/midline_evaluation/ground_truth")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_DISPLAY_WIDTH = 900
MAX_DISPLAY_HEIGHT = 700


class MidlineAnnotator:
    """Collect an ordered coronal-to-apical polyline on one tooth ROI."""

    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        zoom_scale: float = 3.0,
        load_existing: bool = False,
    ) -> None:
        if zoom_scale <= 0:
            raise ValueError("zoom_scale must be greater than zero")

        self.image_path = image_path
        self.output_path = output_path
        self.image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")

        self.height, self.width = self.image.shape
        self.display_scale = min(
            zoom_scale,
            MAX_DISPLAY_WIDTH / self.width,
            MAX_DISPLAY_HEIGHT / self.height,
        )
        self.points: list[tuple[int, int]] = []
        self.window_name = f"Annotate Midline - {image_path.name}"

        if load_existing and output_path.exists():
            self.load_annotation()
        if self.display_scale < zoom_scale:
            print(
                f"Display scale reduced from {zoom_scale:.2f} to "
                f"{self.display_scale:.2f} so the complete image fits on screen."
            )

    def load_annotation(self) -> None:
        with self.output_path.open("r", encoding="utf-8") as annotation_file:
            annotation = json.load(annotation_file)
        loaded_points = annotation.get("points", [])
        self.points = [
            (
                int(np.clip(point[0], 0, self.width - 1)),
                int(np.clip(point[1], 0, self.height - 1)),
            )
            for point in loaded_points
            if isinstance(point, list) and len(point) == 2
        ]

    def display_to_original(self, x: int, y: int) -> tuple[int, int]:
        return (
            int(np.clip(x / self.display_scale, 0, self.width - 1)),
            int(np.clip(y / self.display_scale, 0, self.height - 1)),
        )

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(self.display_to_original(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def create_preview(self) -> np.ndarray:
        preview = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        if len(self.points) >= 2:
            cv2.polylines(
                preview,
                [np.asarray(self.points, dtype=np.int32)],
                False,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        for index, point in enumerate(self.points):
            if index == 0:
                color = (255, 0, 0)  # coronal point: blue
            elif index == len(self.points) - 1:
                color = (0, 0, 255)  # current/apical end: red
            else:
                color = (0, 255, 255)
            cv2.circle(preview, point, 2, color, -1, cv2.LINE_AA)

        preview = cv2.resize(
            preview,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        instructions = [
            "Click from coronal reference to apex",
            "Left click: add point",
            "Right click / Z: undo",
            "C: clear | S / Enter: save",
            "K: skip | Esc: stop",
            f"Points: {len(self.points)}",
        ]
        for index, instruction in enumerate(instructions, start=1):
            cv2.putText(
                preview,
                instruction,
                (10, index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return preview

    def save_annotation(self) -> bool:
        if len(self.points) < 2:
            print(f"Cannot save {self.image_path.name}: select at least 2 points")
            return False
        annotation = {
            "image_name": self.image_path.name,
            "point_order": "coronal_to_apex",
            "points": [[int(x), int(y)] for x, y in self.points],
            "length_px": polyline_length(self.points),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as annotation_file:
            json.dump(annotation, annotation_file, indent=2)
        print(f"Saved midline: {self.output_path}")
        return True

    def run(self) -> str:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        while True:
            cv2.imshow(self.window_name, self.create_preview())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("z"), ord("Z")):
                if self.points:
                    self.points.pop()
            elif key in (ord("c"), ord("C")):
                self.points.clear()
            elif key in (ord("s"), ord("S"), 13):
                if self.save_annotation():
                    cv2.destroyWindow(self.window_name)
                    return "saved"
            elif key in (ord("k"), ord("K")):
                print(f"Skipped: {self.image_path.name}")
                cv2.destroyWindow(self.window_name)
                return "skipped"
            elif key == 27:
                cv2.destroyAllWindows()
                return "exit"


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def annotate_dataset(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    zoom_scale: float,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    image_paths = find_images(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {input_dir}")

    evaluation_root = output_dir.parent
    for directory_name in ("ground_truth", "predicted", "overlays", "results"):
        (evaluation_root / directory_name).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} image(s)")
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_dir / f"{image_path.stem}.json"
        if output_path.exists() and not overwrite:
            print(f"[{index}/{len(image_paths)}] Already annotated: {image_path.name}")
            continue
        print(f"[{index}/{len(image_paths)}] Annotating: {image_path.name}")
        annotator = MidlineAnnotator(
            image_path,
            output_path,
            zoom_scale,
            load_existing=overwrite,
        )
        if annotator.run() == "exit":
            print("Annotation stopped")
            break
    cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ordered ground-truth tooth midlines as JSON polylines"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zoom-scale", type=float, default=3.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Open and allow editing of annotations that already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    annotate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        zoom_scale=args.zoom_scale,
    )


if __name__ == "__main__":
    main()
