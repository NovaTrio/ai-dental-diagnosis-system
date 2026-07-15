"""Create binary ground-truth tooth masks using polygon annotation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path("data/working_length/segmentation/images")
DEFAULT_OUTPUT_DIR = Path(
    "data/working_length/segmentation_evaluation/ground_truth_masks"
)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_DISPLAY_WIDTH = 900
MAX_DISPLAY_HEIGHT = 700


class ToothMaskAnnotator:
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        zoom_scale: float = 3.0,
    ) -> None:
        if zoom_scale <= 0:
            raise ValueError("zoom_scale must be greater than zero")

        self.image_path = image_path
        self.output_path = output_path
        self.zoom_scale = zoom_scale
        self.image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")

        self.height, self.width = self.image.shape
        self.display_scale = min(
            self.zoom_scale,
            MAX_DISPLAY_WIDTH / self.width,
            MAX_DISPLAY_HEIGHT / self.height,
        )
        self.points: list[tuple[int, int]] = []
        self.window_name = f"Annotate Tooth - {image_path.name}"

        if self.display_scale < self.zoom_scale:
            print(
                f"Display scale reduced from {self.zoom_scale:.2f} to "
                f"{self.display_scale:.2f} so the complete image fits on screen."
            )

    def display_to_original(self, x: int, y: int) -> tuple[int, int]:
        original_x = int(np.clip(x / self.display_scale, 0, self.width - 1))
        original_y = int(np.clip(y / self.display_scale, 0, self.height - 1))
        return original_x, original_y

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

    def create_mask(self) -> np.ndarray:
        if len(self.points) < 3:
            raise ValueError("At least three polygon points are required")

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        polygon = np.asarray(self.points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def create_preview(self) -> np.ndarray:
        preview = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        if self.points:
            polygon = np.asarray(self.points, dtype=np.int32)
            for point in self.points:
                cv2.circle(preview, point, 2, (0, 255, 255), -1)

            if len(self.points) >= 2:
                cv2.polylines(preview, [polygon], False, (0, 255, 0), 1)

            if len(self.points) >= 3:
                overlay = preview.copy()
                cv2.fillPoly(overlay, [polygon], (0, 255, 0))
                preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
                cv2.polylines(preview, [polygon], True, (0, 255, 0), 1)

        preview = cv2.resize(
            preview,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )

        instructions = [
            "Left click: add point",
            "Right click / Z: undo",
            "C: clear points",
            "S / Enter: save",
            "K: skip image",
            "Esc: stop",
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

    def save_mask(self) -> bool:
        if len(self.points) < 3:
            print(f"Cannot save {self.image_path.name}: select at least 3 points")
            return False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.output_path), self.create_mask()):
            raise OSError(f"Could not save mask: {self.output_path}")
        print(f"Saved mask: {self.output_path}")
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
                if self.save_mask():
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

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(image_paths)} image(s)")

    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_dir / f"{image_path.stem}.png"
        if output_path.exists() and not overwrite:
            print(f"[{index}/{len(image_paths)}] Already annotated: {image_path.name}")
            continue

        print(f"[{index}/{len(image_paths)}] Annotating: {image_path.name}")
        annotator = ToothMaskAnnotator(image_path, output_path, zoom_scale)
        if annotator.run() == "exit":
            print("Annotation stopped")
            break

    cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ground-truth tooth masks with polygon annotation"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zoom-scale", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
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
