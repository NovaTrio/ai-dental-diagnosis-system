"""Manual click-based calibration of the 10 mm image marker."""

import os
import cv2
import numpy as np
import pandas as pd


# Configuration
INPUT_DIR = "../../../data/abscess/raw/images"
OUTPUT_VIS_DIR = "../../../data/abscess/processed/manual_calibration_visualizations"
OUTPUT_CSV = "../../../data/abscess/processed/scale_calibration.csv"

KNOWN_SCALE_MM = 10.0
MAX_DISPLAY_DIM = 1000
WINDOW_NAME = "Manual Scale Calibration"

CSV_COLUMNS = [
    "image_name", "scale_start_x", "scale_start_y", "scale_end_x",
    "scale_end_y", "scale_pixels", "known_scale_mm", "mm_per_pixel",
]


class ClickState:
    """Holds the current click points and redraw callback for one image."""

    def __init__(self, base_display_image):
        self.base_display_image = base_display_image
        self.points = []
        self.canvas = base_display_image.copy()

    def reset(self):
        self.points = []
        self.canvas = self.base_display_image.copy()

    def add_point(self, x, y):
        if len(self.points) >= 2:
            return
        self.points.append((x, y))
        self._redraw()

    def _redraw(self):
        self.canvas = self.base_display_image.copy()
        for point in self.points:
            cv2.circle(self.canvas, point, 5, (0, 255, 255), -1)
        if len(self.points) == 2:
            cv2.line(self.canvas, self.points[0], self.points[1], (0, 0, 255), 2, cv2.LINE_AA)


def _mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.add_point(x, y)


def _draw_instructions(canvas, message):
    display = canvas.copy()
    overlay_lines = [
        message,
        "Click start and end of the 10 mm scale marker.",
        "[c] confirm  [r] retry  [s] skip  [q] quit",
    ]
    y = 22
    for line in overlay_lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    return display


def calibrate_image_manual(image, window_name):
    """Run the interactive click loop for a single image.

    Returns a dict with original-image-coordinate results, or a string
    status ("skip" / "quit") if the user chose not to calibrate.
    """
    image_height, image_width = image.shape[:2]
    scale = 1.0
    if max(image_height, image_width) > MAX_DISPLAY_DIM:
        scale = MAX_DISPLAY_DIM / float(max(image_height, image_width))
    display_size = (int(round(image_width * scale)), int(round(image_height * scale)))
    base_display_image = cv2.resize(image, display_size) if scale != 1.0 else image.copy()

    state = ClickState(base_display_image)
    cv2.setMouseCallback(window_name, _mouse_callback, state)

    message = "Ready."
    while True:
        frame = _draw_instructions(state.canvas, message)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if len(state.points) < 2:
            message = f"Point {len(state.points)}/2 selected."
            if key == ord('q'):
                return "quit"
            if key == ord('s'):
                return "skip"
            if key == ord('r'):
                state.reset()
                message = "Cleared. Click the start point."
            continue

        message = "Line drawn. Press c to confirm, r to retry, s to skip, q to quit."
        if key == ord('c'):
            (start_disp_x, start_disp_y), (end_disp_x, end_disp_y) = state.points
            start_x, start_y = start_disp_x / scale, start_disp_y / scale
            end_x, end_y = end_disp_x / scale, end_disp_y / scale
            scale_pixels = float(np.hypot(end_x - start_x, end_y - start_y))
            if scale_pixels <= 0:
                message = "Invalid selection (zero length). Click two distinct points."
                state.reset()
                continue
            mm_per_pixel = KNOWN_SCALE_MM / scale_pixels
            return {
                "scale_start_x": round(start_x, 2),
                "scale_start_y": round(start_y, 2),
                "scale_end_x": round(end_x, 2),
                "scale_end_y": round(end_y, 2),
                "scale_pixels": round(scale_pixels, 2),
                "known_scale_mm": KNOWN_SCALE_MM,
                "mm_per_pixel": round(mm_per_pixel, 4),
            }
        if key == ord('r'):
            state.reset()
            message = "Cleared. Click the start point."
        elif key == ord('s'):
            return "skip"
        elif key == ord('q'):
            return "quit"


def build_visualization(image, result):
    """Draw the confirmed calibration line on the full-resolution image."""
    visualization = image.copy()
    start = (int(round(result["scale_start_x"])), int(round(result["scale_start_y"])))
    end = (int(round(result["scale_end_x"])), int(round(result["scale_end_y"])))
    cv2.line(visualization, start, end, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(visualization, start, 6, (0, 255, 255), -1)
    cv2.circle(visualization, end, 6, (255, 0, 255), -1)
    labels = [
        f"Scale length: {result['scale_pixels']:.1f} px",
        f"mm/px: {result['mm_per_pixel']:.4f}",
    ]
    text_x = max(10, min(start[0], end[0]) - 10)
    text_y = max(25, min(start[1], end[1]) - 15)
    for label in labels:
        cv2.putText(visualization, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        text_y += 24
    return visualization


def load_existing_results(csv_path):
    """Load prior calibration results into an {image_name: row} dict."""
    if not os.path.exists(csv_path):
        return {}
    existing = pd.read_csv(csv_path)
    return {row["image_name"]: row.to_dict() for _, row in existing.iterrows()}


def save_results(csv_path, results_by_name):
    ordered_rows = [results_by_name[name] for name in results_by_name]
    pd.DataFrame(ordered_rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)


def main():
    print("=" * 60)
    print("  Manual Scale Calibration")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_abs_dir = os.path.normpath(os.path.join(script_dir, INPUT_DIR))
    if not os.path.exists(input_abs_dir):
        print(f"[ERROR] Input directory not found: {input_abs_dir}")
        return

    valid_extensions = (".jpg", ".jpeg", ".png")
    files = sorted(name for name in os.listdir(input_abs_dir)
                    if name.lower().endswith(valid_extensions))
    if not files:
        print(f"[ERROR] No valid images found in {input_abs_dir}")
        return

    visualization_dir = os.path.normpath(os.path.join(script_dir, OUTPUT_VIS_DIR))
    csv_path = os.path.normpath(os.path.join(script_dir, OUTPUT_CSV))
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    results_by_name = load_existing_results(csv_path)
    print(f"Found {len(files)} images. {len(results_by_name)} already calibrated.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    quit_requested = False
    processed_count = 0
    for filename in files:
        if quit_requested:
            break

        print(f"Processing: {filename}")
        image = cv2.imread(os.path.join(input_abs_dir, filename))
        if image is None:
            print(f"  -> Could not load {filename}, skipping.")
            continue

        outcome = calibrate_image_manual(image, WINDOW_NAME)

        if outcome == "quit":
            print("  -> Quit requested.")
            quit_requested = True
            break
        if outcome == "skip":
            print("  -> Skipped.")
            continue

        result_row = {"image_name": filename, **outcome}
        results_by_name[filename] = result_row
        processed_count += 1

        visualization = build_visualization(image, result_row)
        cv2.imwrite(os.path.join(visualization_dir, f"manual_calib_vis_{filename}"),
                    visualization)
        save_results(csv_path, results_by_name)
        print(f"  -> mm/px: {result_row['mm_per_pixel']}  "
              f"(scale_pixels: {result_row['scale_pixels']})")

    cv2.destroyAllWindows()

    if results_by_name:
        save_results(csv_path, results_by_name)
        print(f"\n[SUCCESS] Calibrated {processed_count} image(s) this run.")
        print(f"Results saved to: {csv_path}")
        print(f"Visualizations saved to: {visualization_dir}")
    else:
        print("\n[INFO] No results to save.")


if __name__ == "__main__":
    main()
