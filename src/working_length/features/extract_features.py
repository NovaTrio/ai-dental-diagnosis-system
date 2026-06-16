import argparse
import csv
import math
import os

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from src.common.preprocessing.base_preprocess import get_dataset_path, load_image


def enhance_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def binarize_otsu(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def morphological_cleanup(binary, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    return cleaned


def keep_best_tooth_component(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return binary

    h, w = binary.shape
    image_area = float(h * w)

    best_label = None
    best_score = 0.0

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        if area < 0.002 * image_area:
            continue

        aspect_ratio = float(bh) / float(bw) if bw > 0 else 0.0

        if aspect_ratio < 1.2:
            continue

        center_x = x + bw * 0.5
        center_score = 1.0 - abs(center_x - w * 0.5) / (w * 0.5)
        center_score = np.clip(center_score, 0.0, 1.0)

        score = area * (0.7 + 0.3 * center_score)

        if score > best_score:
            best_score = score
            best_label = label

    mask = np.zeros_like(binary)

    if best_label is not None:
        mask[labels == best_label] = 255

    return mask


def fill_holes(mask):
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return mask

    for idx, hr in enumerate(hierarchy[0]):
        if hr[3] == -1:
            cv2.drawContours(mask, contours, idx, 255, thickness=cv2.FILLED)

    return mask


def compute_path_length_xy(path):
    if len(path) < 2:
        return 0.0

    length = 0.0

    for p1, p2 in zip(path, path[1:]):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length += math.sqrt(dx * dx + dy * dy)

    return float(length)


def remove_outlier_center_points(center_points, max_jump=20):
    if len(center_points) < 3:
        return center_points

    filtered = [center_points[0]]

    for point in center_points[1:]:
        prev_x, prev_y = filtered[-1]
        curr_x, curr_y = point

        if abs(curr_x - prev_x) <= max_jump:
            filtered.append(point)

    return filtered


def extract_rowwise_midline(mask, smooth_factor=80, min_row_pixels=5):
    """
    Extract curved midline from tooth mask.

    For each row:
        left boundary  = min x
        right boundary = max x
        center x       = (left + right) / 2

    Then smooth x as a function of y.
    """

    h, w = mask.shape
    center_points = []

    for y in range(h):
        xs = np.where(mask[y, :] > 0)[0]

        if len(xs) < min_row_pixels:
            continue

        x_left = xs.min()
        x_right = xs.max()
        x_center = (x_left + x_right) / 2.0

        center_points.append((x_center, y))

    if len(center_points) < 5:
        return [], 0.0

    center_points = remove_outlier_center_points(center_points, max_jump=25)

    if len(center_points) < 5:
        return [], 0.0

    center_points = np.array(center_points, dtype=np.float32)

    x = center_points[:, 0]
    y = center_points[:, 1]

    try:
        spline = UnivariateSpline(y, x, s=smooth_factor)

        y_smooth = np.linspace(y.min(), y.max(), len(y))
        x_smooth = spline(y_smooth)

        path = [
            (int(round(xv)), int(round(yv)))
            for xv, yv in zip(x_smooth, y_smooth)
        ]

    except Exception:
        path = [
            (int(round(xv)), int(round(yv)))
            for xv, yv in center_points
        ]

    length = compute_path_length_xy(path)

    return path, length


def overlay_midline(image, path):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for p1, p2 in zip(path, path[1:]):
        cv2.line(vis, p1, p2, (255, 0, 0), thickness=2)

    if path:
        cv2.circle(vis, path[0], 4, (0, 0, 255), -1)
        cv2.circle(vis, path[-1], 4, (0, 0, 255), -1)

    return vis


def overlay_midline_on_mask(mask, path):
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    dash_length = 6
    gap_length = 5

    for i in range(0, len(path) - 1, dash_length + gap_length):
        segment = path[i:i + dash_length]

        for p1, p2 in zip(segment, segment[1:]):
            cv2.line(vis, p1, p2, (0, 0, 0), thickness=1)

    return vis


def segment_tooth(img_gray, debug=False):
    enhanced = enhance_contrast(img_gray)

    binary = binarize_otsu(enhanced)

    cleaned = morphological_cleanup(binary)

    tooth = keep_best_tooth_component(cleaned)

    tooth = fill_holes(tooth)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tooth = cv2.morphologyEx(tooth, cv2.MORPH_CLOSE, kernel, iterations=2)

    path, length = extract_rowwise_midline(
        tooth,
        smooth_factor=80,
        min_row_pixels=5
    )

    if debug:
        print(f"  Binary pixels: {cv2.countNonZero(binary)}")
        print(f"  Cleaned pixels: {cv2.countNonZero(cleaned)}")
        print(f"  Tooth pixels: {cv2.countNonZero(tooth)}")
        print(f"  Midline points: {len(path)}")
        print(f"  Midline length: {length:.2f} pixels")

    return {
        "enhanced": enhanced,
        "binary": binary,
        "tooth_mask": tooth,
        "midline_path": path,
        "midline_length": length,
    }


def describe_image_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def process_image(path, output_dir, save_results=False, show=False, debug=False):
    img = load_image(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    result = segment_tooth(gray, debug=debug)

    overlay = None

    if save_results or show or debug:
        if result["midline_path"]:
            overlay = overlay_midline_on_mask(
                result["tooth_mask"],
                result["midline_path"]
            )
        else:
            overlay = cv2.cvtColor(result["tooth_mask"], cv2.COLOR_GRAY2BGR)

    if save_results:
        base = describe_image_path(path)
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir, f"{base}_enhanced.png"), result["enhanced"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_binary.png"), result["binary"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_mask.png"), result["tooth_mask"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_final_midline.png"), overlay)

    if show or debug:
        cv2.imshow("Enhanced", result["enhanced"])
        cv2.imshow("Binary", result["binary"])
        cv2.imshow("Tooth Mask", result["tooth_mask"])

        if overlay is not None:
            cv2.imshow("Final Midline Mask", overlay)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit(0)

        cv2.destroyAllWindows()

    return result["midline_length"]


def process_directory(input_dir, output_dir, csv_path, save_results=False, show=False, debug=False):
    os.makedirs(output_dir, exist_ok=True)

    rows = [("filename", "midline_length_pixels")]

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    for filename in image_files:
        path = os.path.join(input_dir, filename)

        print(f"Processing {filename}...")

        length = process_image(
            path,
            output_dir,
            save_results=save_results,
            show=show,
            debug=debug
        )

        rows.append((filename, f"{length:.2f}"))

    if csv_path is not None:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

        print(f"Saved midline length measurements to: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract tooth row-wise curved midline and compute pixel length"
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input directory containing tooth ROI images"
    )

    parser.add_argument(
        "--output-dir",
        default=get_dataset_path("working_length", "features"),
        help="Directory to save masks and overlays"
    )

    parser.add_argument(
        "--csv",
        default=os.path.join(
            get_dataset_path("working_length", "features"),
            "tooth_midline_lengths.csv"
        ),
        help="Output CSV file for pixel-length measurements"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save binary mask and overlay images"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show debug images during processing"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate diagnostics"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir or get_dataset_path(
        "working_length",
        "processed",
        "teeth"
    )

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    process_directory(
        input_dir=input_dir,
        output_dir=args.output_dir,
        csv_path=args.csv,
        save_results=args.save,
        show=args.show,
        debug=args.debug
    )


if __name__ == "__main__":
    main()