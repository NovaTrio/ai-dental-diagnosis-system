import argparse
import csv
import math
import os

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from src.common.preprocessing.base_preprocess import get_dataset_path, load_image


# ============================================================
# 1. BASIC IMAGE ENHANCEMENT
# ============================================================

def enhance_roi(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    filtered = cv2.bilateralFilter(
        enhanced,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    return filtered


# ============================================================
# 2. ADAPTIVE THRESHOLDING
# ============================================================

def remove_metal_artifacts(img_gray):
    """
    Detects very bright metallic objects such as files/clamps
    and removes them using inpainting.
    """
    _, metal_mask = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        metal_mask,
        connectivity=8
    )

    cleaned_metal_mask = np.zeros_like(metal_mask)
    h, w = img_gray.shape

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1)

        if area > 20 and aspect_ratio > 3:
            cleaned_metal_mask[labels == label] = 255

        if area > 0.03 * h * w:
            cleaned_metal_mask[labels == label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_metal_mask = cv2.dilate(cleaned_metal_mask, kernel, iterations=2)

    cleaned_img = cv2.inpaint(
        img_gray,
        cleaned_metal_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return cleaned_img, cleaned_metal_mask


def adaptive_tooth_threshold(img_gray):
    binary = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -2
    )

    return binary


def remove_metal_fixtures(enhanced_img, binary_mask, threshold=235):
    """
    Remove highly radiopaque metal fixtures from the adaptive tooth mask.

    The distance transform should follow biological tooth structure, not a
    solid endodontic file or clamp. This creates a strict high-intensity metal
    mask and subtracts it from the adaptive binary mask before core extraction.
    """
    _, metal_fixture_mask = cv2.threshold(
        enhanced_img,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    metal_fixture_mask = cv2.dilate(
        metal_fixture_mask,
        kernel,
        iterations=1
    )

    clean_binary = cv2.bitwise_and(
        binary_mask,
        cv2.bitwise_not(metal_fixture_mask)
    )

    return clean_binary, metal_fixture_mask


# ============================================================
# 3. CLEAR SIDE MARGINS
# ============================================================

def clear_side_margins(binary, margin_ratio=0.08):
    h, w = binary.shape
    margin = int(w * margin_ratio)

    cleaned = binary.copy()
    cleaned[:, :margin] = 0
    cleaned[:, w - margin:] = 0

    return cleaned


# ============================================================
# 4. MORPHOLOGICAL CLEANUP
# ============================================================

def morphology_cleanup(binary):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return closed


def suppress_top_broad_artifacts(binary):
    """
    Suppress broad coronal clamp/crown artifacts before distance transform.

    Working-length ROIs usually contain the root as a narrower vertical structure
    in the lower image. Rubber dam clamps often appear as wide, bright objects
    across the upper image and can dominate the distance transform. This estimates
    the root corridor from lower foreground rows, then trims unusually wide upper
    rows back to that corridor.
    """
    h, w = binary.shape
    lower_start = int(0.35 * h)

    lower_widths = []
    lower_centers = []

    for y in range(lower_start, h):
        xs = np.where(binary[y, :] > 0)[0]

        if len(xs) < 5:
            continue

        lower_widths.append(xs.max() - xs.min() + 1)
        lower_centers.append((xs.min() + xs.max()) / 2.0)

    if len(lower_widths) < 5:
        return binary

    root_center = float(np.median(lower_centers))
    root_width = float(np.percentile(lower_widths, 70))

    if root_width <= 0:
        return binary

    corridor_half_width = max(0.80 * root_width, 0.12 * w, 8)
    corridor_left = max(0, int(round(root_center - corridor_half_width)))
    corridor_right = min(w, int(round(root_center + corridor_half_width + 1)))

    upper_limit = int(0.55 * h)
    wide_row_threshold = max(1.55 * root_width, 0.35 * w)

    cleaned = binary.copy()

    for y in range(0, upper_limit):
        xs = np.where(cleaned[y, :] > 0)[0]

        if len(xs) < 5:
            continue

        row_width = xs.max() - xs.min() + 1

        if row_width >= wide_row_threshold:
            row = np.zeros(w, dtype=np.uint8)
            row[corridor_left:corridor_right] = cleaned[y, corridor_left:corridor_right]
            cleaned[y, :] = row

    return cleaned


# ============================================================
# 5. DISTANCE TRANSFORM CORE EXTRACTION
# ============================================================

def extract_tooth_core(cleaned_binary):
    dist = cv2.distanceTransform(cleaned_binary, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return np.zeros_like(cleaned_binary), dist

    _, core = cv2.threshold(
        dist,
        0.30 * dist.max(),
        255,
        cv2.THRESH_BINARY
    )

    core = core.astype(np.uint8)

    return core, dist


# ============================================================
# 6. SELECT CENTRAL COMPONENT
# ============================================================

def select_central_component(core_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        core_mask,
        connectivity=8
    )

    h, w = core_mask.shape
    selected = np.zeros_like(core_mask)

    if num_labels <= 1:
        return selected

    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        if area < 20:
            continue

        if bw < max(6, int(0.025 * w)):
            continue

        cx, cy = centroids[label]
        center_score = 1.0 - abs(cx - (w / 2.0)) / (w / 2.0)
        center_score = np.clip(center_score, 0.0, 1.0)

        height_score = bh / h
        bottom_score = (y + bh) / h
        vertical_score = min(bh / (bw + 1), 3.0) / 3.0
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1)

        shape_penalty = 0.0
        if aspect_ratio > 6.0:
            shape_penalty += 1.0
        if bw < max(10, int(0.05 * w)) and bh > 0.35 * h:
            shape_penalty += 1.0
        if bw > 0.55 * w and y < 0.35 * h:
            shape_penalty += 1.5

        score = (
            (area * 0.3)
            + (center_score * 250)
            + (height_score * 180)
            + (bottom_score * 180)
            + (vertical_score * 120)
            - (shape_penalty * 250)
        )

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is not None:
        selected[labels == best_label] = 255

    return selected


# ============================================================
# 7. RECONSTRUCT FULL TOOTH MASK
# ============================================================

def fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)

    filled = mask | flood_inv

    return filled


def _interpolate_missing(values):
    values = np.asarray(values, dtype=np.float32)
    valid = ~np.isnan(values)

    if valid.sum() < 2:
        return values

    indices = np.arange(len(values), dtype=np.float32)
    values[~valid] = np.interp(indices[~valid], indices[valid], values[valid])

    return values


def _smooth_boundary(boundary, smooth_factor=120):
    original_valid = ~np.isnan(boundary)

    if original_valid.sum() < 2:
        return boundary

    valid_start = np.where(original_valid)[0].min()
    valid_end = np.where(original_valid)[0].max()

    boundary = _interpolate_missing(boundary)
    valid = ~np.isnan(boundary)

    if valid.sum() < 5:
        return boundary

    y = np.where(valid)[0].astype(np.float32)
    x = boundary[valid].astype(np.float32)

    try:
        spline = UnivariateSpline(y, x, s=smooth_factor)
        boundary[valid] = spline(y)
    except Exception:
        pass

    boundary[:valid_start] = np.nan
    boundary[valid_end + 1:] = np.nan

    return boundary


def extract_edge_linked_tooth_mask(enhanced_img, binary_hint):
    """
    Reconstruct the root from linked left/right outer edges.

    This is a fallback/refinement for cases where files or clamps dominate the
    threshold mask. It searches for vertical edge responses away from the canal,
    links those row-wise boundary candidates, smooths them, and fills the region
    between the two boundaries.
    """
    h, w = enhanced_img.shape

    blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = np.abs(grad_x)
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)

    hint_centers = []
    hint_widths = []

    for y in range(int(0.25 * h), h):
        xs = np.where(binary_hint[y, :] > 0)[0]

        if len(xs) < 5:
            continue

        hint_centers.append((xs.min() + xs.max()) / 2.0)
        hint_widths.append(xs.max() - xs.min() + 1)

    center_x = float(np.median(hint_centers)) if hint_centers else w / 2.0
    median_width = float(np.median(hint_widths)) if hint_widths else 0.35 * w

    min_half_width = max(6, int(0.10 * w))
    max_half_width = max(min_half_width + 4, int(max(0.24 * w, 0.75 * median_width)))
    max_half_width = min(max_half_width, int(0.48 * w))

    edge_floor = max(8.0, float(np.percentile(grad_x, 72)))

    left_boundary = np.full(h, np.nan, dtype=np.float32)
    right_boundary = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        center = int(round(center_x))

        left_start = max(0, center - max_half_width)
        left_end = max(0, center - min_half_width)
        right_start = min(w - 1, center + min_half_width)
        right_end = min(w - 1, center + max_half_width)

        if left_end > left_start:
            left_strip = grad_x[y, left_start:left_end]
            left_idx = int(np.argmax(left_strip))
            left_score = float(left_strip[left_idx])

            if left_score >= edge_floor:
                left_boundary[y] = left_start + left_idx

        if right_end > right_start:
            right_strip = grad_x[y, right_start:right_end]
            right_idx = int(np.argmax(right_strip))
            right_score = float(right_strip[right_idx])

            if right_score >= edge_floor:
                right_boundary[y] = right_start + right_idx

    left_boundary = _smooth_boundary(left_boundary)
    right_boundary = _smooth_boundary(right_boundary)

    edge_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        if np.isnan(left_boundary[y]) or np.isnan(right_boundary[y]):
            continue

        left = int(round(left_boundary[y]))
        right = int(round(right_boundary[y]))

        if right - left < min_half_width:
            continue

        left = max(0, left)
        right = min(w - 1, right)
        edge_mask[y, left:right + 1] = 255

    if cv2.countNonZero(edge_mask) == 0:
        return edge_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    edge_mask = fill_holes(edge_mask)

    return edge_mask


def reconstruct_tooth_mask(selected_core, cleaned_binary):
    # 1. Expand the central core to approximate the full tooth/root body
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    expanded = cv2.dilate(selected_core, kernel, iterations=6)

    # 2. Intersect with the cleaned edge map to keep the rough shape
    raw_tooth_mask = cv2.bitwise_and(expanded, cleaned_binary)
    raw_tooth_mask = fill_holes(raw_tooth_mask)

    # 3. Smooth the outer contour to remove jagged adaptive-threshold artifacts
    contours, _ = cv2.findContours(
        raw_tooth_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.003 * cv2.arcLength(largest_contour, True)
        smooth_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        smooth_mask = np.zeros_like(raw_tooth_mask)
        cv2.drawContours(smooth_mask, [smooth_contour], -1, 255, thickness=cv2.FILLED)
    else:
        smooth_mask = raw_tooth_mask

    # 4. Sub-pixel anti-aliasing on the mask boundary
    smoothed_blur = cv2.GaussianBlur(smooth_mask, (11, 11), 0)
    _, final_smooth_mask = cv2.threshold(smoothed_blur, 127, 255, cv2.THRESH_BINARY)

    # 5. Seal small ridges and gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_smooth_mask = cv2.morphologyEx(
        final_smooth_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1
    )

    return final_smooth_mask


# ============================================================
# 8. CROP ONLY TOOTH REGION
# ============================================================

def crop_tooth_region(img, mask, padding_ratio=0.08):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return img, mask, (0, 0)

    h, w = img.shape

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bw = x_max - x_min + 1
    bh = y_max - y_min + 1

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w, x_max + pad_x + 1)
    y1 = min(h, y_max + pad_y + 1)

    cropped_img = img[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]

    return cropped_img, cropped_mask, (x0, y0)


# ============================================================
# 9. MAKE TOOTH WHITE / REMOVE BACKGROUND
# ============================================================

def make_tooth_white(mask):
    tooth_white = np.zeros_like(mask)
    tooth_white[mask > 0] = 255
    return tooth_white


def remove_background(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


# ============================================================
# 10. MIDLINE EXTRACTION
# ============================================================

def compute_path_length(path):
    if len(path) < 2:
        return 0.0

    length = 0.0

    for p1, p2 in zip(path, path[1:]):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length += math.sqrt(dx * dx + dy * dy)

    return float(length)


def remove_outlier_center_points(center_points, max_jump=25):
    if len(center_points) < 3:
        return center_points

    filtered = [center_points[0]]

    for point in center_points[1:]:
        prev_x, _ = filtered[-1]
        curr_x, _ = point

        if abs(curr_x - prev_x) <= max_jump:
            filtered.append(point)

    return filtered


def extract_rowwise_midline(mask, smooth_factor=60, min_row_pixels=5):
    center_points = []

    for y in range(mask.shape[0]):
        xs = np.where(mask[y, :] > 0)[0]

        if len(xs) < min_row_pixels:
            continue

        x_left = xs.min()
        x_right = xs.max()
        x_center = (x_left + x_right) / 2.0

        center_points.append((x_center, y))

    if len(center_points) < 5:
        return [], 0.0

    center_points = remove_outlier_center_points(center_points)

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

    length = compute_path_length(path)

    return path, length


# ============================================================
# 11. VISUALIZATION
# ============================================================

def overlay_midline_on_mask(mask, path):
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    dash_length = 6
    gap_length = 5

    for i in range(0, len(path) - 1, dash_length + gap_length):
        segment = path[i:i + dash_length]

        for p1, p2 in zip(segment, segment[1:]):
            cv2.line(vis, p1, p2, (0, 0, 0), thickness=1)

    return vis


# ============================================================
# 12. FULL SEGMENTATION PIPELINE
# ============================================================

def segment_tooth(img_gray, debug=False):
    metal_removed, metal_mask = remove_metal_artifacts(img_gray)

    enhanced = enhance_roi(metal_removed)

    binary = adaptive_tooth_threshold(enhanced)

    metal_stripped_binary, metal_fixture_mask = remove_metal_fixtures(
        enhanced,
        binary
    )

    margin_cleared = clear_side_margins(metal_stripped_binary, margin_ratio=0.08)

    cleaned = morphology_cleanup(margin_cleared)

    clamp_suppressed = suppress_top_broad_artifacts(cleaned)

    core_mask, dist_transform = extract_tooth_core(clamp_suppressed)

    selected_core = select_central_component(core_mask)

    tooth_mask = reconstruct_tooth_mask(selected_core, clamp_suppressed)

    cropped_img, cropped_mask, offset = crop_tooth_region(
        enhanced,
        tooth_mask,
        padding_ratio=0.08
    )

    tooth_only = remove_background(cropped_img, cropped_mask)

    tooth_white = make_tooth_white(cropped_mask)

    path, length = extract_rowwise_midline(
        cropped_mask,
        smooth_factor=60,
        min_row_pixels=5
    )

    final_midline = overlay_midline_on_mask(cropped_mask, path)

    if debug:
        print(f"  Metal pixels: {cv2.countNonZero(metal_mask)}")
        print(f"  Metal fixture pixels: {cv2.countNonZero(metal_fixture_mask)}")
        print(f"  Binary pixels: {cv2.countNonZero(binary)}")
        print(f"  Metal-stripped binary pixels: {cv2.countNonZero(metal_stripped_binary)}")
        print(f"  Cleaned pixels: {cv2.countNonZero(cleaned)}")
        print(f"  Clamp-suppressed pixels: {cv2.countNonZero(clamp_suppressed)}")
        print(f"  Core pixels: {cv2.countNonZero(core_mask)}")
        print(f"  Selected core pixels: {cv2.countNonZero(selected_core)}")
        print(f"  Tooth mask pixels: {cv2.countNonZero(tooth_mask)}")
        print(f"  Cropped mask pixels: {cv2.countNonZero(cropped_mask)}")
        print(f"  Midline points: {len(path)}")
        print(f"  Midline length: {length:.2f} pixels")

    return {
        "metal_removed": metal_removed,
        "metal_mask": metal_mask,
        "enhanced": enhanced,
        "binary": binary,
        "metal_fixture_mask": metal_fixture_mask,
        "metal_stripped_binary": metal_stripped_binary,
        "margin_cleared": margin_cleared,
        "cleaned": cleaned,
        "clamp_suppressed": clamp_suppressed,
        "distance_transform": cv2.normalize(
            dist_transform,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8),
        "core_mask": core_mask,
        "selected_core": selected_core,
        "tooth_mask": tooth_mask,
        "cropped_img": cropped_img,
        "cropped_mask": cropped_mask,
        "tooth_only": tooth_only,
        "tooth_white": tooth_white,
        "final_midline": final_midline,
        "midline_path": path,
        "midline_length": length,
    }


# ============================================================
# 13. FILE PROCESSING
# ============================================================

def describe_image_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def process_image(path, output_dir, save_results=False, show=False, debug=False):
    img = load_image(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    result = segment_tooth(gray, debug=debug)

    if save_results:
        base = describe_image_path(path)
        os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir, f"{base}_01_metal_removed.png"), result["metal_removed"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_02_metal_mask.png"), result["metal_mask"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_03_enhanced.png"), result["enhanced"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_04_binary.png"), result["binary"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_05_metal_fixture_mask.png"), result["metal_fixture_mask"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_06_metal_stripped_binary.png"), result["metal_stripped_binary"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_07_margin_cleared.png"), result["margin_cleared"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_08_cleaned.png"), result["cleaned"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_09_clamp_suppressed.png"), result["clamp_suppressed"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_10_distance_transform.png"), result["distance_transform"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_11_core_mask.png"), result["core_mask"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_12_selected_core.png"), result["selected_core"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_13_tooth_mask.png"), result["tooth_mask"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_14_cropped_img.png"), result["cropped_img"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_15_cropped_mask.png"), result["cropped_mask"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_16_tooth_only.png"), result["tooth_only"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_17_tooth_white.png"), result["tooth_white"])
        cv2.imwrite(os.path.join(output_dir, f"{base}_18_final_midline.png"), result["final_midline"])

    if show or debug:
        cv2.imshow("1 Enhanced", result["enhanced"])
        cv2.imshow("2 Binary", result["binary"])
        cv2.imshow("3 Metal Fixture Mask", result["metal_fixture_mask"])
        cv2.imshow("4 Metal Stripped Binary", result["metal_stripped_binary"])
        cv2.imshow("5 Margin Cleared", result["margin_cleared"])
        cv2.imshow("6 Cleaned", result["cleaned"])
        cv2.imshow("7 Clamp Suppressed", result["clamp_suppressed"])
        cv2.imshow("8 Distance Transform", result["distance_transform"])
        cv2.imshow("9 Core Mask", result["core_mask"])
        cv2.imshow("10 Selected Core", result["selected_core"])
        cv2.imshow("11 Tooth Mask", result["tooth_mask"])
        cv2.imshow("12 Cropped Image", result["cropped_img"])
        cv2.imshow("13 Cropped Mask", result["cropped_mask"])
        cv2.imshow("14 Tooth Only", result["tooth_only"])
        cv2.imshow("15 Tooth White", result["tooth_white"])
        cv2.imshow("16 Final Midline", result["final_midline"])

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


# ============================================================
# 14. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment tooth ROI, remove background, crop tooth, and extract midline"
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input directory containing tooth ROI images"
    )

    parser.add_argument(
        "--output-dir",
        default=get_dataset_path("working_length", "features"),
        help="Directory to save extracted features and debug images"
    )

    parser.add_argument(
        "--csv",
        default=os.path.join(
            get_dataset_path("working_length", "features"),
            "tooth_midline_lengths.csv"
        ),
        help="Output CSV file for midline length values"
    )

    parser.add_argument("--save", action="store_true", help="Save output images")
    parser.add_argument("--show", action="store_true", help="Show debug images")
    parser.add_argument("--debug", action="store_true", help="Print diagnostics")

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
