import os
import csv
import cv2
import numpy as np


# ============================================================
# Basic utilities
# ============================================================

def ensure_uint8_gray(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    return img


def read_binary_mask(path, target_shape=None):
    if path is None or not os.path.exists(path):
        return None

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = ensure_uint8_gray(mask)

    if mask is None:
        return None

    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask = ((mask > 0).astype(np.uint8) * 255)

    return mask


def safe_imwrite(path, image):
    if path is None:
        return

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    cv2.imwrite(path, image)


def odd(value, minimum=3):
    value = int(round(value))

    if value < minimum:
        value = minimum

    if value % 2 == 0:
        value += 1

    return value


def safe_div(a, b, default=0.0):
    if b is None or abs(float(b)) < 1e-8:
        return default

    return float(a) / float(b)


def safe_mean(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.std(values))


def safe_max(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.max(values))


def safe_percentile(values, p):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0

    return float(np.percentile(values, p))


def find_mask_file(mask_dir, image_file_name, prefix):
    """
    Finds mask using different extensions.

    Example:
        image: frac01.jpg
        mask : polynomial_root_frac01.png
        mask : polynomial_root_frac01.jpg
    """

    name, image_ext = os.path.splitext(image_file_name)

    valid_ext = [
        ".png",
        image_ext,
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    ]

    for ext in valid_ext:
        path = os.path.join(mask_dir, f"{prefix}{name}{ext}")

        if os.path.exists(path):
            return path

    return None


# ============================================================
# Preprocessing / upscaling
# ============================================================

def preprocess_for_debug(img_uint8):
    p1, p99 = np.percentile(img_uint8, [1, 99])

    if p99 > p1:
        norm = np.clip(
            (img_uint8.astype(np.float32) - p1) * 255.0 / (p99 - p1),
            0,
            255
        ).astype(np.uint8)
    else:
        norm = img_uint8.copy()

    clahe = cv2.createCLAHE(
        clipLimit=1.20,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(norm)

    enhanced = cv2.bilateralFilter(
        enhanced,
        d=5,
        sigmaColor=18,
        sigmaSpace=7
    )

    return enhanced


def upscale_inputs(img, root_mask, pdl_mask, scale=2):
    scale = max(1, int(scale))

    if scale == 1:
        return img, root_mask, pdl_mask

    h, w = img.shape
    new_size = (w * scale, h * scale)

    img_up = cv2.resize(
        img,
        new_size,
        interpolation=cv2.INTER_CUBIC
    )

    root_up = cv2.resize(
        root_mask,
        new_size,
        interpolation=cv2.INTER_NEAREST
    )

    pdl_up = cv2.resize(
        pdl_mask,
        new_size,
        interpolation=cv2.INTER_NEAREST
    )

    root_up = ((root_up > 0).astype(np.uint8) * 255)
    pdl_up = ((pdl_up > 0).astype(np.uint8) * 255)

    return img_up, root_up, pdl_up


def clean_pdl_mask_for_pattern(pdl_mask, root_mask):
    """
    Small cleanup before measuring pattern.
    """

    mask = pdl_mask.copy()

    mask[root_mask > 0] = 0

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    kernel_vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 5)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vertical)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    mask[root_mask > 0] = 0

    return mask


# ============================================================
# Root geometry
# ============================================================

def extract_root_boundaries(root_mask):
    """
    Extract left/right root boundary row by row.
    """

    h, w = root_mask.shape

    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)
    center = np.full(h, np.nan, dtype=np.float32)
    width = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(root_mask[y] > 0)[0]

        if len(xs) > 0:
            left[y] = xs[0]
            right[y] = xs[-1]
            center[y] = (xs[0] + xs[-1]) / 2.0
            width[y] = xs[-1] - xs[0] + 1

    valid = ~np.isnan(center)

    if np.count_nonzero(valid) < 8:
        return None

    ys = np.arange(h)

    left = np.interp(ys, ys[valid], left[valid])
    right = np.interp(ys, ys[valid], right[valid])
    center = np.interp(ys, ys[valid], center[valid])
    width = np.interp(ys, ys[valid], width[valid])

    k = odd(h * 0.045, 9)

    left = cv2.GaussianBlur(left.reshape(-1, 1), (1, k), 0).flatten()
    right = cv2.GaussianBlur(right.reshape(-1, 1), (1, k), 0).flatten()
    center = cv2.GaussianBlur(center.reshape(-1, 1), (1, k), 0).flatten()
    width = cv2.GaussianBlur(width.reshape(-1, 1), (1, k), 0).flatten()

    valid_rows = np.where(valid)[0]

    return {
        "left": left,
        "right": right,
        "center": center,
        "width": width,
        "y_min": int(valid_rows[0]),
        "y_max": int(valid_rows[-1])
    }


def get_root_zones(root_geom):
    y_min = root_geom["y_min"]
    y_max = root_geom["y_max"]

    height = y_max - y_min + 1

    coronal_end = y_min + int(height / 3)
    middle_end = y_min + int(2 * height / 3)

    return {
        "coronal": (y_min, coronal_end),
        "middle": (coronal_end + 1, middle_end),
        "apical": (middle_end + 1, y_max)
    }


# ============================================================
# Row-wise PDL width measurement
# ============================================================

def measure_pdl_width_profile(
    root_mask,
    pdl_mask,
    root_geom,
    max_search_px=48,
    gap_tolerance_px=4,
    pixel_scale=2
):
    """
    Measures PDL width on left and right sides of the root.

    This is the main part needed for:
        - uniform widening
        - side-dominant widening
        - irregular localized widening
    """

    h, w = pdl_mask.shape

    left_root = root_geom["left"]
    right_root = root_geom["right"]
    y_min = root_geom["y_min"]
    y_max = root_geom["y_max"]

    rows = []

    left_widths = []
    right_widths = []
    total_widths = []
    asymmetries = []

    left_present = []
    right_present = []

    left_outer_x = np.full(h, np.nan, dtype=np.float32)
    right_outer_x = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_min, y_max + 1):
        lx = int(round(left_root[y]))
        rx = int(round(right_root[y]))

        # --------------------------------------------------------
        # Left side width
        # --------------------------------------------------------

        left_start = max(0, lx - max_search_px)
        left_end = max(0, lx - 1)

        left_region = pdl_mask[y, left_start:left_end + 1] > 0

        left_width = 0.0

        if left_region.size > 0 and np.count_nonzero(left_region) > 0:
            xs = np.arange(left_start, left_end + 1)
            pdl_xs = xs[left_region]

            inner_x = int(np.max(pdl_xs))
            outer_x = int(np.min(pdl_xs))

            # Accept if the dark region is close enough to the root boundary.
            if abs(left_end - inner_x) <= gap_tolerance_px + 2:
                left_width = lx - outer_x
                left_outer_x[y] = outer_x

        # --------------------------------------------------------
        # Right side width
        # --------------------------------------------------------

        right_start = min(w - 1, rx + 1)
        right_end = min(w - 1, rx + max_search_px)

        right_region = pdl_mask[y, right_start:right_end + 1] > 0

        right_width = 0.0

        if right_region.size > 0 and np.count_nonzero(right_region) > 0:
            xs = np.arange(right_start, right_end + 1)
            pdl_xs = xs[right_region]

            inner_x = int(np.min(pdl_xs))
            outer_x = int(np.max(pdl_xs))

            if abs(inner_x - right_start) <= gap_tolerance_px + 2:
                right_width = outer_x - rx
                right_outer_x[y] = outer_x

        left_width = left_width / float(pixel_scale)
        right_width = right_width / float(pixel_scale)

        total_width = left_width + right_width
        asymmetry = abs(left_width - right_width)

        rows.append(y / float(pixel_scale))
        left_widths.append(left_width)
        right_widths.append(right_width)
        total_widths.append(total_width)
        asymmetries.append(asymmetry)

        left_present.append(left_width > 0)
        right_present.append(right_width > 0)

    return {
        "rows": np.asarray(rows, dtype=np.float32),
        "left_widths": np.asarray(left_widths, dtype=np.float32),
        "right_widths": np.asarray(right_widths, dtype=np.float32),
        "total_widths": np.asarray(total_widths, dtype=np.float32),
        "asymmetries": np.asarray(asymmetries, dtype=np.float32),
        "left_present": np.asarray(left_present, dtype=bool),
        "right_present": np.asarray(right_present, dtype=bool),
        "left_outer_x": left_outer_x,
        "right_outer_x": right_outer_x
    }


def values_in_zone(width_profile, zone, pixel_scale=2):
    rows = width_profile["rows"]

    y1 = zone[0] / float(pixel_scale)
    y2 = zone[1] / float(pixel_scale)

    idx = (rows >= y1) & (rows <= y2)

    return {
        "left": width_profile["left_widths"][idx],
        "right": width_profile["right_widths"][idx],
        "total": width_profile["total_widths"][idx],
        "asymmetry": width_profile["asymmetries"][idx]
    }


# ============================================================
# PDL pattern feature computation
# ============================================================

def longest_true_run(bool_array):
    longest = 0
    current = 0

    for value in bool_array:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def compute_pattern_features(width_profile, root_geom, pdl_mask, root_mask, pixel_scale=2):
    left = width_profile["left_widths"]
    right = width_profile["right_widths"]
    total = width_profile["total_widths"]
    asym = width_profile["asymmetries"]

    valid = total > 0

    valid_total = total[valid]
    valid_left = left[valid]
    valid_right = right[valid]
    valid_asym = asym[valid]

    root_area_px = np.count_nonzero(root_mask) / float(pixel_scale ** 2)
    pdl_area_px = np.count_nonzero(pdl_mask) / float(pixel_scale ** 2)

    root_width_values = root_geom["width"][root_geom["y_min"]:root_geom["y_max"] + 1] / float(pixel_scale)

    root_mean_width_px = safe_mean(root_width_values)
    root_height_px = (root_geom["y_max"] - root_geom["y_min"] + 1) / float(pixel_scale)

    mean_left = safe_mean(valid_left)
    mean_right = safe_mean(valid_right)
    mean_total = safe_mean(valid_total)
    mean_asym = safe_mean(valid_asym)

    max_left = safe_max(valid_left)
    max_right = safe_max(valid_right)
    max_total = safe_max(valid_total)
    max_asym = safe_max(valid_asym)

    p90_total = safe_percentile(valid_total, 90)
    p95_total = safe_percentile(valid_total, 95)
    median_total = safe_percentile(valid_total, 50)

    width_std = safe_std(valid_total)
    width_cv = safe_div(width_std, mean_total)

    asymmetry_ratio = safe_div(mean_asym, mean_total)

    side_difference_px = abs(mean_left - mean_right)
    side_dominance_ratio = safe_div(side_difference_px, mean_left + mean_right)

    if mean_left > mean_right * 1.15:
        dominant_side = "LEFT"
    elif mean_right > mean_left * 1.15:
        dominant_side = "RIGHT"
    else:
        dominant_side = "BALANCED"

    if dominant_side == "LEFT":
        side_rows = left > (right + 1.0)
    elif dominant_side == "RIGHT":
        side_rows = right > (left + 1.0)
    else:
        side_rows = np.zeros_like(total, dtype=bool)

    side_consistency = safe_div(np.count_nonzero(side_rows), len(total))

    # ------------------------------------------------------------
    # Localized widening detection
    # ------------------------------------------------------------

    q25 = safe_percentile(valid_total, 25)
    q75 = safe_percentile(valid_total, 75)
    iqr = q75 - q25

    localized_width_threshold = max(
        median_total + 1.5 * iqr,
        mean_total + width_std,
        p90_total
    )

    localized_widening_rows = total >= localized_width_threshold

    localized_widening_percent = 100.0 * safe_div(
        np.count_nonzero(localized_widening_rows),
        len(total)
    )

    longest_localized_run = longest_true_run(localized_widening_rows)
    longest_localized_run_ratio = safe_div(longest_localized_run, len(total))

    asym_q75 = safe_percentile(valid_asym, 75)
    asym_iqr = asym_q75 - safe_percentile(valid_asym, 25)

    localized_asymmetry_threshold = max(
        safe_mean(valid_asym) + 1.5 * safe_std(valid_asym),
        asym_q75 + 1.5 * asym_iqr,
        mean_total * 0.35,
        2.0
    )

    localized_asymmetry_rows = asym >= localized_asymmetry_threshold

    localized_asymmetry_percent = 100.0 * safe_div(
        np.count_nonzero(localized_asymmetry_rows),
        len(asym)
    )

    max_to_mean_width_ratio = safe_div(max_total, mean_total)
    p95_to_median_width_ratio = safe_div(p95_total, median_total)

    # ------------------------------------------------------------
    # Zone features
    # ------------------------------------------------------------

    zones = get_root_zones(root_geom)

    coronal_values = values_in_zone(width_profile, zones["coronal"], pixel_scale)
    middle_values = values_in_zone(width_profile, zones["middle"], pixel_scale)
    apical_values = values_in_zone(width_profile, zones["apical"], pixel_scale)

    coronal_mean = safe_mean(coronal_values["total"])
    middle_mean = safe_mean(middle_values["total"])
    apical_mean = safe_mean(apical_values["total"])

    apical_to_coronal_ratio = safe_div(apical_mean, coronal_mean)

    # ------------------------------------------------------------
    # Continuity
    # ------------------------------------------------------------

    left_present = width_profile["left_present"]
    right_present = width_profile["right_present"]
    any_present = left_present | right_present

    left_continuity_percent = 100.0 * safe_div(np.count_nonzero(left_present), len(left_present))
    right_continuity_percent = 100.0 * safe_div(np.count_nonzero(right_present), len(right_present))
    total_continuity_percent = 100.0 * safe_div(np.count_nonzero(any_present), len(any_present))

    total_longest_run_ratio = safe_div(longest_true_run(any_present), len(any_present))

    # ------------------------------------------------------------
    # Irregularity index
    # ------------------------------------------------------------

    irregularity_index = (
        0.30 * min(1.0, asymmetry_ratio / 0.60)
        + 0.25 * min(1.0, width_cv / 0.70)
        + 0.20 * min(1.0, localized_asymmetry_percent / 35.0)
        + 0.15 * min(1.0, localized_widening_percent / 35.0)
        + 0.10 * min(1.0, safe_div(max_to_mean_width_ratio, 2.50))
    )

    features = {
        "num_profile_rows": int(len(total)),
        "valid_pdl_row_count": int(np.count_nonzero(valid)),
        "valid_pdl_row_percent": 100.0 * safe_div(np.count_nonzero(valid), len(total)),

        "root_area_px": root_area_px,
        "pdl_area_px": pdl_area_px,
        "pdl_root_area_ratio": safe_div(pdl_area_px, root_area_px),

        "root_height_px": root_height_px,
        "root_mean_width_px": root_mean_width_px,

        "mean_left_width_px": mean_left,
        "mean_right_width_px": mean_right,
        "mean_total_width_px": mean_total,

        "max_left_width_px": max_left,
        "max_right_width_px": max_right,
        "max_total_width_px": max_total,

        "p90_total_width_px": p90_total,
        "p95_total_width_px": p95_total,
        "median_total_width_px": median_total,

        "width_std_px": width_std,
        "width_cv": width_cv,

        "mean_asymmetry_px": mean_asym,
        "max_asymmetry_px": max_asym,
        "asymmetry_ratio": asymmetry_ratio,

        "side_difference_px": side_difference_px,
        "side_dominance_ratio": side_dominance_ratio,
        "dominant_side": dominant_side,
        "side_consistency": side_consistency,

        "localized_width_threshold_px": localized_width_threshold,
        "localized_widening_row_count": int(np.count_nonzero(localized_widening_rows)),
        "localized_widening_percent": localized_widening_percent,
        "longest_localized_run_ratio": longest_localized_run_ratio,

        "localized_asymmetry_threshold_px": localized_asymmetry_threshold,
        "localized_asymmetry_row_count": int(np.count_nonzero(localized_asymmetry_rows)),
        "localized_asymmetry_percent": localized_asymmetry_percent,

        "max_to_mean_width_ratio": max_to_mean_width_ratio,
        "p95_to_median_width_ratio": p95_to_median_width_ratio,

        "coronal_mean_width_px": coronal_mean,
        "middle_mean_width_px": middle_mean,
        "apical_mean_width_px": apical_mean,
        "apical_to_coronal_width_ratio": apical_to_coronal_ratio,

        "left_continuity_percent": left_continuity_percent,
        "right_continuity_percent": right_continuity_percent,
        "total_continuity_percent": total_continuity_percent,
        "total_longest_run_ratio": total_longest_run_ratio,

        "irregularity_index": irregularity_index
    }

    return features


# ============================================================
# Dentist-rule PDL pattern classification
# ============================================================

def classify_pdl_pattern(features):
    """
    Dentist-guided rule classification.

    1 = Uniform widening
    2 = Side-dominant widening
    3 = Irregular localized widening
    """

    valid_pdl_row_percent = features["valid_pdl_row_percent"]
    asymmetry_ratio = features["asymmetry_ratio"]
    side_dominance_ratio = features["side_dominance_ratio"]
    side_consistency = features["side_consistency"]
    localized_asymmetry_percent = features["localized_asymmetry_percent"]
    localized_widening_percent = features["localized_widening_percent"]
    width_cv = features["width_cv"]
    irregularity_index = features["irregularity_index"]
    max_to_mean_width_ratio = features["max_to_mean_width_ratio"]

    # No reliable PDL detected.
    if valid_pdl_row_percent < 8:
        return {
            "pdl_pattern_score_rule": 0,
            "pdl_pattern_label_rule": "NO_RELIABLE_PDL",
            "fracture_probability_score_rule": 0,
            "fracture_probability_label_rule": "VERY_LOW",
            "rct_success_score_rule": 1.0,
            "rct_success_label_rule": "HIGH"
        }

    # ------------------------------------------------------------
    # 3 = Irregular localized widening
    # High asymmetry, strong localized widening, or unstable width profile.
    # ------------------------------------------------------------

    irregular_condition = (
        asymmetry_ratio >= 0.45
        or localized_asymmetry_percent >= 22.0
        or localized_widening_percent >= 25.0
        or irregularity_index >= 0.58
        or max_to_mean_width_ratio >= 2.30
        or width_cv >= 0.75
    )

    if irregular_condition:
        return {
            "pdl_pattern_score_rule": 3,
            "pdl_pattern_label_rule": "IRREGULAR_LOCALIZED_WIDENING",
            "fracture_probability_score_rule": 3,
            "fracture_probability_label_rule": "HIGH",
            "rct_success_score_rule": 0.0,
            "rct_success_label_rule": "LOW"
        }

    # ------------------------------------------------------------
    # 2 = Side-dominant widening
    # One side is consistently wider, but not highly localized/irregular.
    # ------------------------------------------------------------

    side_dominant_condition = (
        side_dominance_ratio >= 0.20
        and side_consistency >= 0.45
        and asymmetry_ratio >= 0.20
    )

    if side_dominant_condition:
        return {
            "pdl_pattern_score_rule": 2,
            "pdl_pattern_label_rule": "SIDE_DOMINANT_WIDENING",
            "fracture_probability_score_rule": 2,
            "fracture_probability_label_rule": "MODERATE",
            "rct_success_score_rule": 0.5,
            "rct_success_label_rule": "MODERATE"
        }

    # ------------------------------------------------------------
    # 1 = Uniform widening
    # PDL is approximately even around root.
    # ------------------------------------------------------------

    return {
        "pdl_pattern_score_rule": 1,
        "pdl_pattern_label_rule": "UNIFORM_WIDENING",
        "fracture_probability_score_rule": 1,
        "fracture_probability_label_rule": "LOW",
        "rct_success_score_rule": 1.0,
        "rct_success_label_rule": "HIGH"
    }


# ============================================================
# Debug visualization
# ============================================================

def overlay_mask(base_gray_or_bgr, mask, color_bgr, alpha=0.35):
    if len(base_gray_or_bgr.shape) == 2:
        out = cv2.cvtColor(base_gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        out = base_gray_or_bgr.copy()

    color = np.zeros_like(out)
    color[:, :] = color_bgr

    idx = mask > 0

    out[idx] = (
        (1.0 - alpha) * out[idx]
        + alpha * color[idx]
    ).astype(np.uint8)

    return out


def create_width_profile_panel(width_profile, image_shape):
    h, w = image_shape

    panel = np.zeros((h, w, 3), dtype=np.uint8)

    total = width_profile["total_widths"]
    left = width_profile["left_widths"]
    right = width_profile["right_widths"]
    rows = width_profile["rows"]

    if len(total) == 0:
        return panel

    max_value = max(1.0, float(np.percentile(total, 95)))

    for i, y_value in enumerate(rows):
        y = int(round(y_value))

        if y < 0 or y >= h:
            continue

        total_len = int(np.clip(total[i] / max_value, 0, 1) * (w - 1))
        left_len = int(np.clip(left[i] / max_value, 0, 1) * (w - 1))
        right_len = int(np.clip(right[i] / max_value, 0, 1) * (w - 1))

        cv2.line(panel, (0, y), (total_len, y), (255, 255, 255), 1)
        cv2.line(panel, (0, y), (left_len, y), (0, 255, 255), 1)
        cv2.line(panel, (0, y), (right_len, y), (0, 180, 255), 1)

    return panel


def create_debug_image(img, root_mask, pdl_mask, root_geom, width_profile, features):
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    overlay = base.copy()

    overlay = overlay_mask(
        overlay,
        root_mask,
        color_bgr=(0, 0, 255),
        alpha=0.22
    )

    overlay = overlay_mask(
        overlay,
        pdl_mask,
        color_bgr=(0, 255, 255),
        alpha=0.55
    )

    left_root = root_geom["left"]
    right_root = root_geom["right"]
    y_min = root_geom["y_min"]
    y_max = root_geom["y_max"]

    left_points = []
    right_points = []

    for y in range(y_min, y_max + 1):
        left_points.append([int(left_root[y]), y])
        right_points.append([int(right_root[y]), y])

    if len(left_points) > 2:
        cv2.polylines(
            overlay,
            [np.array(left_points, dtype=np.int32)],
            False,
            (0, 0, 255),
            1
        )

    if len(right_points) > 2:
        cv2.polylines(
            overlay,
            [np.array(right_points, dtype=np.int32)],
            False,
            (0, 0, 255),
            1
        )

    left_outer = width_profile["left_outer_x"]
    right_outer = width_profile["right_outer_x"]

    for y in range(y_min, y_max + 1, 8):
        lx = int(left_root[y])
        rx = int(right_root[y])

        if np.isfinite(left_outer[y]):
            cv2.line(
                overlay,
                (int(left_outer[y]), y),
                (lx, y),
                (0, 255, 0),
                1
            )

        if np.isfinite(right_outer[y]):
            cv2.line(
                overlay,
                (rx, y),
                (int(right_outer[y]), y),
                (0, 255, 0),
                1
            )

    profile_panel = create_width_profile_panel(
        width_profile,
        image_shape=img.shape
    )

    text_panel = np.zeros_like(base)

    lines = [
        f"PDL Pattern: {features['pdl_pattern_label_rule']}",
        f"Pattern Score: {features['pdl_pattern_score_rule']}",
        f"Fracture Prob: {features['fracture_probability_label_rule']} ({features['fracture_probability_score_rule']})",
        f"RCT Success: {features['rct_success_label_rule']} ({features['rct_success_score_rule']})",
        f"Mean L/R: {features['mean_left_width_px']:.2f} / {features['mean_right_width_px']:.2f}",
        f"Asym Ratio: {features['asymmetry_ratio']:.2f}",
        f"Side Dom Ratio: {features['side_dominance_ratio']:.2f}",
        f"Localized Asym %: {features['localized_asymmetry_percent']:.1f}",
        f"Localized Width %: {features['localized_widening_percent']:.1f}",
        f"Irregularity: {features['irregularity_index']:.2f}"
    ]

    y_text = 25

    for line in lines:
        cv2.putText(
            text_panel,
            line,
            (8, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y_text += 22

    debug = np.hstack(
        [
            base,
            overlay,
            cv2.cvtColor(root_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pdl_mask, cv2.COLOR_GRAY2BGR),
            profile_panel,
            text_panel
        ]
    )

    return debug


# ============================================================
# Main extraction function
# ============================================================

def extract_pdl_pattern_features(
    image_path,
    root_mask_path,
    pdl_mask_path,
    debug_path=None,
    upscale_factor=2
):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = ensure_uint8_gray(img)

    if img is None:
        print("ERROR: Could not read image:", image_path)
        return None

    root_mask = read_binary_mask(
        root_mask_path,
        target_shape=img.shape
    )

    pdl_mask = read_binary_mask(
        pdl_mask_path,
        target_shape=img.shape
    )

    if root_mask is None:
        print("ERROR: Could not read root mask:", root_mask_path)
        return None

    if pdl_mask is None:
        print("ERROR: Could not read PDL mask:", pdl_mask_path)
        return None

    img_up, root_up, pdl_up = upscale_inputs(
        img,
        root_mask,
        pdl_mask,
        scale=upscale_factor
    )

    pdl_up = clean_pdl_mask_for_pattern(
        pdl_mask=pdl_up,
        root_mask=root_up
    )

    root_geom = extract_root_boundaries(root_up)

    if root_geom is None:
        print("ERROR: Could not extract root boundary:", image_path)
        return None

    width_profile = measure_pdl_width_profile(
        root_mask=root_up,
        pdl_mask=pdl_up,
        root_geom=root_geom,
        max_search_px=max(30, int(24 * upscale_factor)),
        gap_tolerance_px=max(3, int(2 * upscale_factor)),
        pixel_scale=upscale_factor
    )

    features = compute_pattern_features(
        width_profile=width_profile,
        root_geom=root_geom,
        pdl_mask=pdl_up,
        root_mask=root_up,
        pixel_scale=upscale_factor
    )

    rule_scores = classify_pdl_pattern(features)

    features.update(rule_scores)

    features["file_name"] = os.path.basename(image_path)
    features["image_name"] = os.path.basename(image_path)
    features["upscale_factor"] = upscale_factor

    if debug_path:
        debug = create_debug_image(
            img=img_up,
            root_mask=root_up,
            pdl_mask=pdl_up,
            root_geom=root_geom,
            width_profile=width_profile,
            features=features
        )

        safe_imwrite(debug_path, debug)

    print(
        f"{os.path.basename(image_path)} | "
        f"pattern={features['pdl_pattern_label_rule']} "
        f"score={features['pdl_pattern_score_rule']} | "
        f"fracture={features['fracture_probability_label_rule']} | "
        f"rct={features['rct_success_label_rule']} | "
        f"asym={features['asymmetry_ratio']:.2f} | "
        f"side={features['side_dominance_ratio']:.2f} | "
        f"irreg={features['irregularity_index']:.2f}"
    )

    return features


def save_features_csv(features_list, csv_path):
    if len(features_list) == 0:
        print("No features to save.")
        return

    folder = os.path.dirname(csv_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    all_keys = []

    for row in features_list:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_keys
        )

        writer.writeheader()

        for row in features_list:
            writer.writerow(row)

    print("Saved:", csv_path)