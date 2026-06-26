import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

EPS = 1e-6


# ============================================================
# Basic utilities
# ============================================================

def ensure_uint8_binary(mask):
    """
    Converts any mask to clean binary uint8:
        foreground = 255
        background = 0
    """
    if mask is None:
        return None

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = mask.astype(np.uint8)
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask > 0] = 255

    return binary


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def normalize_stem_for_matching(name):
    """
    Converts filenames to a comparable stem.

    Example:
        frac01.jpg                     -> frac01
        polynomial_root_frac01.jpg     -> frac01
        dark_pdl_frac01.jpg            -> frac01
    """

    stem = os.path.splitext(os.path.basename(str(name)))[0].lower()

    removable_prefixes = [
        "polynomial_root_",
        "root_",
        "dark_pdl_",
        "pdl_",
        "mask_",
    ]

    removable_suffixes = [
        "_polynomial_root",
        "_root_mask",
        "_root",
        "_dark_pdl_mask",
        "_dark_pdl",
        "_pdl_mask",
        "_pdl",
        "_mask",
    ]

    changed = True

    while changed:
        changed = False

        for prefix in removable_prefixes:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                changed = True

        for suffix in removable_suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True

    compact = (
        stem.replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace(".", "")
    )

    return stem, compact


def find_mask_file(image_name, mask_dir):
    """
    Finds the matching mask file for an image.

    Handles:
        frac01.jpg -> polynomial_root_frac01.jpg
        frac01.jpg -> dark_pdl_frac01.jpg
    """

    if not os.path.exists(mask_dir):
        return None

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    image_stem, image_compact = normalize_stem_for_matching(image_name)

    mask_files = [
        f for f in os.listdir(mask_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    # 1. Exact normalized match
    for filename in mask_files:
        mask_stem, mask_compact = normalize_stem_for_matching(filename)

        if mask_stem == image_stem:
            return os.path.join(mask_dir, filename)

    # 2. Compact normalized match
    for filename in mask_files:
        mask_stem, mask_compact = normalize_stem_for_matching(filename)

        if mask_compact == image_compact:
            return os.path.join(mask_dir, filename)

    # 3. Fallback: image stem contained in mask stem
    for filename in mask_files:
        original_mask_stem = os.path.splitext(filename)[0].lower()

        if image_stem in original_mask_stem:
            return os.path.join(mask_dir, filename)

        if image_compact in original_mask_stem.replace("_", "").replace("-", ""):
            return os.path.join(mask_dir, filename)

    return None


def resize_mask_to_image(mask, image_shape):
    """
    Resizes mask to image size if dimensions do not match.
    Uses nearest-neighbor interpolation to preserve labels.
    """

    h, w = image_shape[:2]

    if mask.shape[:2] == (h, w):
        return mask

    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return resized


def largest_connected_component(mask):
    """
    Keeps largest connected component.
    Useful for root mask cleanup.
    """

    binary = ensure_uint8_binary(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return binary

    # Ignore background label 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    output = np.zeros_like(binary, dtype=np.uint8)
    output[labels == largest_label] = 255

    return output


def clean_root_mask(root_mask):
    """
    Root should be one smooth main component.
    """

    root_mask = ensure_uint8_binary(root_mask)
    root_mask = largest_connected_component(root_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    root_mask = cv2.morphologyEx(root_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    root_mask = cv2.morphologyEx(root_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    root_mask = largest_connected_component(root_mask)

    return root_mask


def clean_pdl_mask(pdl_mask):
    """
    PDL can be thin and disconnected, so do not keep only largest component.
    Only light cleanup is applied.
    """

    pdl_mask = ensure_uint8_binary(pdl_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    pdl_mask = cv2.morphologyEx(pdl_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    pdl_mask = cv2.morphologyEx(pdl_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    return pdl_mask


def safe_mean(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def safe_std(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return 0.0

    return float(np.std(values))


def safe_percentile(values, q):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, q))


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    valid = np.isfinite(a) & np.isfinite(b)

    if np.sum(valid) < 3:
        return 0.0

    a = a[valid]
    b = b[valid]

    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# Row-wise width profile extraction
# ============================================================

def get_root_bounds_per_row(root_mask, min_root_width_px=5):
    """
    For each row, finds left and right root boundary.
    Returns:
        list of dicts:
            {
                "y": y,
                "left_x": left_x,
                "right_x": right_x,
                "root_width": width
            }
    """

    root_binary = root_mask > 0
    h, w = root_binary.shape

    rows = []

    for y in range(h):
        xs = np.where(root_binary[y])[0]

        if len(xs) == 0:
            continue

        left_x = int(xs.min())
        right_x = int(xs.max())
        root_width = right_x - left_x + 1

        if root_width < min_root_width_px:
            continue

        rows.append(
            {
                "y": int(y),
                "left_x": left_x,
                "right_x": right_x,
                "root_width": int(root_width),
            }
        )

    return rows


def scan_pdl_width_from_edge(
    row_binary,
    edge_x,
    direction,
    max_search_px=80,
    initial_gap_tolerance=5,
    internal_gap_tolerance=2,
):
    """
    Measures PDL width from root boundary outward.

    direction:
        -1 = scan left
         1 = scan right

    The function:
        1. Starts from root edge.
        2. Allows small initial gap.
        3. Finds first PDL pixel.
        4. Continues until PDL run ends.
        5. Allows small internal gaps.
    """

    w = len(row_binary)

    x = edge_x + direction
    searched = 0

    # ------------------------------------------------------------
    # Find first PDL pixel near the root edge
    # ------------------------------------------------------------
    found_start = False
    start_x = None

    while 0 <= x < w and searched < max_search_px:
        if row_binary[x]:
            found_start = True
            start_x = x
            break

        searched += 1

        if searched > initial_gap_tolerance:
            break

        x += direction

    if not found_start:
        return 0.0

    # ------------------------------------------------------------
    # Continue PDL run
    # ------------------------------------------------------------
    last_pdl_x = start_x
    gap_count = 0

    x = start_x

    while 0 <= x < w and abs(x - edge_x) <= max_search_px:
        if row_binary[x]:
            last_pdl_x = x
            gap_count = 0
        else:
            gap_count += 1
            if gap_count > internal_gap_tolerance:
                break

        x += direction

    width = abs(last_pdl_x - edge_x)

    return float(width)


def extract_width_profiles(
    root_mask,
    pdl_mask,
    max_search_px=80,
    min_valid_rows=10,
):
    """
    Extracts row-wise left and right PDL width profiles.

    Returns:
        profile_df
    """

    rows = get_root_bounds_per_row(root_mask)

    pdl_binary = pdl_mask > 0

    profile_rows = []

    for row in rows:
        y = row["y"]
        left_x = row["left_x"]
        right_x = row["right_x"]

        pdl_row = pdl_binary[y]

        left_width = scan_pdl_width_from_edge(
            row_binary=pdl_row,
            edge_x=left_x,
            direction=-1,
            max_search_px=max_search_px,
        )

        right_width = scan_pdl_width_from_edge(
            row_binary=pdl_row,
            edge_x=right_x,
            direction=1,
            max_search_px=max_search_px,
        )

        total_width = left_width + right_width

        # Keep rows where at least one side has detected PDL
        if total_width <= 0:
            continue

        profile_rows.append(
            {
                "y": y,
                "left_width_px": left_width,
                "right_width_px": right_width,
                "total_width_px": total_width,
                "asymmetry_px": abs(left_width - right_width),
            }
        )

    profile_df = pd.DataFrame(profile_rows)

    if len(profile_df) < min_valid_rows:
        return profile_df

    return profile_df


# ============================================================
# 1D smoothing and local peak logic
# ============================================================

def rolling_median(values, window_size=5):
    """
    Simple median smoothing without scipy dependency.
    """

    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    if window_size <= 1:
        return values.copy()

    pad = window_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")

    output = []

    for i in range(len(values)):
        local_window = padded[i:i + window_size]
        output.append(np.median(local_window))

    return np.asarray(output, dtype=np.float32)


def count_local_peaks(
    values,
    min_prominence_px=1.5,
    min_relative_prominence=0.25,
):
    """
    Counts local peaks in a 1D width profile.

    A point is a peak if:
        value[i] > value[i-1]
        value[i] > value[i+1]
        and prominence is meaningful.
    """

    values = np.asarray(values, dtype=np.float32)

    if len(values) < 3:
        return 0

    median_value = np.median(values)
    peak_count = 0

    for i in range(1, len(values) - 1):
        current = values[i]
        left = values[i - 1]
        right = values[i + 1]

        if current <= left or current <= right:
            continue

        local_base = max(left, right)
        prominence = current - local_base

        prominence_threshold = max(
            min_prominence_px,
            min_relative_prominence * max(median_value, EPS),
        )

        if prominence >= prominence_threshold:
            peak_count += 1

    return int(peak_count)


def local_spike_percent(
    values,
    window_size=7,
    min_abs_spike_px=1.5,
    min_relative_spike=0.30,
):
    """
    Detects local spike rows by comparing each row to rolling median.

    A row is a spike if:
        width - local_median >= threshold
    """

    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return 0.0

    local_med = rolling_median(values, window_size=window_size)

    median_global = np.median(values)

    threshold = np.maximum(
        min_abs_spike_px,
        min_relative_spike * np.maximum(local_med, median_global),
    )

    spikes = (values - local_med) >= threshold

    return float(np.mean(spikes))


def count_sustained_runs(flags, target_flag, min_run_length):
    """
    Counts rows that belong to sustained runs.

    flags:
         1 = left dominant
        -1 = right dominant
         0 = no meaningful dominance
    """

    flags = np.asarray(flags)

    sustained_mask = np.zeros_like(flags, dtype=bool)

    start = None

    for i, value in enumerate(flags):
        if value == target_flag:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= min_run_length:
                    sustained_mask[start:i] = True
                start = None

    if start is not None:
        length = len(flags) - start
        if length >= min_run_length:
            sustained_mask[start:] = True

    return sustained_mask


# ============================================================
# V3 side-dominance logic
# ============================================================

def compute_v3_side_dominance_features(left_widths, right_widths):
    """
    Improved side dominance logic.

    Main V3 improvement:
    A row is side-dominant only when the difference is clinically meaningful,
    not just slightly larger on one side.

    This avoids wrongly giving uniform cases high side-dominance values.
    """

    left = np.asarray(left_widths, dtype=np.float32)
    right = np.asarray(right_widths, dtype=np.float32)

    n = len(left)

    if n == 0:
        return {
            "side_dominant_length_ratio": 0.0,
            "left_dominant_percent": 0.0,
            "right_dominant_percent": 0.0,
            "no_dominant_percent": 1.0,
            "side_switch_count": 0,
            "side_dominance_score": 0.0,
            "mean_side_dominance_strength": 0.0,
            "max_side_dominance_strength": 0.0,
        }

    total = left + right
    diff = left - right
    abs_diff = np.abs(diff)

    median_total = np.median(total)

    # Dynamic meaningful-difference threshold
    # Prevents tiny left-right differences from being counted as dominance.
    min_abs_difference_px = max(1.5, 0.18 * median_total)

    # Relative dominance threshold:
    # one side should be meaningfully wider than the other.
    smaller_side = np.minimum(left, right)
    relative_requirement = 0.30 * np.maximum(smaller_side, 1.0)

    dominance_threshold = np.maximum(
        min_abs_difference_px,
        relative_requirement,
    )

    meaningful = abs_diff >= dominance_threshold

    flags = np.zeros(n, dtype=np.int32)
    flags[(diff > 0) & meaningful] = 1       # left dominant
    flags[(diff < 0) & meaningful] = -1      # right dominant

    # Sustained dominance should last for several rows.
    min_run_length = max(3, int(round(0.08 * n)))

    left_sustained = count_sustained_runs(
        flags=flags,
        target_flag=1,
        min_run_length=min_run_length,
    )

    right_sustained = count_sustained_runs(
        flags=flags,
        target_flag=-1,
        min_run_length=min_run_length,
    )

    sustained_any = left_sustained | right_sustained

    left_dominant_percent = float(np.mean(left_sustained))
    right_dominant_percent = float(np.mean(right_sustained))
    no_dominant_percent = float(1.0 - np.mean(sustained_any))

    side_dominant_length_ratio = float(
        max(left_dominant_percent, right_dominant_percent)
    )

    # Count side switches only among sustained meaningful flags
    sustained_flags = np.zeros(n, dtype=np.int32)
    sustained_flags[left_sustained] = 1
    sustained_flags[right_sustained] = -1

    valid_flags = sustained_flags[sustained_flags != 0]

    if len(valid_flags) <= 1:
        side_switch_count = 0
    else:
        side_switch_count = int(np.sum(valid_flags[1:] != valid_flags[:-1]))

    # Strength of dominance, normalized by total width
    dominance_strength = abs_diff / (total + EPS)

    if np.any(sustained_any):
        mean_strength = float(np.mean(dominance_strength[sustained_any]))
        max_strength = float(np.max(dominance_strength[sustained_any]))
    else:
        mean_strength = 0.0
        max_strength = 0.0

    # Final side dominance score
    # High only when dominance is sustained and strong.
    length_component = clip01(side_dominant_length_ratio / 0.65)
    strength_component = clip01(mean_strength / 0.45)

    # Penalize frequent switching because side-dominant widening should usually
    # stay on one side, not jump left/right repeatedly.
    switch_penalty = clip01(side_switch_count / 4.0)

    side_dominance_score = clip01(
        0.65 * length_component
        + 0.35 * strength_component
        - 0.20 * switch_penalty
    )

    return {
        "side_dominant_length_ratio": float(side_dominant_length_ratio),
        "left_dominant_percent": float(left_dominant_percent),
        "right_dominant_percent": float(right_dominant_percent),
        "no_dominant_percent": float(no_dominant_percent),
        "side_switch_count": int(side_switch_count),
        "side_dominance_score": float(side_dominance_score),
        "mean_side_dominance_strength": float(mean_strength),
        "max_side_dominance_strength": float(max_strength),
    }


# ============================================================
# V3 uniformity logic
# ============================================================

def compute_v3_uniformity_score(
    left_widths,
    right_widths,
    localized_spike_score,
    side_dominance_score,
):
    """
    Improved uniformity score.

    V2 weakness:
        A side-dominant case can still be smooth, so smoothness alone
        can wrongly create high uniformity.

    V3 definition:
        Uniform widening should be:
            smooth
            balanced left/right
            low local spikes
            low side dominance
            high left/right correlation
    """

    left = np.asarray(left_widths, dtype=np.float32)
    right = np.asarray(right_widths, dtype=np.float32)

    if len(left) == 0:
        return 0.0

    total = left + right
    asymmetry = np.abs(left - right)

    mean_total = np.mean(total)
    median_total = np.median(total)

    width_cv = np.std(total) / (mean_total + EPS)
    asymmetry_ratio = np.mean(asymmetry) / (mean_total + EPS)

    left_right_correlation = safe_corr(left, right)

    total_max_to_median_ratio = np.max(total) / (median_total + EPS)

    # Normalize penalties
    width_variation_penalty = clip01(width_cv / 0.45)

    asymmetry_penalty = clip01(asymmetry_ratio / 0.35)

    local_spike_penalty = clip01(localized_spike_score)

    side_dominance_penalty = clip01(side_dominance_score)

    # Correlation should be high for uniform widening.
    # If correlation is low or negative, penalty increases.
    correlation_penalty = clip01((1.0 - left_right_correlation) / 2.0)

    max_ratio_penalty = clip01((total_max_to_median_ratio - 1.0) / 1.5)

    total_penalty = (
        0.20 * width_variation_penalty
        + 0.22 * asymmetry_penalty
        + 0.25 * local_spike_penalty
        + 0.23 * side_dominance_penalty
        + 0.07 * correlation_penalty
        + 0.03 * max_ratio_penalty
    )

    uniformity_score = clip01(1.0 - total_penalty)

    return float(uniformity_score)


# ============================================================
# Main feature computation
# ============================================================

def compute_profile_features(profile_df):
    """
    Computes V3 PDL pattern features from row-wise width profile.
    """

    if profile_df is None or len(profile_df) < 10:
        return {
            "feature_error": "too_few_valid_profile_rows"
        }

    # Sort by y so coronal/middle/apical regions are consistent
    profile_df = profile_df.sort_values("y").reset_index(drop=True)

    left = profile_df["left_width_px"].values.astype(np.float32)
    right = profile_df["right_width_px"].values.astype(np.float32)

    # Mild smoothing to reduce row-level segmentation noise
    left_smooth = rolling_median(left, window_size=5)
    right_smooth = rolling_median(right, window_size=5)

    total = left_smooth + right_smooth
    asymmetry = np.abs(left_smooth - right_smooth)

    n = len(profile_df)

    mean_left = safe_mean(left_smooth)
    mean_right = safe_mean(right_smooth)
    mean_total = safe_mean(total)

    median_total = safe_percentile(total, 50)
    p90_total = safe_percentile(total, 90)
    p95_total = safe_percentile(total, 95)
    max_total = float(np.max(total)) if len(total) > 0 else 0.0

    width_std = safe_std(total)
    width_cv = width_std / (mean_total + EPS)

    mean_asymmetry = safe_mean(asymmetry)
    max_asymmetry = float(np.max(asymmetry)) if len(asymmetry) > 0 else 0.0

    asymmetry_ratio = mean_asymmetry / (mean_total + EPS)

    side_difference_px = abs(mean_left - mean_right)

    side_dominance_ratio = (
        max(mean_left, mean_right) / (min(mean_left, mean_right) + EPS)
    )

    # Basic side consistency from mean side dominance
    if mean_left > mean_right:
        dominant_side = "left"
    elif mean_right > mean_left:
        dominant_side = "right"
    else:
        dominant_side = "none"

    side_consistency = side_difference_px / (mean_total + EPS)

    max_to_mean_width_ratio = max_total / (mean_total + EPS)
    p95_to_median_width_ratio = p95_total / (median_total + EPS)

    # Regional features
    one_third = max(1, n // 3)

    coronal_total = total[:one_third]
    middle_total = total[one_third:2 * one_third]
    apical_total = total[2 * one_third:]

    coronal_asym = asymmetry[:one_third]
    middle_asym = asymmetry[one_third:2 * one_third]
    apical_asym = asymmetry[2 * one_third:]

    coronal_mean_width = safe_mean(coronal_total)
    middle_mean_width = safe_mean(middle_total)
    apical_mean_width = safe_mean(apical_total)

    apical_to_coronal_width_ratio = (
        apical_mean_width / (coronal_mean_width + EPS)
    )

    coronal_asymmetry_ratio = safe_mean(coronal_asym) / (
        coronal_mean_width + EPS
    )

    middle_asymmetry_ratio = safe_mean(middle_asym) / (
        middle_mean_width + EPS
    )

    apical_asymmetry_ratio = safe_mean(apical_asym) / (
        apical_mean_width + EPS
    )

    # Local spikes and peaks
    left_local_spike_percent = local_spike_percent(left_smooth)
    right_local_spike_percent = local_spike_percent(right_smooth)
    total_local_spike_percent = local_spike_percent(total)
    asymmetry_local_spike_percent = local_spike_percent(asymmetry)

    left_local_peak_count = count_local_peaks(left_smooth)
    right_local_peak_count = count_local_peaks(right_smooth)
    total_local_peak_count = count_local_peaks(total)
    asymmetry_local_peak_count = count_local_peaks(asymmetry)

    localized_peak_count = int(
        total_local_peak_count
        + asymmetry_local_peak_count
    )

    left_width_std = safe_std(left_smooth)
    right_width_std = safe_std(right_smooth)

    left_width_cv = left_width_std / (mean_left + EPS)
    right_width_cv = right_width_std / (mean_right + EPS)

    left_median = safe_percentile(left_smooth, 50)
    right_median = safe_percentile(right_smooth, 50)

    left_max_to_median_ratio = (
        float(np.max(left_smooth)) / (left_median + EPS)
        if len(left_smooth) > 0
        else 0.0
    )

    right_max_to_median_ratio = (
        float(np.max(right_smooth)) / (right_median + EPS)
        if len(right_smooth) > 0
        else 0.0
    )

    total_max_to_median_ratio = max_total / (median_total + EPS)

    left_right_correlation = safe_corr(left_smooth, right_smooth)

    # Localized widening percent
    localized_widening_threshold = median_total * 1.35
    localized_widening_percent = float(
        np.mean(total > localized_widening_threshold)
    )

    localized_asymmetry_threshold = (
        safe_percentile(asymmetry, 75)
        + 0.5 * safe_std(asymmetry)
    )

    localized_asymmetry_percent = float(
        np.mean(asymmetry > localized_asymmetry_threshold)
    )

    # Irregularity score components
    spike_component = clip01(total_local_spike_percent / 0.35)
    asym_spike_component = clip01(asymmetry_local_spike_percent / 0.35)
    peak_component = clip01(localized_peak_count / 6.0)
    max_ratio_component = clip01((total_max_to_median_ratio - 1.0) / 1.5)

    localized_spike_score = clip01(
        0.35 * spike_component
        + 0.25 * asym_spike_component
        + 0.25 * peak_component
        + 0.15 * max_ratio_component
    )

    # Irregularity index
    irregularity_index = clip01(
        0.40 * localized_spike_score
        + 0.25 * clip01(width_cv / 0.50)
        + 0.20 * clip01(total_max_to_median_ratio / 3.0)
        + 0.15 * clip01(localized_asymmetry_percent / 0.40)
    )

    # V3 side dominance
    side_features = compute_v3_side_dominance_features(
        left_widths=left_smooth,
        right_widths=right_smooth,
    )

    # V3 uniformity score
    uniformity_score = compute_v3_uniformity_score(
        left_widths=left_smooth,
        right_widths=right_smooth,
        localized_spike_score=localized_spike_score,
        side_dominance_score=side_features["side_dominance_score"],
    )

    features = {
        "feature_error": "",

        # Profile size
        "num_profile_rows": int(n),

        # Basic width features
        "mean_left_width_px": float(mean_left),
        "mean_right_width_px": float(mean_right),
        "mean_total_width_px": float(mean_total),
        "max_total_width_px": float(max_total),
        "p90_total_width_px": float(p90_total),
        "p95_total_width_px": float(p95_total),
        "median_total_width_px": float(median_total),
        "width_std_px": float(width_std),
        "width_cv": float(width_cv),

        # Asymmetry features
        "mean_asymmetry_px": float(mean_asymmetry),
        "max_asymmetry_px": float(max_asymmetry),
        "asymmetry_ratio": float(asymmetry_ratio),
        "side_difference_px": float(side_difference_px),
        "side_dominance_ratio": float(side_dominance_ratio),
        "side_consistency": float(side_consistency),
        "dominant_side": dominant_side,

        # Localized widening
        "localized_widening_percent": float(localized_widening_percent),
        "localized_asymmetry_percent": float(localized_asymmetry_percent),
        "max_to_mean_width_ratio": float(max_to_mean_width_ratio),
        "p95_to_median_width_ratio": float(p95_to_median_width_ratio),

        # Regional width
        "coronal_mean_width_px": float(coronal_mean_width),
        "middle_mean_width_px": float(middle_mean_width),
        "apical_mean_width_px": float(apical_mean_width),
        "apical_to_coronal_width_ratio": float(apical_to_coronal_width_ratio),

        # Irregularity
        "irregularity_index": float(irregularity_index),

        # Left/right variation
        "left_width_std_px": float(left_width_std),
        "right_width_std_px": float(right_width_std),
        "left_width_cv": float(left_width_cv),
        "right_width_cv": float(right_width_cv),

        # Max-to-median features
        "left_max_to_median_ratio": float(left_max_to_median_ratio),
        "right_max_to_median_ratio": float(right_max_to_median_ratio),
        "total_max_to_median_ratio": float(total_max_to_median_ratio),

        # Local spikes
        "left_local_spike_percent": float(left_local_spike_percent),
        "right_local_spike_percent": float(right_local_spike_percent),
        "total_local_spike_percent": float(total_local_spike_percent),
        "asymmetry_local_spike_percent": float(asymmetry_local_spike_percent),

        # Local peaks
        "left_local_peak_count": int(left_local_peak_count),
        "right_local_peak_count": int(right_local_peak_count),
        "total_local_peak_count": int(total_local_peak_count),
        "asymmetry_local_peak_count": int(asymmetry_local_peak_count),
        "localized_peak_count": int(localized_peak_count),

        # V3 improved side dominance
        "side_dominant_length_ratio": float(
            side_features["side_dominant_length_ratio"]
        ),
        "side_switch_count": int(side_features["side_switch_count"]),
        "left_dominant_percent": float(side_features["left_dominant_percent"]),
        "right_dominant_percent": float(side_features["right_dominant_percent"]),
        "no_dominant_percent": float(side_features["no_dominant_percent"]),
        "mean_side_dominance_strength": float(
            side_features["mean_side_dominance_strength"]
        ),
        "max_side_dominance_strength": float(
            side_features["max_side_dominance_strength"]
        ),

        # Correlation and regional asymmetry
        "left_right_correlation": float(left_right_correlation),
        "coronal_asymmetry_ratio": float(coronal_asymmetry_ratio),
        "middle_asymmetry_ratio": float(middle_asymmetry_ratio),
        "apical_asymmetry_ratio": float(apical_asymmetry_ratio),

        # Final V3 clinical scores
        "localized_spike_score": float(localized_spike_score),
        "side_dominance_score": float(side_features["side_dominance_score"]),
        "uniformity_score": float(uniformity_score),
    }

    return features


# ============================================================
# Debug image
# ============================================================

def draw_profile_plot(panel, values, x0, y0, width, height, label):
    """
    Draws simple profile graph using OpenCV.
    """

    values = np.asarray(values, dtype=np.float32)

    cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (80, 80, 80), 1)
    cv2.putText(
        panel,
        label,
        (x0, y0 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if len(values) < 2:
        return panel

    min_v = float(np.min(values))
    max_v = float(np.max(values))

    if abs(max_v - min_v) < EPS:
        max_v = min_v + 1.0

    points = []

    for i, value in enumerate(values):
        px = x0 + int(i / max(1, len(values) - 1) * width)
        py = y0 + height - int((value - min_v) / (max_v - min_v) * height)
        points.append((px, py))

    for i in range(1, len(points)):
        cv2.line(panel, points[i - 1], points[i], (255, 255, 255), 1)

    return panel


def save_debug_image(
    image,
    root_mask,
    pdl_mask,
    profile_df,
    features,
    output_path,
):
    """
    Saves debug visualization.
    """

    if image is None:
        return

    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    root_binary = root_mask > 0
    pdl_binary = pdl_mask > 0

    overlay = vis.copy()

    overlay[root_binary] = (0, 255, 0)
    overlay[pdl_binary] = (0, 0, 255)

    vis = cv2.addWeighted(vis, 0.70, overlay, 0.30, 0)

    # Draw root and PDL contours
    root_contours, _ = cv2.findContours(
        root_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    pdl_contours, _ = cv2.findContours(
        pdl_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(vis, root_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(vis, pdl_contours, -1, (0, 0, 255), 1)

    h, w = vis.shape[:2]
    panel_w = 450

    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = vis

    panel = canvas[:, w:]

    cv2.putText(
        panel,
        "V3 PDL Feature Debug",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    text_rows = [
        f"Rows: {features.get('num_profile_rows', 0)}",
        f"Uniformity: {features.get('uniformity_score', 0):.3f}",
        f"Side dominance: {features.get('side_dominance_score', 0):.3f}",
        f"Localized spike: {features.get('localized_spike_score', 0):.3f}",
        f"Irregularity: {features.get('irregularity_index', 0):.3f}",
        f"Side length ratio: {features.get('side_dominant_length_ratio', 0):.3f}",
    ]

    y_text = 70

    for text in text_rows:
        cv2.putText(
            panel,
            text,
            (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y_text += 25

    if profile_df is not None and len(profile_df) > 0:
        left = rolling_median(profile_df["left_width_px"].values, 5)
        right = rolling_median(profile_df["right_width_px"].values, 5)
        total = left + right
        asym = np.abs(left - right)

        draw_profile_plot(panel, left, 20, 240, 400, 70, "Left width")
        draw_profile_plot(panel, right, 20, 340, 400, 70, "Right width")
        draw_profile_plot(panel, total, 20, 440, 400, 70, "Total width")
        draw_profile_plot(panel, asym, 20, 540, 400, 70, "Asymmetry")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


# ============================================================
# Main extraction function
# ============================================================

def extract_pdl_pattern_features_v3(
    image_path,
    root_mask_path,
    pdl_mask_path,
    debug_path=None,
):
    """
    Extracts V3 PDL pattern features from one image.
    """

    image_name = os.path.basename(image_path)

    base_features = {
        "image_name": image_name,
    }

    image = read_gray(image_path)

    if image is None:
        base_features["feature_error"] = "image_read_failed"
        return base_features

    root_mask = cv2.imread(root_mask_path, cv2.IMREAD_GRAYSCALE)
    pdl_mask = cv2.imread(pdl_mask_path, cv2.IMREAD_GRAYSCALE)

    if root_mask is None:
        base_features["feature_error"] = "root_mask_read_failed"
        return base_features

    if pdl_mask is None:
        base_features["feature_error"] = "pdl_mask_read_failed"
        return base_features

    root_mask = resize_mask_to_image(root_mask, image.shape)
    pdl_mask = resize_mask_to_image(pdl_mask, image.shape)

    root_mask = clean_root_mask(root_mask)
    pdl_mask = clean_pdl_mask(pdl_mask)

    profile_df = extract_width_profiles(
        root_mask=root_mask,
        pdl_mask=pdl_mask,
    )

    if profile_df is None or len(profile_df) < 10:
        base_features["feature_error"] = "too_few_valid_profile_rows"
        base_features["num_profile_rows"] = 0 if profile_df is None else len(profile_df)
        return base_features

    computed_features = compute_profile_features(profile_df)

    base_features.update(computed_features)

    if debug_path is not None:
        save_debug_image(
            image=image,
            root_mask=root_mask,
            pdl_mask=pdl_mask,
            profile_df=profile_df,
            features=base_features,
            output_path=debug_path,
        )

    return base_features


def save_features_csv(features_list, output_csv_path):
    """
    Saves extracted features to CSV.
    """

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False)

    return df