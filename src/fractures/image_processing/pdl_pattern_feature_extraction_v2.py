import os
import cv2
import numpy as np
import pandas as pd


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


def binary_mask(mask):
    mask = ensure_uint8_gray(mask)
    if mask is None:
        return None

    return (mask > 0).astype(np.uint8)


def safe_imwrite(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def safe_divide(a, b, eps=1e-6):
    return float(a) / float(b + eps)


def nan_array_stats(values):
    values = np.asarray(values, dtype=np.float32)
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "cv": 0.0,
        }

    mean = float(np.mean(valid))
    std = float(np.std(valid))
    median = float(np.median(valid))

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p75": float(np.percentile(valid, 75)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "cv": safe_divide(std, mean),
    }


def interpolate_nan(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    x = np.arange(len(values))
    valid = ~np.isnan(values)

    if valid.sum() == 0:
        return np.zeros_like(values, dtype=np.float32)

    if valid.sum() == 1:
        return np.full_like(values, values[valid][0], dtype=np.float32)

    return np.interp(x, x[valid], values[valid]).astype(np.float32)


def smooth_1d(values, ksize=9):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    if ksize % 2 == 0:
        ksize += 1

    if len(values) < ksize:
        ksize = max(3, len(values) if len(values) % 2 == 1 else len(values) - 1)

    if ksize < 3:
        return values

    return cv2.GaussianBlur(values.reshape(-1, 1), (1, ksize), 0).flatten()


def count_true_runs(mask):
    mask = np.asarray(mask).astype(bool)

    if len(mask) == 0:
        return 0

    count = 0
    in_run = False

    for value in mask:
        if value and not in_run:
            count += 1
            in_run = True
        elif not value:
            in_run = False

    return count


def longest_true_run_ratio(mask):
    mask = np.asarray(mask).astype(bool)

    if len(mask) == 0:
        return 0.0

    longest = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return safe_divide(longest, len(mask))


def side_switch_count(side_profile):
    """
    side_profile:
        -1 = left dominant
         0 = no dominant side
         1 = right dominant
    """

    side_profile = np.asarray(side_profile, dtype=np.int32)
    side_profile = side_profile[side_profile != 0]

    if len(side_profile) <= 1:
        return 0

    return int(np.sum(side_profile[1:] != side_profile[:-1]))


def profile_correlation(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    valid = (~np.isnan(a)) & (~np.isnan(b))

    if valid.sum() < 3:
        return 0.0

    a = a[valid]
    b = b[valid]

    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def split_profile_thirds(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values, values, values

    n = len(values)
    one = n // 3
    two = 2 * n // 3

    coronal = values[:one]
    middle = values[one:two]
    apical = values[two:]

    return coronal, middle, apical


def local_spike_features(widths, local_ratio_threshold=1.35, absolute_margin_px=2.0):
    """
    Detect localized widening spikes.

    A spike means:
    current width is clearly higher than smoothed local baseline.
    """

    widths = np.asarray(widths, dtype=np.float32)

    if len(widths) == 0:
        return {
            "spike_percent": 0.0,
            "spike_count": 0,
            "max_spike_ratio": 0.0,
            "mean_spike_ratio": 0.0,
        }

    filled = interpolate_nan(widths)
    smooth = smooth_1d(filled, ksize=11)

    ratio = (filled + 1e-6) / (smooth + 1e-6)
    diff = filled - smooth

    spike_mask = (ratio >= local_ratio_threshold) & (diff >= absolute_margin_px)

    spike_ratios = ratio[spike_mask]

    return {
        "spike_percent": float(np.mean(spike_mask)),
        "spike_count": int(count_true_runs(spike_mask)),
        "max_spike_ratio": float(np.max(spike_ratios)) if len(spike_ratios) else 0.0,
        "mean_spike_ratio": float(np.mean(spike_ratios)) if len(spike_ratios) else 0.0,
    }


# ============================================================
# Width-profile extraction
# ============================================================

def scan_pdl_width_from_root_boundary(
    pdl_mask,
    y,
    root_left,
    root_right,
    max_scan_px=45,
    gap_tolerance=2
):
    h, w = pdl_mask.shape

    # -----------------------------
    # Left side
    # -----------------------------
    left_width = 0
    gap = 0
    seen_pdl = False

    x_start = max(0, root_left - 1)
    x_end = max(0, root_left - max_scan_px)

    for x in range(x_start, x_end - 1, -1):
        if pdl_mask[y, x] > 0:
            seen_pdl = True
            gap = 0
            left_width = root_left - x
        else:
            if seen_pdl:
                gap += 1
                if gap > gap_tolerance:
                    break

    # -----------------------------
    # Right side
    # -----------------------------
    right_width = 0
    gap = 0
    seen_pdl = False

    x_start = min(w - 1, root_right + 1)
    x_end = min(w - 1, root_right + max_scan_px)

    for x in range(x_start, x_end + 1):
        if pdl_mask[y, x] > 0:
            seen_pdl = True
            gap = 0
            right_width = x - root_right
        else:
            if seen_pdl:
                gap += 1
                if gap > gap_tolerance:
                    break

    return float(left_width), float(right_width)


def extract_row_width_profiles(root_mask, pdl_mask):
    """
    Extract row-wise PDL width profile from root and PDL masks.

    Output:
        y_positions
        left_widths
        right_widths
        total_widths
        asymmetry
    """

    root_mask = binary_mask(root_mask)
    pdl_mask = binary_mask(pdl_mask)

    h, w = root_mask.shape

    y_positions = []
    left_widths = []
    right_widths = []

    ys = np.where(root_mask > 0)[0]

    if len(ys) == 0:
        return {
            "y_positions": np.array([], dtype=np.int32),
            "left_widths": np.array([], dtype=np.float32),
            "right_widths": np.array([], dtype=np.float32),
            "total_widths": np.array([], dtype=np.float32),
            "asymmetry": np.array([], dtype=np.float32),
        }

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))

    for y in range(y_min, y_max + 1):
        root_xs = np.where(root_mask[y, :] > 0)[0]

        if len(root_xs) < 3:
            continue

        root_left = int(np.min(root_xs))
        root_right = int(np.max(root_xs))

        left_w, right_w = scan_pdl_width_from_root_boundary(
            pdl_mask=pdl_mask,
            y=y,
            root_left=root_left,
            root_right=root_right
        )

        # Keep only rows where at least one side has measurable PDL
        if left_w <= 0 and right_w <= 0:
            continue

        y_positions.append(y)
        left_widths.append(left_w if left_w > 0 else np.nan)
        right_widths.append(right_w if right_w > 0 else np.nan)

    y_positions = np.asarray(y_positions, dtype=np.int32)
    left_widths = np.asarray(left_widths, dtype=np.float32)
    right_widths = np.asarray(right_widths, dtype=np.float32)

    left_filled = interpolate_nan(left_widths)
    right_filled = interpolate_nan(right_widths)

    total_widths = left_filled + right_filled
    asymmetry = np.abs(left_filled - right_filled)

    return {
        "y_positions": y_positions,
        "left_widths": left_filled,
        "right_widths": right_filled,
        "total_widths": total_widths,
        "asymmetry": asymmetry,
    }


# ============================================================
# Feature extraction
# ============================================================

def extract_pdl_pattern_features_v2(
    image_path,
    root_mask_path,
    pdl_mask_path,
    debug_path=None
):
    image_name = os.path.basename(image_path)

    image = cv2.imread(image_path)
    root_mask = cv2.imread(root_mask_path, cv2.IMREAD_GRAYSCALE)
    pdl_mask = cv2.imread(pdl_mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if root_mask is None:
        raise ValueError(f"Could not read root mask: {root_mask_path}")

    if pdl_mask is None:
        raise ValueError(f"Could not read PDL mask: {pdl_mask_path}")

    root_mask = binary_mask(root_mask)
    pdl_mask = binary_mask(pdl_mask)

    profile = extract_row_width_profiles(root_mask, pdl_mask)

    y_positions = profile["y_positions"]
    left_widths = profile["left_widths"]
    right_widths = profile["right_widths"]
    total_widths = profile["total_widths"]
    asymmetry = profile["asymmetry"]

    left_stats = nan_array_stats(left_widths)
    right_stats = nan_array_stats(right_widths)
    total_stats = nan_array_stats(total_widths)
    asym_stats = nan_array_stats(asymmetry)

    n_rows = len(y_positions)

    if n_rows == 0:
        return {
            "image_name": image_name,
            "num_profile_rows": 0,
            "feature_error": "no_valid_profile_rows"
        }

    # --------------------------------------------------------
    # Side dominance profile
    # --------------------------------------------------------

    dominance_margin_px = 2.0

    side_profile = np.zeros(n_rows, dtype=np.int32)
    side_profile[left_widths - right_widths >= dominance_margin_px] = -1
    side_profile[right_widths - left_widths >= dominance_margin_px] = 1

    left_dominant_percent = float(np.mean(side_profile == -1))
    right_dominant_percent = float(np.mean(side_profile == 1))
    no_dominant_percent = float(np.mean(side_profile == 0))

    side_dominant_length_ratio = max(
        longest_true_run_ratio(side_profile == -1),
        longest_true_run_ratio(side_profile == 1)
    )

    side_consistency = max(left_dominant_percent, right_dominant_percent)

    switch_count = side_switch_count(side_profile)

    if right_dominant_percent > left_dominant_percent:
        dominant_side = "right"
    elif left_dominant_percent > right_dominant_percent:
        dominant_side = "left"
    else:
        dominant_side = "none"

    # --------------------------------------------------------
    # Local spike features
    # --------------------------------------------------------

    left_spikes = local_spike_features(left_widths)
    right_spikes = local_spike_features(right_widths)
    total_spikes = local_spike_features(total_widths)
    asym_spikes = local_spike_features(asymmetry, local_ratio_threshold=1.30, absolute_margin_px=1.5)

    localized_spike_percent = max(
        left_spikes["spike_percent"],
        right_spikes["spike_percent"],
        total_spikes["spike_percent"],
        asym_spikes["spike_percent"]
    )

    localized_peak_count = (
        left_spikes["spike_count"]
        + right_spikes["spike_count"]
        + total_spikes["spike_count"]
        + asym_spikes["spike_count"]
    )

    # --------------------------------------------------------
    # Coronal / middle / apical region features
    # --------------------------------------------------------

    coronal_asym, middle_asym, apical_asym = split_profile_thirds(asymmetry)
    coronal_total, middle_total, apical_total = split_profile_thirds(total_widths)

    coronal_asym_mean = nan_array_stats(coronal_asym)["mean"]
    middle_asym_mean = nan_array_stats(middle_asym)["mean"]
    apical_asym_mean = nan_array_stats(apical_asym)["mean"]

    coronal_total_mean = nan_array_stats(coronal_total)["mean"]
    middle_total_mean = nan_array_stats(middle_total)["mean"]
    apical_total_mean = nan_array_stats(apical_total)["mean"]

    coronal_asymmetry_ratio = safe_divide(coronal_asym_mean, coronal_total_mean)
    middle_asymmetry_ratio = safe_divide(middle_asym_mean, middle_total_mean)
    apical_asymmetry_ratio = safe_divide(apical_asym_mean, apical_total_mean)

    # --------------------------------------------------------
    # Profile correlation
    # --------------------------------------------------------

    left_right_correlation = profile_correlation(left_widths, right_widths)

    # --------------------------------------------------------
    # Higher-level pattern scores
    # --------------------------------------------------------

    mean_total_width_px = total_stats["mean"]
    median_total_width_px = total_stats["median"]

    mean_asymmetry_px = asym_stats["mean"]

    asymmetry_ratio = safe_divide(mean_asymmetry_px, mean_total_width_px)

    side_difference_px = abs(left_stats["mean"] - right_stats["mean"])

    side_width_ratio = safe_divide(
        max(left_stats["mean"], right_stats["mean"]),
        min(left_stats["mean"], right_stats["mean"])
    )

    left_max_to_median_ratio = safe_divide(left_stats["max"], left_stats["median"])
    right_max_to_median_ratio = safe_divide(right_stats["max"], right_stats["median"])
    total_max_to_median_ratio = safe_divide(total_stats["max"], total_stats["median"])
    total_p95_to_median_ratio = safe_divide(total_stats["p95"], total_stats["median"])

    width_cv = total_stats["cv"]

    localized_spike_score = (
        0.30 * localized_spike_percent
        + 0.20 * min(1.0, localized_peak_count / max(1.0, n_rows / 10.0))
        + 0.20 * min(1.0, total_spikes["max_spike_ratio"] / 2.0)
        + 0.15 * min(1.0, asym_spikes["max_spike_ratio"] / 2.0)
        + 0.15 * min(1.0, total_max_to_median_ratio / 3.0)
    )

    side_dominance_score = (
        0.30 * side_consistency
        + 0.25 * side_dominant_length_ratio
        + 0.20 * min(1.0, side_width_ratio / 3.0)
        + 0.15 * min(1.0, side_difference_px / max(1.0, mean_total_width_px))
        + 0.10 * (1.0 - min(1.0, switch_count / max(1.0, n_rows / 5.0)))
    )

    irregularity_score = (
        0.25 * min(1.0, width_cv)
        + 0.25 * localized_spike_score
        + 0.20 * min(1.0, total_p95_to_median_ratio / 3.0)
        + 0.15 * min(1.0, asymmetry_ratio)
        + 0.15 * min(1.0, abs(left_right_correlation))
    )

    uniformity_score = (
        1.0
        - (
            0.35 * min(1.0, irregularity_score)
            + 0.30 * min(1.0, side_dominance_score)
            + 0.20 * min(1.0, width_cv)
            + 0.15 * min(1.0, localized_spike_percent)
        )
    )

    uniformity_score = float(np.clip(uniformity_score, 0.0, 1.0))

    features = {
        "image_name": image_name,
        "num_profile_rows": int(n_rows),

        # Existing-compatible features
        "mean_left_width_px": left_stats["mean"],
        "mean_right_width_px": right_stats["mean"],
        "mean_total_width_px": total_stats["mean"],
        "max_total_width_px": total_stats["max"],
        "p90_total_width_px": total_stats["p90"],
        "p95_total_width_px": total_stats["p95"],
        "median_total_width_px": total_stats["median"],
        "width_std_px": total_stats["std"],
        "width_cv": width_cv,

        "mean_asymmetry_px": asym_stats["mean"],
        "max_asymmetry_px": asym_stats["max"],
        "asymmetry_ratio": asymmetry_ratio,

        "side_difference_px": side_difference_px,
        "side_dominance_ratio": side_width_ratio,
        "side_consistency": side_consistency,
        "dominant_widening_side": dominant_side,

        "localized_widening_percent": localized_spike_percent,
        "localized_asymmetry_percent": asym_spikes["spike_percent"],
        "max_to_mean_width_ratio": safe_divide(total_stats["max"], total_stats["mean"]),
        "p95_to_median_width_ratio": total_p95_to_median_ratio,

        "coronal_mean_width_px": coronal_total_mean,
        "middle_mean_width_px": middle_total_mean,
        "apical_mean_width_px": apical_total_mean,
        "apical_to_coronal_width_ratio": safe_divide(apical_total_mean, coronal_total_mean),

        "irregularity_index": irregularity_score,

        # New V2 features
        "left_width_std_px": left_stats["std"],
        "right_width_std_px": right_stats["std"],
        "left_width_cv": left_stats["cv"],
        "right_width_cv": right_stats["cv"],

        "left_max_to_median_ratio": left_max_to_median_ratio,
        "right_max_to_median_ratio": right_max_to_median_ratio,
        "total_max_to_median_ratio": total_max_to_median_ratio,

        "left_local_spike_percent": left_spikes["spike_percent"],
        "right_local_spike_percent": right_spikes["spike_percent"],
        "total_local_spike_percent": total_spikes["spike_percent"],
        "asymmetry_local_spike_percent": asym_spikes["spike_percent"],

        "left_local_peak_count": left_spikes["spike_count"],
        "right_local_peak_count": right_spikes["spike_count"],
        "total_local_peak_count": total_spikes["spike_count"],
        "asymmetry_local_peak_count": asym_spikes["spike_count"],
        "localized_peak_count": localized_peak_count,

        "side_dominant_length_ratio": side_dominant_length_ratio,
        "side_switch_count": switch_count,
        "left_dominant_percent": left_dominant_percent,
        "right_dominant_percent": right_dominant_percent,
        "no_dominant_percent": no_dominant_percent,

        "left_right_correlation": left_right_correlation,

        "coronal_asymmetry_ratio": coronal_asymmetry_ratio,
        "middle_asymmetry_ratio": middle_asymmetry_ratio,
        "apical_asymmetry_ratio": apical_asymmetry_ratio,

        "localized_spike_score": localized_spike_score,
        "side_dominance_score": side_dominance_score,
        "uniformity_score": uniformity_score,

        "feature_error": ""
    }

    if debug_path is not None:
        create_debug_image(
            image=image,
            root_mask=root_mask,
            pdl_mask=pdl_mask,
            y_positions=y_positions,
            left_widths=left_widths,
            right_widths=right_widths,
            debug_path=debug_path
        )

    return features


# ============================================================
# Debug visualization
# ============================================================

def create_debug_image(
    image,
    root_mask,
    pdl_mask,
    y_positions,
    left_widths,
    right_widths,
    debug_path
):
    if len(image.shape) == 2:
        debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        debug = image.copy()

    root_mask = binary_mask(root_mask)
    pdl_mask = binary_mask(pdl_mask)

    overlay = debug.copy()

    # Root: green
    overlay[root_mask > 0] = (0, 255, 0)

    # PDL: red
    overlay[pdl_mask > 0] = (0, 0, 255)

    debug = cv2.addWeighted(debug, 0.65, overlay, 0.35, 0)

    for idx, y in enumerate(y_positions):
        if idx % 5 != 0:
            continue

        root_xs = np.where(root_mask[y, :] > 0)[0]
        if len(root_xs) == 0:
            continue

        root_left = int(np.min(root_xs))
        root_right = int(np.max(root_xs))

        left_w = int(round(left_widths[idx]))
        right_w = int(round(right_widths[idx]))

        if left_w > 0:
            cv2.line(
                debug,
                (root_left, int(y)),
                (max(0, root_left - left_w), int(y)),
                (255, 255, 0),
                1
            )

        if right_w > 0:
            cv2.line(
                debug,
                (root_right, int(y)),
                (min(debug.shape[1] - 1, root_right + right_w), int(y)),
                (255, 255, 0),
                1
            )

    safe_imwrite(debug_path, debug)


# ============================================================
# Dataset processing
# ============================================================

def get_stem(filename):
    return os.path.splitext(os.path.basename(filename))[0].lower()


def find_mask_file(mask_dir, image_name):
    image_stem = get_stem(image_name)

    if not os.path.exists(mask_dir):
        return None

    valid_ext = [".png", ".jpg", ".jpeg", ".bmp"]

    for filename in os.listdir(mask_dir):
        if os.path.splitext(filename)[1].lower() not in valid_ext:
            continue

        mask_stem = get_stem(filename)

        if mask_stem == image_stem:
            return os.path.join(mask_dir, filename)

        if image_stem in mask_stem:
            return os.path.join(mask_dir, filename)

        if mask_stem in image_stem:
            return os.path.join(mask_dir, filename)

    return None


def merge_with_dentist_labels(feature_csv_path, label_csv_path, output_csv_path):
    if not os.path.exists(label_csv_path):
        print("Dentist label CSV not found. Skipping merge:", label_csv_path)
        return

    features_df = pd.read_csv(feature_csv_path)
    labels_df = pd.read_csv(label_csv_path)

    possible_keys = [
        "image_name",
        "filename",
        "file_name",
        "image",
        "Image",
        "Image Name"
    ]

    feature_key = None
    label_key = None

    for col in possible_keys:
        if col in features_df.columns:
            feature_key = col
            break

    for col in possible_keys:
        if col in labels_df.columns:
            label_key = col
            break

    if feature_key is None:
        raise ValueError("Image-name column not found in feature CSV.")

    if label_key is None:
        raise ValueError("Image-name column not found in dentist label CSV.")

    features_df["_merge_stem"] = (
        features_df[feature_key]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
        .str.replace(".png", "", regex=False)
    )

    labels_df["_merge_stem"] = (
        labels_df[label_key]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
        .str.replace(".png", "", regex=False)
    )

    merged_df = features_df.merge(
        labels_df,
        on="_merge_stem",
        how="left",
        suffixes=("", "_dentist")
    )

    merged_df.to_csv(output_csv_path, index=False)

    print("Merged CSV saved:", output_csv_path)
    print("Rows:", len(merged_df))


def process_dataset_v2(
    image_dir,
    root_mask_dir,
    pdl_mask_dir,
    output_csv_path,
    debug_dir=None,
    dentist_label_csv_path=None,
    merged_output_csv_path=None
):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp"]

    rows = []

    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    image_files = sorted(image_files)

    print("Images found:", len(image_files))

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)

        root_mask_path = find_mask_file(root_mask_dir, image_name)
        pdl_mask_path = find_mask_file(pdl_mask_dir, image_name)

        if root_mask_path is None:
            print("Missing root mask:", image_name)
            rows.append({
                "image_name": image_name,
                "feature_error": "missing_root_mask"
            })
            continue

        if pdl_mask_path is None:
            print("Missing PDL mask:", image_name)
            rows.append({
                "image_name": image_name,
                "feature_error": "missing_pdl_mask"
            })
            continue

        debug_path = None
        if debug_dir is not None:
            debug_path = os.path.join(
                debug_dir,
                f"debug_v2_{get_stem(image_name)}.png"
            )

        try:
            features = extract_pdl_pattern_features_v2(
                image_path=image_path,
                root_mask_path=root_mask_path,
                pdl_mask_path=pdl_mask_path,
                debug_path=debug_path
            )

            rows.append(features)
            print("Processed:", image_name)

        except Exception as e:
            print("ERROR processing:", image_name, str(e))
            rows.append({
                "image_name": image_name,
                "feature_error": str(e)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    print("\nV2 feature CSV saved:")
    print(output_csv_path)

    if dentist_label_csv_path is not None and merged_output_csv_path is not None:
        merge_with_dentist_labels(
            feature_csv_path=output_csv_path,
            label_csv_path=dentist_label_csv_path,
            output_csv_path=merged_output_csv_path
        )

    return df