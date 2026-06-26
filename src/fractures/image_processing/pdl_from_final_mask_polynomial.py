import os
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


def normalize_01(arr, low=2, high=98):
    arr = arr.astype(np.float32)

    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    p_low = np.percentile(valid, low)
    p_high = np.percentile(valid, high)

    if p_high <= p_low:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - p_low) / (p_high - p_low)
    out = np.clip(out, 0.0, 1.0)

    return out.astype(np.float32)


def read_binary_mask(mask_path, target_shape=None):
    if mask_path is None or not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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


# ============================================================
# Preprocessing
# ============================================================

def preprocess_xray_for_root_pdl(img_uint8):
    """
    Mild X-ray enhancement.

    Strong CLAHE is avoided because trabecular bone can become
    too dark and can be confused with PDL.
    """

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
        clipLimit=1.15,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(norm)

    enhanced = cv2.bilateralFilter(
        enhanced,
        d=5,
        sigmaColor=18,
        sigmaSpace=7
    )

    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced


def local_mean_std(img_uint8, ksize=15):
    ksize = odd(ksize, 3)

    img = img_uint8.astype(np.float32) / 255.0

    mean = cv2.blur(img, (ksize, ksize))
    mean_sq = cv2.blur(img * img, (ksize, ksize))

    var = np.maximum(mean_sq - mean * mean, 0)
    std = np.sqrt(var)

    return mean, std


def build_edge_map(img_uint8):
    blurred = cv2.GaussianBlur(img_uint8, (3, 3), 0)

    med = np.median(blurred)

    lower = int(max(0, 0.60 * med))
    upper = int(min(255, 1.45 * med))

    canny = cv2.Canny(blurred, lower, upper)

    gx = cv2.Sobel(
        img_uint8.astype(np.float32),
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        img_uint8.astype(np.float32),
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    grad = cv2.magnitude(gx, gy)
    grad01 = normalize_01(grad, 5, 97)

    edge01 = np.maximum(canny.astype(np.float32) / 255.0, grad01)
    edge01 = cv2.GaussianBlur(edge01, (3, 3), 0)
    edge01 = normalize_01(edge01, 2, 98)

    return edge01, canny


# ============================================================
# Multi-scale dark feature extraction
# ============================================================

def build_multiscale_dark_features(enhanced, candidate_mask):
    """
    Build strong dark-region features for uneven/asymmetric PDL.

    Features:
        1. Raw darkness
        2. Local dark contrast
        3. Local z-score darkness
        4. Multi-scale black-hat dark-band response
        5. Laplacian detail response
        6. Sobel/Canny edge response
    """

    h, w = enhanced.shape

    img01 = normalize_01(enhanced, 2, 98)

    raw_darkness = 1.0 - img01

    mean_small, std_small = local_mean_std(
        enhanced,
        ksize=max(9, int(w * 0.045))
    )

    mean_large, std_large = local_mean_std(
        enhanced,
        ksize=max(15, int(w * 0.090))
    )

    local_dark_small = np.clip(mean_small - img01, 0, 1)
    local_dark_large = np.clip(mean_large - img01, 0, 1)

    z_dark = (mean_large - img01) / (std_large + 1e-4)
    z_dark = normalize_01(z_dark, 2, 98)

    blackhat_responses = []

    for k in [5, 9, 13, 17]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k)
        )

        blackhat = cv2.morphologyEx(
            enhanced,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        blackhat_responses.append(
            normalize_01(blackhat, 2, 98)
        )

    blackhat_multi = np.maximum.reduce(blackhat_responses)
    blackhat_multi[candidate_mask == 0] = 0.0

    lap = cv2.Laplacian(
        enhanced.astype(np.float32),
        cv2.CV_32F,
        ksize=3
    )

    lap_abs = normalize_01(np.abs(lap), 5, 98)

    edge01, canny = build_edge_map(enhanced)

    def normalize_inside_candidate(feature):
        out = feature.copy().astype(np.float32)

        vals = out[candidate_mask > 0]

        if vals.size > 20:
            p1 = np.percentile(vals, 2)
            p2 = np.percentile(vals, 98)

            if p2 > p1:
                out = np.clip((out - p1) / (p2 - p1), 0, 1)

        out[candidate_mask == 0] = 0.0

        return out.astype(np.float32)

    raw_darkness = normalize_inside_candidate(raw_darkness)
    local_dark_small = normalize_inside_candidate(local_dark_small)
    local_dark_large = normalize_inside_candidate(local_dark_large)
    z_dark = normalize_inside_candidate(z_dark)
    blackhat_multi = normalize_inside_candidate(blackhat_multi)
    lap_abs = normalize_inside_candidate(lap_abs)
    edge01 = normalize_inside_candidate(edge01)

    return {
        "img01": img01,
        "raw_darkness": raw_darkness,
        "local_dark_small": local_dark_small,
        "local_dark_large": local_dark_large,
        "z_dark": z_dark,
        "blackhat": blackhat_multi,
        "laplacian": lap_abs,
        "edge01": edge01,
        "canny": canny
    }


# ============================================================
# Final-mask geometry
# ============================================================

def get_mask_row_geometry(final_mask):
    h, w = final_mask.shape

    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)
    center = np.full(h, np.nan, dtype=np.float32)
    width = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        xs = np.where(final_mask[y] > 0)[0]

        if len(xs) > 0:
            left[y] = xs[0]
            right[y] = xs[-1]
            center[y] = (xs[0] + xs[-1]) / 2.0
            width[y] = xs[-1] - xs[0] + 1

    valid = ~np.isnan(center)

    if np.sum(valid) < 5:
        image_center = (w - 1) / 2.0

        center[:] = image_center
        width[:] = w * 0.5
        left[:] = image_center - width[0] / 2.0
        right[:] = image_center + width[0] / 2.0

        y_min = 0
        y_max = h - 1

        return left, right, center, width, y_min, y_max

    ys = np.arange(h)

    center = np.interp(ys, ys[valid], center[valid])
    width = np.interp(ys, ys[valid], width[valid])
    left = np.interp(ys, ys[valid], left[valid])
    right = np.interp(ys, ys[valid], right[valid])

    k = odd(h * 0.045, 9)

    center = cv2.GaussianBlur(
        center.reshape(-1, 1),
        (1, k),
        0
    ).flatten()

    width = cv2.GaussianBlur(
        width.reshape(-1, 1),
        (1, k),
        0
    ).flatten()

    left = cv2.GaussianBlur(
        left.reshape(-1, 1),
        (1, k),
        0
    ).flatten()

    right = cv2.GaussianBlur(
        right.reshape(-1, 1),
        (1, k),
        0
    ).flatten()

    valid_rows = np.where(valid)[0]

    y_min = int(valid_rows[0])
    y_max = int(valid_rows[-1])

    return left, right, center, width, y_min, y_max


# ============================================================
# Root likelihood from X-ray + final mask only
# ============================================================

def build_root_likelihood_from_final_mask(enhanced, final_mask):
    """
    Root is expected to be:
        - bright
        - relatively homogeneous
        - central inside final mask
    """

    h, w = enhanced.shape

    img01 = normalize_01(enhanced, 2, 98)

    mean, std = local_mean_std(
        enhanced,
        ksize=max(9, int(w * 0.065))
    )

    std01 = normalize_01(std, 5, 95)
    homogeneity = 1.0 - std01

    _, _, mask_center, mask_width, _, _ = get_mask_row_geometry(final_mask)

    xs_grid = np.tile(
        np.arange(w, dtype=np.float32).reshape(1, -1),
        (h, 1)
    )

    center_grid = np.tile(
        mask_center.reshape(-1, 1),
        (1, w)
    )

    width_grid = np.tile(
        mask_width.reshape(-1, 1),
        (1, w)
    )

    sigma = np.maximum(6.0, width_grid * 0.30)

    central_prior = np.exp(
        -0.5 * ((xs_grid - center_grid) / sigma) ** 2
    ).astype(np.float32)

    central_prior[final_mask == 0] = 0.0

    ys, xs = np.where(final_mask > 0)

    root_cluster = np.zeros((h, w), dtype=np.float32)

    if len(xs) >= max(50, int(h * w * 0.005)):
        features = np.stack(
            [
                img01[ys, xs],
                mean[ys, xs],
                homogeneity[ys, xs],
                central_prior[ys, xs]
            ],
            axis=1
        ).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            80,
            0.01
        )

        try:
            _, labels, centers = cv2.kmeans(
                features,
                4,
                None,
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS
            )

            labels = labels.flatten()

            root_scores = (
                0.48 * centers[:, 0]
                + 0.22 * centers[:, 1]
                + 0.20 * centers[:, 2]
                + 0.10 * centers[:, 3]
            )

            root_label = int(np.argmax(root_scores))

            selected = labels == root_label
            root_cluster[ys[selected], xs[selected]] = 1.0

        except Exception:
            root_cluster[:, :] = 0.0

    root_like = (
        0.36 * img01
        + 0.20 * mean
        + 0.20 * homogeneity
        + 0.14 * root_cluster
        + 0.10 * central_prior
    )

    root_like[final_mask == 0] = 0.0

    root_like = cv2.GaussianBlur(root_like, (5, 5), 0)
    root_like = normalize_01(root_like, 2, 98)
    root_like[final_mask == 0] = 0.0

    return root_like, img01, homogeneity, central_prior, root_cluster


def keep_largest_central_component(mask, final_mask):
    h, w = mask.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        8
    )

    if num_labels <= 1:
        return mask

    _, _, mask_center, _, _, _ = get_mask_row_geometry(final_mask)

    image_center = np.nanmedian(mask_center)

    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < h * w * 0.002:
            continue

        cx = centroids[label][0]

        central_score = 1.0 - min(
            1.0,
            abs(cx - image_center) / max(w * 0.5, 1)
        )

        score = area * (0.40 + central_score)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return mask

    out = ((labels == best_label).astype(np.uint8) * 255)

    return out


def initial_root_mask_from_likelihood(root_like, final_mask, root_percentile=58):
    inside_values = root_like[final_mask > 0]

    if inside_values.size < 20:
        threshold_value = 0.55
    else:
        p_value = np.percentile(inside_values, root_percentile)

        tmp = (inside_values * 255).astype(np.uint8).reshape(-1, 1)

        otsu_value, _ = cv2.threshold(
            tmp,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        otsu_value = otsu_value / 255.0

        threshold_value = max(p_value, otsu_value * 0.92)
        threshold_value = min(threshold_value, 0.82)

    root_mask = (
        (root_like >= threshold_value)
        & (final_mask > 0)
    ).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    root_mask = cv2.morphologyEx(root_mask, cv2.MORPH_CLOSE, kernel_close)
    root_mask = cv2.morphologyEx(root_mask, cv2.MORPH_OPEN, kernel_open)

    root_mask = keep_largest_central_component(
        root_mask,
        final_mask=final_mask
    )

    return root_mask, threshold_value


# ============================================================
# Root end / apex from bright root region
# ============================================================

def extract_raw_root_boundaries(root_mask, final_mask, min_row_width_px=8):
    h, w = root_mask.shape

    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)

    mask_left, mask_right, _, _, y_min, y_max = get_mask_row_geometry(final_mask)

    for y in range(y_min, y_max + 1):
        xs = np.where(root_mask[y] > 0)[0]

        if len(xs) < min_row_width_px:
            continue

        lx = xs[0]
        rx = xs[-1]

        final_width = mask_right[y] - mask_left[y] + 1
        root_width = rx - lx + 1

        if final_width <= 0:
            continue

        if root_width > final_width * 0.88:
            continue

        left[y] = lx
        right[y] = rx

    return left, right


def detect_root_end_from_bright_root(
    initial_root_mask,
    final_mask,
    root_like,
    min_width_ratio=0.10,
    min_group_len_ratio=0.18,
    apex_padding_px=4
):
    """
    Detect true root end using the bright root region.

    This prevents the root/PDL from continuing too far into the bone
    below the actual apex.
    """

    h, w = initial_root_mask.shape

    mask_left, mask_right, mask_center, mask_width, final_y_min, final_y_max = get_mask_row_geometry(
        final_mask
    )

    row_width = np.zeros(h, dtype=np.float32)
    row_center = np.full(h, np.nan, dtype=np.float32)
    row_score = np.zeros(h, dtype=np.float32)

    for y in range(final_y_min, final_y_max + 1):
        xs = np.where(initial_root_mask[y] > 0)[0]

        if len(xs) == 0:
            continue

        row_width[y] = xs[-1] - xs[0] + 1
        row_center[y] = (xs[0] + xs[-1]) / 2.0

        vals = root_like[y, xs]

        if len(vals) > 0:
            row_score[y] = np.percentile(vals, 60)

    valid_widths = row_width[row_width > 0]

    if valid_widths.size < 10:
        return final_y_min, final_y_max

    max_root_width = np.percentile(valid_widths, 85)
    min_width_px = max(4, int(max_root_width * min_width_ratio))

    center_distance = np.abs(row_center - mask_center)
    center_allowed = np.maximum(8, mask_width * 0.25)

    valid = (
        (row_width >= min_width_px)
        & (row_score >= 0.20)
        & (center_distance <= center_allowed)
    )

    valid_u8 = valid.astype(np.uint8).reshape(-1, 1) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(5, odd(h * 0.025, 5)))
    )

    valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, kernel)
    valid = valid_u8.flatten() > 0

    valid_rows = np.where(valid)[0]

    if len(valid_rows) < 10:
        return final_y_min, final_y_max

    breaks = np.where(np.diff(valid_rows) > 1)[0]

    groups = []

    start_idx = 0

    for b in breaks:
        groups.append(valid_rows[start_idx:b + 1])
        start_idx = b + 1

    groups.append(valid_rows[start_idx:])

    min_group_len = max(
        8,
        int((final_y_max - final_y_min + 1) * min_group_len_ratio)
    )

    useful_groups = [
        g for g in groups
        if len(g) >= min_group_len
    ]

    if len(useful_groups) == 0:
        useful_groups = groups

    upper_limit = final_y_min + int((final_y_max - final_y_min + 1) * 0.35)

    upper_groups = [
        g for g in useful_groups
        if g[0] <= upper_limit
    ]

    if len(upper_groups) > 0:
        chosen = max(
            upper_groups,
            key=lambda g: np.sum(row_width[g])
        )
    else:
        chosen = max(
            useful_groups,
            key=lambda g: np.sum(row_width[g])
        )

    root_y_min = int(chosen[0])
    root_y_max = int(chosen[-1])

    root_y_max = root_y_max + int(apex_padding_px)
    root_y_max = int(np.clip(root_y_max, root_y_min + 10, final_y_max))

    return root_y_min, root_y_max


# ============================================================
# Polynomial root boundary fitting
# ============================================================

def robust_polyfit_y_to_x(
    y,
    x,
    y_min,
    y_max,
    degree=3,
    iterations=5,
    mad_scale=2.8
):
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)

    if len(y) < degree + 3:
        return None

    denom = max(float(y_max - y_min), 1.0)
    yn = (y - y_min) / denom

    keep = np.ones(len(y), dtype=bool)

    coeff = None

    for _ in range(iterations):
        if np.sum(keep) < degree + 3:
            break

        coeff = np.polyfit(
            yn[keep],
            x[keep],
            degree
        )

        pred = np.polyval(coeff, yn)

        residual = np.abs(x - pred)

        med = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - med)) + 1e-6

        keep = residual <= med + mad_scale * mad

    if coeff is None:
        return None

    return coeff


def evaluate_poly(coeff, rows, y_min, y_max):
    denom = max(float(y_max - y_min), 1.0)
    yn = (rows.astype(np.float32) - y_min) / denom

    return np.polyval(coeff, yn)


def fit_polynomial_root_boundaries(
    raw_left,
    raw_right,
    final_mask,
    polynomial_degree=3,
    end_half_width_px=2,
    apex_synthetic_weight=10,
    top_synthetic_weight=3,
    taper_ratio=0.13,
    forced_y_min=None,
    forced_y_max=None
):
    """
    Fit smooth root boundary using polynomial center + half-width curves.

    y_max is forced to bright-root end, not final mask bottom.
    """

    h, w = final_mask.shape

    mask_left, mask_right, mask_center, mask_width, mask_y_min, mask_y_max = get_mask_row_geometry(
        final_mask
    )

    if forced_y_min is None:
        y_min = mask_y_min
    else:
        y_min = int(np.clip(forced_y_min, mask_y_min, mask_y_max))

    if forced_y_max is None:
        y_max = mask_y_max
    else:
        y_max = int(np.clip(forced_y_max, y_min + 10, mask_y_max))

    valid = (
        ~np.isnan(raw_left)
        & ~np.isnan(raw_right)
        & (raw_right > raw_left + 4)
    )

    row_ids = np.arange(h)
    valid = valid & (row_ids >= y_min) & (row_ids <= y_max)

    ys_valid = np.where(valid)[0]

    if len(ys_valid) < max(12, polynomial_degree + 5):
        half = np.maximum(3, mask_width * 0.32)

        center_line = mask_center.copy()
        half_line = half.copy()

    else:
        center_raw = (raw_left[ys_valid] + raw_right[ys_valid]) / 2.0
        half_raw = (raw_right[ys_valid] - raw_left[ys_valid]) / 2.0

        y_data_center = ys_valid.astype(np.float32)
        x_data_center = center_raw.astype(np.float32)

        y_data_half = ys_valid.astype(np.float32)
        x_data_half = half_raw.astype(np.float32)

        apex_x = float(mask_center[y_max])

        for _ in range(apex_synthetic_weight):
            y_data_center = np.append(y_data_center, float(y_max))
            x_data_center = np.append(x_data_center, apex_x)

            y_data_half = np.append(y_data_half, float(y_max))
            x_data_half = np.append(x_data_half, float(end_half_width_px))

        top_y = int(ys_valid[0])
        top_center = float(center_raw[0])
        top_half = float(half_raw[0])

        for _ in range(top_synthetic_weight):
            y_data_center = np.append(y_data_center, float(top_y))
            x_data_center = np.append(x_data_center, top_center)

            y_data_half = np.append(y_data_half, float(top_y))
            x_data_half = np.append(x_data_half, top_half)

        center_coeff = robust_polyfit_y_to_x(
            y=y_data_center,
            x=x_data_center,
            y_min=y_min,
            y_max=y_max,
            degree=min(polynomial_degree, 3),
            iterations=5,
            mad_scale=3.0
        )

        half_coeff = robust_polyfit_y_to_x(
            y=y_data_half,
            x=x_data_half,
            y_min=y_min,
            y_max=y_max,
            degree=min(polynomial_degree, 3),
            iterations=5,
            mad_scale=3.0
        )

        rows = np.arange(h, dtype=np.float32)

        if center_coeff is None:
            center_line = mask_center.copy()
        else:
            center_line = evaluate_poly(
                center_coeff,
                rows,
                y_min,
                y_max
            )

        if half_coeff is None:
            half_line = np.maximum(3, mask_width * 0.32)
        else:
            half_line = evaluate_poly(
                half_coeff,
                rows,
                y_min,
                y_max
            )

    taper_len = max(8, int((y_max - y_min + 1) * taper_ratio))
    taper_start = max(y_min, y_max - taper_len)

    for y in range(taper_start, y_max + 1):
        if y_max == taper_start:
            t = 1.0
        else:
            t = (y - taper_start) / float(y_max - taper_start)

        t = t * t * (3.0 - 2.0 * t)

        half_line[y] = (1.0 - t) * half_line[y] + t * end_half_width_px
        center_line[y] = (1.0 - t) * center_line[y] + t * mask_center[y_max]

    k = odd((y_max - y_min + 1) * 0.055, 11)

    center_line = cv2.GaussianBlur(
        center_line.reshape(-1, 1).astype(np.float32),
        (1, k),
        0
    ).flatten()

    half_line = cv2.GaussianBlur(
        half_line.reshape(-1, 1).astype(np.float32),
        (1, k),
        0
    ).flatten()

    max_half = np.maximum(4, mask_width * 0.46)

    half_line = np.clip(
        half_line,
        end_half_width_px,
        max_half
    )

    left = center_line - half_line
    right = center_line + half_line

    for y in range(h):
        left[y] = max(left[y], mask_left[y] + 1)
        right[y] = min(right[y], mask_right[y] - 1)

        if right[y] <= left[y] + 2:
            cx = mask_center[y]
            left[y] = cx - end_half_width_px
            right[y] = cx + end_half_width_px

        left[y] = np.clip(left[y], 0, w - 2)
        right[y] = np.clip(right[y], left[y] + 1, w - 1)

    left[:y_min] = 0
    right[:y_min] = 0
    left[y_max + 1:] = 0
    right[y_max + 1:] = 0

    return left.astype(np.int32), right.astype(np.int32), y_min, y_max


def create_polygon_mask_from_lines(image_shape, left_line, right_line, y_min, y_max):
    h, w = image_shape

    y_min = max(0, int(y_min))
    y_max = min(h - 1, int(y_max))

    left_points = []
    right_points = []

    for y in range(y_min, y_max + 1):
        lx = int(np.clip(left_line[y], 0, w - 1))
        rx = int(np.clip(right_line[y], 0, w - 1))

        if rx <= lx:
            continue

        left_points.append([lx, y])
        right_points.append([rx, y])

    polygon = np.array(
        left_points + right_points[::-1],
        dtype=np.int32
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    if len(polygon) >= 3:
        cv2.fillPoly(mask, [polygon], 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ============================================================
# PDL candidate band
# ============================================================

def build_pdl_candidate_band(
    final_mask,
    polynomial_root_mask,
    max_pdl_search_px=24,
    min_pdl_search_px=1
):
    root_bool = polynomial_root_mask > 0
    final_bool = final_mask > 0

    outside_root = (~root_bool).astype(np.uint8) * 255

    dist_from_root = cv2.distanceTransform(
        outside_root,
        cv2.DIST_L2,
        5
    )

    candidate = (
        final_bool
        & (~root_bool)
        & (dist_from_root >= min_pdl_search_px)
        & (dist_from_root <= max_pdl_search_px)
    )

    candidate = candidate.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    return candidate, dist_from_root


# ============================================================
# Dark PDL probability
# ============================================================

def otsu_dark_threshold_inside_candidate(enhanced, candidate_mask):
    pixels = enhanced[candidate_mask > 0]

    if pixels.size < 20:
        return float(np.percentile(enhanced, 35))

    tmp = pixels.astype(np.uint8).reshape(-1, 1)

    otsu_value, _ = cv2.threshold(
        tmp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return float(otsu_value)


def kmeans_dark_cluster_inside_candidate(
    enhanced,
    candidate_mask,
    dist_from_root,
    edge01,
    n_clusters=4
):
    h, w = enhanced.shape

    img01 = normalize_01(enhanced, 2, 98)

    mean, std = local_mean_std(
        enhanced,
        ksize=max(9, int(w * 0.060))
    )

    std01 = normalize_01(std, 5, 95)
    dist01 = normalize_01(dist_from_root, 2, 98)

    ys, xs = np.where(candidate_mask > 0)

    cluster_mask = np.zeros((h, w), dtype=np.uint8)

    if len(xs) < max(40, int(h * w * 0.002)):
        return cluster_mask

    features = np.stack(
        [
            img01[ys, xs],
            mean[ys, xs],
            std01[ys, xs],
            edge01[ys, xs],
            dist01[ys, xs]
        ],
        axis=1
    ).astype(np.float32)

    k = min(n_clusters, max(2, len(xs) // 30))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        80,
        0.01
    )

    try:
        _, labels, centers = cv2.kmeans(
            features,
            int(k),
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS
        )
    except Exception:
        return cluster_mask

    labels = labels.flatten()

    dark_scores = (
        0.56 * (1.0 - centers[:, 0])
        + 0.20 * (1.0 - centers[:, 1])
        + 0.10 * centers[:, 2]
        + 0.07 * centers[:, 3]
        + 0.07 * (1.0 - np.abs(centers[:, 4] - 0.30))
    )

    best_label = int(np.argmax(dark_scores))

    selected = labels == best_label

    cluster_mask[ys[selected], xs[selected]] = 255

    return cluster_mask


def build_dark_pdl_probability(
    enhanced,
    candidate_mask,
    dist_from_root,
    expected_pdl_px=6,
    max_pdl_px=22,
    dark_percentile=50,
    adaptive_offset=0.035
):
    """
    Strong dark PDL probability map.

    This is designed to detect the entire dark region around the root,
    not only a thin line.
    """

    h, w = enhanced.shape

    features = build_multiscale_dark_features(
        enhanced=enhanced,
        candidate_mask=candidate_mask
    )

    img01 = features["img01"]
    edge01 = features["edge01"]

    candidate_pixels = enhanced[candidate_mask > 0]

    if candidate_pixels.size < 20:
        percentile_threshold = np.percentile(enhanced, 40)
    else:
        percentile_threshold = np.percentile(candidate_pixels, dark_percentile)

    otsu_threshold = otsu_dark_threshold_inside_candidate(
        enhanced,
        candidate_mask
    )

    mean, std = local_mean_std(
        enhanced,
        ksize=max(9, int(w * 0.065))
    )

    percentile_dark = (
        (enhanced.astype(np.float32) <= percentile_threshold)
        & (candidate_mask > 0)
    )

    otsu_dark = (
        (enhanced.astype(np.float32) <= otsu_threshold)
        & (candidate_mask > 0)
    )

    adaptive_dark = (
        (img01 < mean - adaptive_offset)
        & (candidate_mask > 0)
    )

    blackhat_dark = (
        (features["blackhat"] >= 0.35)
        & (candidate_mask > 0)
    )

    local_dark = (
        (
            (features["local_dark_small"] >= 0.35)
            | (features["local_dark_large"] >= 0.35)
            | (features["z_dark"] >= 0.42)
        )
        & (candidate_mask > 0)
    )

    threshold_vote = (
        percentile_dark.astype(np.float32)
        + otsu_dark.astype(np.float32)
        + adaptive_dark.astype(np.float32)
        + blackhat_dark.astype(np.float32)
        + local_dark.astype(np.float32)
    ) / 5.0

    cluster_mask = kmeans_dark_cluster_inside_candidate(
        enhanced=enhanced,
        candidate_mask=candidate_mask,
        dist_from_root=dist_from_root,
        edge01=edge01,
        n_clusters=4
    )

    cluster_like = cluster_mask.astype(np.float32) / 255.0

    sigma = max(2.5, expected_pdl_px * 1.10)

    distance_prior = np.exp(
        -0.5 * ((dist_from_root - expected_pdl_px) / sigma) ** 2
    ).astype(np.float32)

    distance_prior[dist_from_root > max_pdl_px] = 0.0
    distance_prior[dist_from_root < 1] = 0.0

    edge_support = cv2.dilate(
        ((edge01 > 0.38).astype(np.uint8) * 255),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    ).astype(np.float32) / 255.0

    probability = (
        0.24 * features["raw_darkness"]
        + 0.16 * features["local_dark_small"]
        + 0.14 * features["local_dark_large"]
        + 0.14 * features["z_dark"]
        + 0.14 * features["blackhat"]
        + 0.10 * threshold_vote
        + 0.08 * cluster_like
        + 0.04 * distance_prior
        + 0.02 * edge_support
    )

    probability[candidate_mask == 0] = 0.0

    probability = cv2.GaussianBlur(probability, (3, 3), 0)
    probability = normalize_01(probability, 2, 99)
    probability[candidate_mask == 0] = 0.0

    aux = {
        "img01": img01,
        "edge01": edge01,
        "canny": features["canny"],
        "threshold_vote": threshold_vote,
        "cluster_mask": cluster_mask,
        "distance_prior": distance_prior,
        "percentile_dark": percentile_dark.astype(np.uint8) * 255,
        "otsu_dark": otsu_dark.astype(np.uint8) * 255,
        "adaptive_dark": adaptive_dark.astype(np.uint8) * 255,
        "blackhat": features["blackhat"],
        "local_dark_small": features["local_dark_small"],
        "local_dark_large": features["local_dark_large"],
        "z_dark": features["z_dark"],
        "laplacian": features["laplacian"],
        "percentile_threshold": percentile_threshold,
        "otsu_threshold": otsu_threshold
    }

    return probability, aux


# ============================================================
# Attached dark-region growing for complete PDL
# ============================================================

def extract_attached_dark_pdl_region(
    probability,
    enhanced,
    candidate_mask,
    polynomial_root_mask,
    dist_from_root,
    high_threshold=0.48,
    low_threshold=0.30,
    contact_px=4,
    max_pdl_px=22,
    min_component_area_px=8
):
    """
    Detect the entire dark PDL region attached to the root boundary.

    Method:
        1. Strong dark pixels near root boundary become seeds.
        2. Weak dark pixels connected to seeds are grown.
        3. Only components touching the root contact zone are retained.

    This captures uneven/asymmetric dark PDL areas.
    """

    h, w = probability.shape

    candidate = candidate_mask > 0

    root_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (contact_px * 2 + 1, contact_px * 2 + 1)
    )

    dilated_root = cv2.dilate(
        polynomial_root_mask,
        root_kernel,
        iterations=1
    )

    contact_zone = (
        (dilated_root > 0)
        & (polynomial_root_mask == 0)
        & candidate
    )

    strong_dark = (
        (probability >= high_threshold)
        & candidate
        & (dist_from_root <= max_pdl_px)
    )

    weak_dark = (
        (probability >= low_threshold)
        & candidate
        & (dist_from_root <= max_pdl_px)
    )

    candidate_pixels = enhanced[candidate]

    if candidate_pixels.size > 20:
        raw_dark_threshold = np.percentile(candidate_pixels, 48)

        raw_dark = (
            (enhanced.astype(np.float32) <= raw_dark_threshold)
            & candidate
            & (dist_from_root <= max_pdl_px)
        )

        weak_dark = weak_dark | raw_dark

    seeds = strong_dark & contact_zone

    if np.count_nonzero(seeds) < 5:
        seeds = weak_dark & contact_zone

    seeds_u8 = seeds.astype(np.uint8) * 255
    weak_u8 = weak_dark.astype(np.uint8) * 255

    grown = seeds_u8.copy()

    grow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    for _ in range(80):
        previous = grown.copy()

        grown = cv2.dilate(
            grown,
            grow_kernel,
            iterations=1
        )

        grown = cv2.bitwise_and(grown, weak_u8)

        if np.array_equal(previous, grown):
            break

    grown[candidate_mask == 0] = 0
    grown[polynomial_root_mask > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        grown,
        8
    )

    out = np.zeros_like(grown)

    for label in range(1, num_labels):
        component = labels == label

        area = stats[label, cv2.CC_STAT_AREA]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area < min_component_area_px or height < 2:
            continue

        touches_root = np.count_nonzero(component & contact_zone) > 0

        if not touches_root:
            continue

        out[component] = 255

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    kernel_connect = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 5)
    )

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_connect)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_small)

    out[candidate_mask == 0] = 0
    out[polynomial_root_mask > 0] = 0

    return out


# ============================================================
# PDL cleanup
# ============================================================

def remove_small_components(mask, min_area_px=8, min_height_px=3):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        8
    )

    out = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area >= min_area_px and height >= min_height_px:
            out[labels == label] = 255

    return out


def clean_dark_pdl_mask(
    pdl_mask,
    candidate_mask,
    polynomial_root_mask,
    min_area_px=8
):
    mask = pdl_mask.copy()

    mask[candidate_mask == 0] = 0
    mask[polynomial_root_mask > 0] = 0

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

    mask = remove_small_components(
        mask,
        min_area_px=min_area_px,
        min_height_px=3
    )

    mask[candidate_mask == 0] = 0
    mask[polynomial_root_mask > 0] = 0

    return mask


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


def draw_polynomial_lines(
    image_bgr,
    left_line,
    right_line,
    y_min,
    y_max,
    color,
    thickness=1
):
    left_points = []
    right_points = []

    for y in range(int(y_min), int(y_max) + 1):
        left_points.append([int(left_line[y]), y])
        right_points.append([int(right_line[y]), y])

    if len(left_points) >= 2:
        cv2.polylines(
            image_bgr,
            [np.array(left_points, dtype=np.int32)],
            False,
            color,
            thickness
        )

    if len(right_points) >= 2:
        cv2.polylines(
            image_bgr,
            [np.array(right_points, dtype=np.int32)],
            False,
            color,
            thickness
        )

    return image_bgr


def create_debug_image(
    original,
    enhanced,
    final_mask,
    root_like,
    initial_root_mask,
    polynomial_root_mask,
    pdl_candidate,
    probability,
    raw_pdl,
    refined_pdl,
    aux,
    left_root,
    right_root,
    y_min,
    y_max,
    bright_root_y_min,
    bright_root_y_max
):
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    final_overlay = overlay_mask(
        original,
        final_mask,
        color_bgr=(255, 0, 0),
        alpha=0.22
    )

    root_like_bgr = cv2.cvtColor(
        (root_like * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    initial_root_overlay = overlay_mask(
        original,
        initial_root_mask,
        color_bgr=(0, 0, 255),
        alpha=0.35
    )

    cv2.line(
        initial_root_overlay,
        (0, int(bright_root_y_max)),
        (initial_root_overlay.shape[1] - 1, int(bright_root_y_max)),
        (255, 255, 255),
        1
    )

    polynomial_overlay = overlay_mask(
        original,
        polynomial_root_mask,
        color_bgr=(0, 0, 255),
        alpha=0.25
    )

    polynomial_overlay = draw_polynomial_lines(
        polynomial_overlay,
        left_root,
        right_root,
        y_min,
        y_max,
        color=(0, 0, 255),
        thickness=1
    )

    cv2.line(
        polynomial_overlay,
        (0, int(y_max)),
        (polynomial_overlay.shape[1] - 1, int(y_max)),
        (255, 255, 255),
        1
    )

    candidate_overlay = overlay_mask(
        original,
        pdl_candidate,
        color_bgr=(255, 255, 0),
        alpha=0.30
    )

    vote_bgr = cv2.cvtColor(
        (aux["threshold_vote"] * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    cluster_bgr = cv2.cvtColor(
        aux["cluster_mask"],
        cv2.COLOR_GRAY2BGR
    )

    edge_bgr = cv2.cvtColor(
        (aux["edge01"] * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    blackhat_bgr = cv2.cvtColor(
        (aux["blackhat"] * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    z_dark_bgr = cv2.cvtColor(
        (aux["z_dark"] * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    prob_bgr = cv2.cvtColor(
        (probability * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    raw_bgr = cv2.cvtColor(raw_pdl, cv2.COLOR_GRAY2BGR)
    refined_bgr = cv2.cvtColor(refined_pdl, cv2.COLOR_GRAY2BGR)

    final_result = overlay_mask(
        original,
        polynomial_root_mask,
        color_bgr=(0, 0, 255),
        alpha=0.20
    )

    final_result = overlay_mask(
        final_result,
        refined_pdl,
        color_bgr=(0, 255, 255),
        alpha=0.55
    )

    final_result = draw_polynomial_lines(
        final_result,
        left_root,
        right_root,
        y_min,
        y_max,
        color=(0, 0, 255),
        thickness=1
    )

    cv2.line(
        final_result,
        (0, int(y_max)),
        (final_result.shape[1] - 1, int(y_max)),
        (255, 255, 255),
        1
    )

    debug = np.hstack(
        [
            enhanced_bgr,
            final_overlay,
            root_like_bgr,
            initial_root_overlay,
            polynomial_overlay,
            candidate_overlay,
            vote_bgr,
            cluster_bgr,
            blackhat_bgr,
            z_dark_bgr,
            edge_bgr,
            prob_bgr,
            raw_bgr,
            refined_bgr,
            final_result
        ]
    )

    return debug


# ============================================================
# Main function
# ============================================================

def extract_polynomial_root_and_dark_pdl_from_final_mask(
    image_path,
    final_mask_path,

    root_mask_path=None,
    pdl_mask_path=None,
    debug_path=None,

    # Bright root detection
    root_percentile=58,

    # Polynomial root boundary
    polynomial_degree=3,
    root_end_half_width_px=2,
    apex_synthetic_weight=10,
    root_taper_ratio=0.13,

    # True root end detection
    apex_padding_px=4,

    # PDL candidate region
    min_pdl_search_px=1,
    max_pdl_search_px=24,

    # Dark PDL region detection
    expected_pdl_px=6,
    max_pdl_px=22,
    dark_percentile=50,
    adaptive_offset=0.035,
    probability_threshold=0.48,
    gap_tolerance_px=2,

    # Cleanup
    min_component_area_px=8
):
    """
    Inputs:
        1. Original X-ray
        2. bone_pdl_root_region_mask only

    Outputs:
        1. Polynomial root mask
        2. Complete attached dark PDL region mask
        3. Debug image
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = ensure_uint8_gray(img)

    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None

    final_mask = read_binary_mask(
        final_mask_path,
        target_shape=img.shape
    )

    if final_mask is None:
        print(f"ERROR: Could not read final mask: {final_mask_path}")
        return None

    enhanced = preprocess_xray_for_root_pdl(img)

    # ------------------------------------------------------------
    # 1. Bright root likelihood inside final mask.
    # ------------------------------------------------------------

    root_like, img01, homogeneity, central_prior, root_cluster = build_root_likelihood_from_final_mask(
        enhanced=enhanced,
        final_mask=final_mask
    )

    initial_root_mask, threshold_value = initial_root_mask_from_likelihood(
        root_like=root_like,
        final_mask=final_mask,
        root_percentile=root_percentile
    )

    # ------------------------------------------------------------
    # 2. Real root end from bright root.
    # ------------------------------------------------------------

    bright_root_y_min, bright_root_y_max = detect_root_end_from_bright_root(
        initial_root_mask=initial_root_mask,
        final_mask=final_mask,
        root_like=root_like,
        min_width_ratio=0.10,
        min_group_len_ratio=0.18,
        apex_padding_px=apex_padding_px
    )

    # ------------------------------------------------------------
    # 3. Raw boundary from bright root mask.
    # ------------------------------------------------------------

    h, w = img.shape

    raw_left, raw_right = extract_raw_root_boundaries(
        root_mask=initial_root_mask,
        final_mask=final_mask,
        min_row_width_px=max(6, int(w * 0.035))
    )

    # ------------------------------------------------------------
    # 4. Smooth polynomial root boundary.
    # ------------------------------------------------------------

    left_root, right_root, y_min, y_max = fit_polynomial_root_boundaries(
        raw_left=raw_left,
        raw_right=raw_right,
        final_mask=final_mask,
        polynomial_degree=polynomial_degree,
        end_half_width_px=root_end_half_width_px,
        apex_synthetic_weight=apex_synthetic_weight,
        top_synthetic_weight=3,
        taper_ratio=root_taper_ratio,
        forced_y_min=bright_root_y_min,
        forced_y_max=bright_root_y_max
    )

    polynomial_root_mask = create_polygon_mask_from_lines(
        image_shape=img.shape,
        left_line=left_root,
        right_line=right_root,
        y_min=y_min,
        y_max=y_max
    )

    polynomial_root_mask[final_mask == 0] = 0

    # ------------------------------------------------------------
    # 5. Candidate area outside root, inside final mask.
    # ------------------------------------------------------------

    pdl_candidate, dist_from_root = build_pdl_candidate_band(
        final_mask=final_mask,
        polynomial_root_mask=polynomial_root_mask,
        max_pdl_search_px=max_pdl_search_px,
        min_pdl_search_px=min_pdl_search_px
    )

    # ------------------------------------------------------------
    # 6. Multi-feature dark PDL probability.
    # ------------------------------------------------------------

    probability, aux = build_dark_pdl_probability(
        enhanced=enhanced,
        candidate_mask=pdl_candidate,
        dist_from_root=dist_from_root,
        expected_pdl_px=expected_pdl_px,
        max_pdl_px=max_pdl_px,
        dark_percentile=dark_percentile,
        adaptive_offset=adaptive_offset
    )

    # ------------------------------------------------------------
    # 7. Full attached dark-region growing.
    # ------------------------------------------------------------

    raw_pdl = extract_attached_dark_pdl_region(
        probability=probability,
        enhanced=enhanced,
        candidate_mask=pdl_candidate,
        polynomial_root_mask=polynomial_root_mask,
        dist_from_root=dist_from_root,
        high_threshold=probability_threshold,
        low_threshold=max(0.22, probability_threshold - 0.18),
        contact_px=4,
        max_pdl_px=max_pdl_px,
        min_component_area_px=min_component_area_px
    )

    refined_pdl = clean_dark_pdl_mask(
        pdl_mask=raw_pdl,
        candidate_mask=pdl_candidate,
        polynomial_root_mask=polynomial_root_mask,
        min_area_px=min_component_area_px
    )

    # ------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------

    safe_imwrite(root_mask_path, polynomial_root_mask)
    safe_imwrite(pdl_mask_path, refined_pdl)

    if debug_path:
        debug = create_debug_image(
            original=img,
            enhanced=enhanced,
            final_mask=final_mask,
            root_like=root_like,
            initial_root_mask=initial_root_mask,
            polynomial_root_mask=polynomial_root_mask,
            pdl_candidate=pdl_candidate,
            probability=probability,
            raw_pdl=raw_pdl,
            refined_pdl=refined_pdl,
            aux=aux,
            left_root=left_root,
            right_root=right_root,
            y_min=y_min,
            y_max=y_max,
            bright_root_y_min=bright_root_y_min,
            bright_root_y_max=bright_root_y_max
        )

        safe_imwrite(debug_path, debug)

    print(
        f"Processed: {os.path.basename(image_path)} | "
        f"root_threshold={threshold_value:.3f} | "
        f"bright_root_end={bright_root_y_max} | "
        f"poly_end={y_max} | "
        f"root_px={np.count_nonzero(polynomial_root_mask)} | "
        f"pdl_px={np.count_nonzero(refined_pdl)}"
    )

    return {
        "polynomial_root_mask": polynomial_root_mask,
        "dark_pdl_mask": refined_pdl,
        "pdl_candidate": pdl_candidate,
        "root_like": root_like,
        "pdl_probability": probability,
        "left_root": left_root,
        "right_root": right_root,
        "y_min": y_min,
        "y_max": y_max,
        "bright_root_y_min": bright_root_y_min,
        "bright_root_y_max": bright_root_y_max
    }