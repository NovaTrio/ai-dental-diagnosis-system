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

    p_low = np.percentile(arr, low)
    p_high = np.percentile(arr, high)

    if p_high <= p_low:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - p_low) / (p_high - p_low)
    out = np.clip(out, 0.0, 1.0)

    return out.astype(np.float32)


def smooth_1d(signal, ksize):
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) < 3:
        return signal

    ksize = odd(ksize, 3)

    if ksize >= len(signal):
        ksize = len(signal) - 1 if len(signal) % 2 == 0 else len(signal)

    if ksize < 3:
        return signal

    return cv2.GaussianBlur(
        signal.reshape(-1, 1),
        (1, ksize),
        0
    ).flatten()


def sample_x_offset(img, dx):
    h, w = img.shape
    pad = abs(int(dx)) + 4

    padded = np.pad(
        img,
        ((0, 0), (pad, pad)),
        mode="edge"
    )

    start = pad + int(dx)
    end = start + w

    return padded[:, start:end]


def robust_smooth_boundary(boundary, image_width, ksize=21):
    boundary = boundary.astype(np.float32)

    smooth = smooth_1d(boundary, ksize)

    residual = np.abs(boundary - smooth)
    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) + 1e-6

    outlier = residual > med + 2.8 * mad

    cleaned = boundary.copy()
    cleaned[outlier] = smooth[outlier]

    cleaned = smooth_1d(cleaned, ksize)
    cleaned = np.clip(cleaned, 0, image_width - 1)

    return cleaned.astype(np.int32)


# ============================================================
# Preprocessing
# ============================================================

def preprocess_xray(img_uint8):
    """
    Mild enhancement only.

    Strong CLAHE is avoided because it can make trabecular bone
    look as dark as the PDL space.
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


# ============================================================
# Feature maps
# ============================================================

def build_edge_link_map(img_uint8):
    """
    Canny + vertical morphological linking.
    This helps connect broken root and lamina dura edges.
    """

    h, w = img_uint8.shape

    blurred = cv2.GaussianBlur(img_uint8, (3, 3), 0)

    med = np.median(blurred)

    lower = int(max(0, 0.60 * med))
    upper = int(min(255, 1.45 * med))

    canny = cv2.Canny(blurred, lower, upper)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(9, odd(h * 0.035, 9)))
    )

    linked = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, vertical_kernel)

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    linked = cv2.dilate(linked, small_kernel, iterations=1)

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

    linked01 = linked.astype(np.float32) / 255.0

    edge_link = np.maximum(linked01, grad01)
    edge_link = cv2.GaussianBlur(edge_link, (3, 3), 0)
    edge_link = normalize_01(edge_link, 2, 98)

    return edge_link


def build_root_likelihood(img_uint8):
    """
    Root likelihood map.

    Root is expected to be:
    - brighter than PDL
    - relatively homogeneous
    - closer to the selected tooth center
    """

    h, w = img_uint8.shape

    img01 = normalize_01(img_uint8, 2, 98)

    mean, std = local_mean_std(
        img_uint8,
        ksize=max(9, int(w * 0.07))
    )

    std01 = normalize_01(std, 5, 95)
    homogeneity = 1.0 - std01

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
    grad01 = normalize_01(grad, 5, 95)

    # K-means feature clustering.
    features = np.stack(
        [
            img01.reshape(-1),
            mean.reshape(-1),
            homogeneity.reshape(-1),
            0.35 * grad01.reshape(-1)
        ],
        axis=1
    ).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
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

        labels = labels.reshape(h, w)

        root_scores = (
            0.55 * centers[:, 0]
            + 0.25 * centers[:, 1]
            + 0.20 * centers[:, 2]
        )

        root_label = int(np.argmax(root_scores))
        root_cluster = (labels == root_label).astype(np.float32)

    except Exception:
        root_cluster = np.zeros((h, w), dtype=np.float32)

    xs = np.arange(w, dtype=np.float32)
    center_x = (w - 1) / 2.0

    sigma = max(8.0, w * 0.27)

    central_prior = np.exp(
        -0.5 * ((xs - center_x) / sigma) ** 2
    ).astype(np.float32)

    central_prior = np.tile(
        central_prior.reshape(1, -1),
        (h, 1)
    )

    root_like = (
        0.34 * img01
        + 0.22 * mean
        + 0.20 * homogeneity
        + 0.14 * root_cluster
        + 0.10 * central_prior
    )

    root_like = cv2.GaussianBlur(root_like, (5, 5), 0)
    root_like = normalize_01(root_like, 2, 98)

    return root_like, img01, homogeneity, grad01


# ============================================================
# Centerline estimation
# ============================================================

def estimate_centerline(root_like, central_width_ratio=0.72):
    h, w = root_like.shape

    xs = np.arange(w, dtype=np.float32)
    image_center = (w - 1) / 2.0

    sigma = max(8.0, w * 0.24)

    central_prior = np.exp(
        -0.5 * ((xs - image_center) / sigma) ** 2
    ).astype(np.float32)

    x1 = int(w * (1.0 - central_width_ratio) / 2.0)
    x2 = int(w * (1.0 + central_width_ratio) / 2.0)

    x1 = max(0, x1)
    x2 = min(w, x2)

    centerline = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        weights = root_like[y].copy()

        weights[:x1] = 0
        weights[x2:] = 0

        weights *= central_prior

        total = np.sum(weights)

        if total > 1e-6:
            centerline[y] = np.sum(weights * xs) / total

    valid = ~np.isnan(centerline)

    if np.sum(valid) < max(10, int(h * 0.15)):
        centerline[:] = image_center
    else:
        ys = np.arange(h)

        centerline = np.interp(
            ys,
            ys[valid],
            centerline[valid]
        ).astype(np.float32)

    centerline = smooth_1d(centerline, odd(h * 0.080, 21))
    centerline = np.clip(centerline, 0, w - 1)

    return centerline


# ============================================================
# Root boundary tracing
# ============================================================

def build_root_boundary_response(
    img_uint8,
    root_like,
    img01,
    edge_link,
    side
):
    """
    Inner root surface boundary response.

    Left root boundary:
        dark outside / PDL -> bright root
        horizontal gradient is positive

    Right root boundary:
        bright root -> dark outside / PDL
        horizontal gradient is negative
    """

    h, w = img_uint8.shape

    gx = cv2.Sobel(
        img_uint8.astype(np.float32),
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    offset = max(2, int(w * 0.018))

    if side == "left":
        signed_edge = np.maximum(gx, 0.0)
        inside_root = sample_x_offset(root_like, +offset)
        outside_dark = 1.0 - sample_x_offset(img01, -offset)
    else:
        signed_edge = np.maximum(-gx, 0.0)
        inside_root = sample_x_offset(root_like, -offset)
        outside_dark = 1.0 - sample_x_offset(img01, +offset)

    signed_edge = normalize_01(signed_edge, 2, 99)

    response = (
        0.34 * signed_edge
        + 0.24 * edge_link
        + 0.32 * inside_root
        + 0.10 * outside_dark
    )

    response = cv2.GaussianBlur(response, (3, 3), 0)
    response = normalize_01(response, 2, 99)

    response[:, :2] = 0
    response[:, -2:] = 0

    return response


def trace_boundary_dp(
    response,
    centerline,
    side,
    min_half_width_ratio=0.040,
    max_half_width_ratio=0.36,
    expected_half_width_ratio=0.19,
    max_step_ratio=0.035,
    smoothness=1.75
):
    """
    Dynamic programming line tracing.

    This links weak/broken edge fragments into one continuous smooth
    anatomical boundary.
    """

    h, w = response.shape

    min_half_width = max(5, int(w * min_half_width_ratio))
    max_half_width = max(min_half_width + 8, int(w * max_half_width_ratio))
    expected_half_width = max(min_half_width + 2, int(w * expected_half_width_ratio))

    max_step = max(3, int(w * max_step_ratio))

    large = 1e8

    local_cost = np.full((h, w), large, dtype=np.float32)

    for y in range(h):
        cx = int(round(centerline[y]))

        if side == "left":
            x1 = max(0, cx - max_half_width)
            x2 = max(0, cx - min_half_width)
        else:
            x1 = min(w - 1, cx + min_half_width)
            x2 = min(w - 1, cx + max_half_width)

        if x2 <= x1:
            continue

        xs = np.arange(x1, x2 + 1)

        distance = np.abs(xs - cx)

        width_sigma = max(5.0, w * 0.13)

        width_prior = np.exp(
            -0.5 * ((distance - expected_half_width) / width_sigma) ** 2
        ).astype(np.float32)

        score = (
            0.86 * response[y, xs]
            + 0.14 * width_prior
        )

        local_cost[y, xs] = -score

    dp = np.full((h, w), large, dtype=np.float32)
    backtrack = np.full((h, w), -1, dtype=np.int32)

    dp[0] = local_cost[0]

    for y in range(1, h):
        valid_xs = np.where(local_cost[y] < large / 2)[0]

        for x in valid_xs:
            p1 = max(0, x - max_step)
            p2 = min(w, x + max_step + 1)

            prev_xs = np.arange(p1, p2)

            transition = smoothness * (
                np.abs(prev_xs - x).astype(np.float32) / max_step
            )

            costs = dp[y - 1, p1:p2] + transition

            best_idx = int(np.argmin(costs))

            dp[y, x] = local_cost[y, x] + costs[best_idx]
            backtrack[y, x] = p1 + best_idx

    final_candidates = np.where(dp[h - 1] < large / 2)[0]

    if len(final_candidates) == 0:
        fallback = int(w * expected_half_width_ratio)

        if side == "left":
            boundary = centerline - fallback
        else:
            boundary = centerline + fallback

        boundary = np.clip(boundary, 0, w - 1)
        return boundary.astype(np.int32)

    x = int(final_candidates[np.argmin(dp[h - 1, final_candidates])])

    boundary = np.zeros(h, dtype=np.int32)

    for y in range(h - 1, -1, -1):
        boundary[y] = x

        previous_x = backtrack[y, x]

        if y > 0 and previous_x >= 0:
            x = int(previous_x)

    boundary = robust_smooth_boundary(
        boundary,
        image_width=w,
        ksize=odd(h * 0.060, 15)
    )

    return boundary


def enforce_root_line_order(
    left_root,
    right_root,
    centerline,
    image_width,
    min_gap_px=10,
    fallback_half_width_ratio=0.19
):
    h = len(left_root)

    left = left_root.copy()
    right = right_root.copy()

    fallback_half = max(5, int(image_width * fallback_half_width_ratio))

    for y in range(h):
        cx = int(round(centerline[y]))

        if right[y] <= left[y] + min_gap_px:
            left[y] = cx - fallback_half
            right[y] = cx + fallback_half

        left[y] = int(np.clip(left[y], 0, image_width - 2))
        right[y] = int(np.clip(right[y], left[y] + min_gap_px, image_width - 1))

    left = robust_smooth_boundary(
        left,
        image_width=image_width,
        ksize=odd(h * 0.050, 13)
    )

    right = robust_smooth_boundary(
        right,
        image_width=image_width,
        ksize=odd(h * 0.050, 13)
    )

    for y in range(h):
        right[y] = int(np.clip(right[y], left[y] + min_gap_px, image_width - 1))

    return left, right


# ============================================================
# Mask helpers
# ============================================================

def create_between_lines_mask(image_shape, left_boundary, right_boundary, end_y=None):
    h, w = image_shape

    if end_y is None:
        rows = h
    else:
        rows = min(h, int(end_y) + 1)

    mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(rows):
        lx = int(np.clip(left_boundary[y], 0, w - 1))
        rx = int(np.clip(right_boundary[y], 0, w - 1))

        if rx > lx:
            mask[y, lx:rx + 1] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def fill_internal_root_dark_regions(root_like, left_root, right_root):
    """
    Fill dark internal tooth/root structures such as pulp canal shadows.

    This prevents internal dark root texture from being confused with
    external PDL.
    """

    h, w = root_like.shape

    root_core_mask = create_between_lines_mask(
        image_shape=(h, w),
        left_boundary=left_root,
        right_boundary=right_root,
        end_y=h - 1
    )

    filled = root_like.copy()

    inside = root_core_mask > 0
    filled[inside] = np.maximum(filled[inside], 0.88)

    filled = cv2.GaussianBlur(filled, (5, 5), 0)
    filled = normalize_01(filled, 2, 98)

    return filled, root_core_mask


# ============================================================
# Outer PDL / lamina dura boundary tracing
# ============================================================

def build_outer_pdl_score_map(
    img01,
    edge_link,
    root_boundary,
    side,
    pdl_min_px=1,
    pdl_max_px=16,
    lamina_search_px=20,
    fallback_pdl_width_px=7
):
    """
    Score map for the yellow boundary.

    Search direction:
        root surface -> dark PDL valley -> bright lamina dura recovery

    The candidate yellow line receives a high score when:
    - the region between root and candidate is dark
    - the candidate/outside area becomes bright again
    - contrast exists between PDL valley and lamina dura
    - an edge exists near the candidate
    """

    h, w = img01.shape

    score = np.zeros((h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.uint8)

    direction = -1 if side == "left" else 1

    min_d = max(2, int(pdl_min_px + 1))
    max_d = int(pdl_max_px + lamina_search_px)
    max_d = max(max_d, fallback_pdl_width_px + 5)

    expected_d = max(4, int(fallback_pdl_width_px + 4))
    sigma_d = max(3.0, expected_d * 0.85)

    for y in range(h):
        root_x = int(root_boundary[y])

        for d in range(min_d, max_d + 1):
            x = root_x + direction * d

            if x < 2 or x >= w - 2:
                continue

            if side == "left":
                inner_start = x + 1
                inner_end = root_x
                candidate_start = max(0, x - 2)
                candidate_end = min(w, x + 3)
                outside_start = max(0, x - 4)
                outside_end = x + 1
            else:
                inner_start = root_x + 1
                inner_end = x
                candidate_start = max(0, x - 2)
                candidate_end = min(w, x + 3)
                outside_start = x
                outside_end = min(w, x + 5)

            if inner_end <= inner_start:
                continue

            pdl_segment = img01[y, inner_start:inner_end]

            if len(pdl_segment) < 2:
                continue

            candidate_segment = img01[y, candidate_start:candidate_end]
            outside_segment = img01[y, outside_start:outside_end]

            valley_value = np.percentile(pdl_segment, 15)
            pdl_darkness = 1.0 - valley_value

            candidate_bright = np.percentile(candidate_segment, 75)
            outside_bright = np.percentile(outside_segment, 70)

            lamina_bright = max(candidate_bright, outside_bright)

            contrast = max(0.0, lamina_bright - valley_value)

            edge_strength = edge_link[y, x]

            width_prior = np.exp(
                -0.5 * ((d - expected_d) / sigma_d) ** 2
            )

            candidate_score = (
                0.34 * pdl_darkness
                + 0.28 * lamina_bright
                + 0.22 * contrast
                + 0.10 * edge_strength
                + 0.06 * width_prior
            )

            score[y, x] = candidate_score
            valid[y, x] = 1

    score = normalize_01(score, 2, 99)

    return score, valid


def trace_outer_pdl_boundary_dp(
    score,
    valid,
    root_boundary,
    side,
    pdl_min_px=1,
    pdl_max_px=16,
    lamina_search_px=20,
    fallback_pdl_width_px=7,
    max_step_ratio=0.035,
    smoothness=1.95
):
    h, w = score.shape

    direction = -1 if side == "left" else 1

    min_d = max(2, int(pdl_min_px + 1))
    max_d = int(pdl_max_px + lamina_search_px)
    max_d = max(max_d, fallback_pdl_width_px + 5)

    max_step = max(3, int(w * max_step_ratio))

    large = 1e8

    local_cost = np.full((h, w), large, dtype=np.float32)

    for y in range(h):
        root_x = int(root_boundary[y])

        for d in range(min_d, max_d + 1):
            x = root_x + direction * d

            if x < 0 or x >= w:
                continue

            if valid[y, x] == 0:
                continue

            local_cost[y, x] = -score[y, x]

    dp = np.full((h, w), large, dtype=np.float32)
    backtrack = np.full((h, w), -1, dtype=np.int32)

    dp[0] = local_cost[0]

    for y in range(1, h):
        valid_xs = np.where(local_cost[y] < large / 2)[0]

        for x in valid_xs:
            p1 = max(0, x - max_step)
            p2 = min(w, x + max_step + 1)

            prev_xs = np.arange(p1, p2)

            transition = smoothness * (
                np.abs(prev_xs - x).astype(np.float32) / max_step
            )

            costs = dp[y - 1, p1:p2] + transition

            best_idx = int(np.argmin(costs))

            dp[y, x] = local_cost[y, x] + costs[best_idx]
            backtrack[y, x] = p1 + best_idx

    final_candidates = np.where(dp[h - 1] < large / 2)[0]

    if len(final_candidates) == 0:
        outer = root_boundary + direction * fallback_pdl_width_px
        outer = np.clip(outer, 0, w - 1)
        return outer.astype(np.int32)

    x = int(final_candidates[np.argmin(dp[h - 1, final_candidates])])

    outer = np.zeros(h, dtype=np.int32)

    for y in range(h - 1, -1, -1):
        outer[y] = x

        previous_x = backtrack[y, x]

        if y > 0 and previous_x >= 0:
            x = int(previous_x)

    outer = robust_smooth_boundary(
        outer,
        image_width=w,
        ksize=odd(h * 0.060, 15)
    )

    return outer


def detect_outer_pdl_boundary(
    img01,
    edge_link,
    root_boundary,
    side,
    pdl_min_px=1,
    pdl_max_px=16,
    lamina_search_px=20,
    fallback_pdl_width_px=7
):
    score, valid = build_outer_pdl_score_map(
        img01=img01,
        edge_link=edge_link,
        root_boundary=root_boundary,
        side=side,
        pdl_min_px=pdl_min_px,
        pdl_max_px=pdl_max_px,
        lamina_search_px=lamina_search_px,
        fallback_pdl_width_px=fallback_pdl_width_px
    )

    outer = trace_outer_pdl_boundary_dp(
        score=score,
        valid=valid,
        root_boundary=root_boundary,
        side=side,
        pdl_min_px=pdl_min_px,
        pdl_max_px=pdl_max_px,
        lamina_search_px=lamina_search_px,
        fallback_pdl_width_px=fallback_pdl_width_px
    )

    return outer, score


def enforce_outer_line_order(
    left_root,
    right_root,
    left_outer,
    right_outer,
    image_width,
    pdl_min_px=1,
    pdl_max_px=16,
    lamina_search_px=20
):
    h = len(left_root)

    left_fixed = left_outer.copy()
    right_fixed = right_outer.copy()

    min_dist = max(2, int(pdl_min_px + 1))
    max_dist = max(min_dist + 5, int(pdl_max_px + lamina_search_px))

    for y in range(h):
        lr = int(left_root[y])
        rr = int(right_root[y])

        left_min = max(0, lr - max_dist)
        left_max = max(0, lr - min_dist)

        right_min = min(image_width - 1, rr + min_dist)
        right_max = min(image_width - 1, rr + max_dist)

        left_fixed[y] = int(np.clip(left_fixed[y], left_min, left_max))
        right_fixed[y] = int(np.clip(right_fixed[y], right_min, right_max))

        right_fixed[y] = int(
            np.clip(right_fixed[y], left_fixed[y] + 1, image_width - 1)
        )

    left_fixed = robust_smooth_boundary(
        left_fixed,
        image_width=image_width,
        ksize=odd(h * 0.050, 13)
    )

    right_fixed = robust_smooth_boundary(
        right_fixed,
        image_width=image_width,
        ksize=odd(h * 0.050, 13)
    )

    for y in range(h):
        right_fixed[y] = int(
            np.clip(right_fixed[y], left_fixed[y] + 1, image_width - 1)
        )

    return left_fixed, right_fixed


# ============================================================
# Apex / root-end detection
# ============================================================

def choose_lamina_convergence_end_y(
    left_outer,
    right_outer,
    left_root,
    right_root,
    root_like_filled,
    min_end_ratio=0.50,
    max_end_ratio=0.93,
    convergence_ratio=0.30,
    end_padding_px=5
):
    """
    Detect the bottom margin using convergence of left/right yellow lines.

    The true root-end / PDL-end is where the outer PDL/lamina lines
    converge around the apex.
    """

    h, w = root_like_filled.shape

    left_outer = left_outer.astype(np.float32)
    right_outer = right_outer.astype(np.float32)
    left_root = left_root.astype(np.float32)
    right_root = right_root.astype(np.float32)

    outer_gap = right_outer - left_outer
    root_gap = right_root - left_root

    outer_gap = np.maximum(outer_gap, 1)
    root_gap = np.maximum(root_gap, 1)

    outer_gap = smooth_1d(outer_gap, odd(h * 0.070, 17))
    root_gap = smooth_1d(root_gap, odd(h * 0.070, 17))

    root_profile = np.zeros(h, dtype=np.float32)

    for y in range(h):
        lx = int(np.clip(left_root[y], 0, w - 1))
        rx = int(np.clip(right_root[y], 0, w - 1))

        if rx <= lx + 3:
            continue

        vals = root_like_filled[y, lx:rx + 1]

        if len(vals) > 0:
            root_profile[y] = np.percentile(vals, 60)

    root_profile = smooth_1d(root_profile, odd(h * 0.060, 15))
    root_profile = normalize_01(root_profile, 5, 95)

    ref_start = int(h * 0.15)
    ref_end = int(h * 0.45)

    if ref_end <= ref_start + 5:
        ref_start = 0
        ref_end = max(8, int(h * 0.35))

    reference_gap = np.percentile(outer_gap[ref_start:ref_end], 70)

    convergence_threshold = max(
        w * 0.050,
        reference_gap * convergence_ratio
    )

    start_y = int(h * min_end_ratio)
    end_y = int(h * max_end_ratio)

    start_y = max(3, min(start_y, h - 10))
    end_y = max(start_y + 10, min(end_y, h - 2))

    search_gap = outer_gap[start_y:end_y]
    search_root = root_profile[start_y:end_y]

    gap_norm = normalize_01(search_gap, 5, 95)
    root_norm = normalize_01(search_root, 5, 95)

    y_norm = np.linspace(0, 1, len(search_gap)).astype(np.float32)

    # Lower score = better apex candidate.
    # Small gap is strongest signal.
    # Weak root likelihood and deeper position are secondary signals.
    score = (
        0.68 * gap_norm
        + 0.22 * root_norm
        - 0.10 * y_norm
    )

    candidate_rows = np.where(search_gap <= convergence_threshold)[0]

    if len(candidate_rows) > 0:
        # Use the deepest converged section, not the first one.
        splits = np.where(np.diff(candidate_rows) > 1)[0]

        groups = []

        start_idx = 0

        for split in splits:
            groups.append(candidate_rows[start_idx:split + 1])
            start_idx = split + 1

        groups.append(candidate_rows[start_idx:])

        groups = [g for g in groups if len(g) >= max(3, int(h * 0.010))]

        if len(groups) > 0:
            chosen_group = groups[-1]
            local_scores = score[chosen_group]
            local_best = chosen_group[int(np.argmin(local_scores))]
            detected_y = start_y + int(local_best)
        else:
            detected_y = start_y + int(candidate_rows[-1])
    else:
        detected_y = start_y + int(np.argmin(score))

    detected_y = detected_y + int(end_padding_px)
    detected_y = int(np.clip(detected_y, start_y, end_y))

    return detected_y, root_profile, outer_gap


# ============================================================
# Final line tapering and mask creation
# ============================================================

def taper_lines_to_apex(
    left_line,
    right_line,
    centerline,
    end_y,
    image_width,
    taper_ratio=0.12,
    end_half_width_px=3
):
    rows = int(end_y) + 1

    left = left_line[:rows].astype(np.float32).copy()
    right = right_line[:rows].astype(np.float32).copy()

    taper_len = max(8, int(rows * taper_ratio))
    start_y = max(0, rows - taper_len)

    end_cx = int(round(centerline[end_y]))
    end_cx = int(np.clip(end_cx, 0, image_width - 1))

    target_left = end_cx - end_half_width_px
    target_right = end_cx + end_half_width_px

    for y in range(start_y, rows):
        if rows - 1 == start_y:
            t = 1.0
        else:
            t = (y - start_y) / float((rows - 1) - start_y)

        # Smoothstep.
        t = t * t * (3 - 2 * t)

        left[y] = (1.0 - t) * left[y] + t * target_left
        right[y] = (1.0 - t) * right[y] + t * target_right

    smooth_k = odd(rows * 0.045, 11)

    left = smooth_1d(left, smooth_k)
    right = smooth_1d(right, smooth_k)

    left = np.clip(left, 0, image_width - 2).astype(np.int32)

    fixed_right = np.zeros_like(left)

    for y in range(rows):
        fixed_right[y] = int(np.clip(right[y], left[y] + 1, image_width - 1))

    return left, fixed_right


def create_polygon_mask(image_shape, left_boundary, right_boundary, end_y):
    h, w = image_shape

    rows = int(end_y) + 1
    rows = min(rows, len(left_boundary), len(right_boundary), h)

    left_points = []
    right_points = []

    for y in range(rows):
        lx = int(np.clip(left_boundary[y], 0, w - 1))
        rx = int(np.clip(right_boundary[y], 0, w - 1))

        left_points.append([lx, y])
        right_points.append([rx, y])

    polygon = np.array(
        left_points + right_points[::-1],
        dtype=np.int32
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    if len(polygon) >= 3:
        cv2.fillPoly(mask, [polygon], 255)

    return mask


def keep_largest_central_component(mask):
    h, w = mask.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        8
    )

    if num_labels <= 1:
        return mask

    image_center = w / 2.0

    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        cx = centroids[label][0]

        if area < h * w * 0.002:
            continue

        central_score = 1.0 - min(
            1.0,
            abs(cx - image_center) / max(image_center, 1)
        )

        score = area * (0.35 + central_score)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return mask

    return ((labels == best_label).astype(np.uint8) * 255)


def add_fixed_outer_space(mask, outer_space_px=4):
    outer_space_px = max(0, int(outer_space_px))

    if outer_space_px <= 0:
        return mask

    k = outer_space_px * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k)
    )

    return cv2.dilate(mask, kernel, iterations=1)


def smooth_mask_edge(mask, smooth_px=5):
    smooth_px = max(1, int(smooth_px))

    k = odd(smooth_px * 2 + 1, 3)

    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    smooth = ((blurred >= 127).astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k)
    )

    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel)

    smooth = keep_largest_central_component(smooth)

    return smooth


# ============================================================
# Debug visualization
# ============================================================

def line_points(left_line, right_line, end_y):
    rows = int(end_y) + 1
    rows = min(rows, len(left_line), len(right_line))

    pts = []

    for y in range(rows):
        pts.append([int(left_line[y]), y])

    for y in range(rows - 1, -1, -1):
        pts.append([int(right_line[y]), y])

    return np.array(pts, dtype=np.int32)


def draw_side_lines(overlay, left_line, right_line, end_y, color, thickness=1):
    rows = int(end_y) + 1
    rows = min(rows, len(left_line), len(right_line), overlay.shape[0])

    left_pts = []
    right_pts = []

    for y in range(rows):
        left_pts.append([int(left_line[y]), y])
        right_pts.append([int(right_line[y]), y])

    if len(left_pts) >= 2:
        cv2.polylines(
            overlay,
            [np.array(left_pts, dtype=np.int32)],
            False,
            color,
            thickness
        )

    if len(right_pts) >= 2:
        cv2.polylines(
            overlay,
            [np.array(right_pts, dtype=np.int32)],
            False,
            color,
            thickness
        )


def create_debug_image(
    original,
    enhanced,
    root_like,
    root_like_filled,
    edge_link,
    root_mask,
    pdl_mask,
    outer_mask,
    final_mask,
    left_root,
    right_root,
    left_outer,
    right_outer,
    centerline,
    end_y,
    root_profile,
    gap_profile,
    left_outer_score,
    right_outer_score
):
    h, w = original.shape

    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Root mask red transparent.
    red_fill = np.zeros_like(overlay)
    red_fill[:, :, 2] = 255

    overlay[root_mask > 0] = (
        0.82 * overlay[root_mask > 0]
        + 0.18 * red_fill[root_mask > 0]
    ).astype(np.uint8)

    # PDL mask yellow transparent.
    yellow_fill = np.zeros_like(overlay)
    yellow_fill[:, :, 1] = 255
    yellow_fill[:, :, 2] = 255

    overlay[pdl_mask > 0] = (
        0.72 * overlay[pdl_mask > 0]
        + 0.28 * yellow_fill[pdl_mask > 0]
    ).astype(np.uint8)

    # Red = root boundary.
    draw_side_lines(
        overlay,
        left_root,
        right_root,
        end_y,
        color=(0, 0, 255),
        thickness=1
    )

    # Yellow = outer PDL / lamina dura boundary.
    draw_side_lines(
        overlay,
        left_outer,
        right_outer,
        end_y,
        color=(0, 255, 255),
        thickness=1
    )

    # Outer yellow closed line.
    outer_pts = line_points(left_outer, right_outer, end_y)

    if len(outer_pts) >= 3:
        cv2.polylines(
            overlay,
            [outer_pts],
            True,
            (0, 255, 255),
            1
        )

    # Root red closed line.
    root_pts = line_points(left_root, right_root, end_y)

    if len(root_pts) >= 3:
        cv2.polylines(
            overlay,
            [root_pts],
            True,
            (0, 0, 255),
            1
        )

    # Cyan centerline.
    rows = min(int(end_y) + 1, h)

    for y in range(0, rows, 5):
        cv2.circle(
            overlay,
            (int(centerline[y]), y),
            1,
            (255, 255, 0),
            -1
        )

    # Apex/end row.
    cv2.line(
        overlay,
        (0, int(end_y)),
        (w - 1, int(end_y)),
        (255, 255, 255),
        1
    )

    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    root_like_bgr = cv2.cvtColor((root_like * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    filled_like_bgr = cv2.cvtColor((root_like_filled * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    edge_bgr = cv2.cvtColor((edge_link * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    root_mask_bgr = cv2.cvtColor(root_mask, cv2.COLOR_GRAY2BGR)
    pdl_mask_bgr = cv2.cvtColor(pdl_mask, cv2.COLOR_GRAY2BGR)
    outer_mask_bgr = cv2.cvtColor(outer_mask, cv2.COLOR_GRAY2BGR)
    final_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

    outer_score = np.maximum(left_outer_score, right_outer_score)
    outer_score_bgr = cv2.cvtColor((outer_score * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    profile_img = np.zeros((h, w), dtype=np.uint8)

    root_norm = normalize_01(root_profile, 5, 95)
    gap_norm = normalize_01(gap_profile, 5, 95)

    for y in range(h):
        r_len = int(root_norm[y] * (w - 1))
        g_len = int(gap_norm[y] * (w - 1))

        cv2.line(profile_img, (0, y), (r_len, y), 120, 1)
        cv2.line(profile_img, (0, y), (g_len, y), 255, 1)

    profile_bgr = cv2.cvtColor(profile_img, cv2.COLOR_GRAY2BGR)

    debug = np.hstack(
        [
            enhanced_bgr,
            root_like_bgr,
            filled_like_bgr,
            edge_bgr,
            outer_score_bgr,
            overlay,
            root_mask_bgr,
            pdl_mask_bgr,
            outer_mask_bgr,
            final_mask_bgr,
            profile_bgr
        ]
    )

    return debug


# ============================================================
# Main extraction function
# ============================================================

def extract_root_pdl_layered_roi(
    image_path,
    mask_path=None,
    roi_path=None,
    debug_path=None,

    root_mask_path=None,
    pdl_mask_path=None,
    outer_mask_path=None,

    # Space outside yellow PDL/lamina boundary.
    outer_space_px=4,

    # Final mask smoothness.
    edge_smooth_px=5,

    # Apex/root-end detection.
    min_end_ratio=0.50,
    max_end_ratio=0.93,
    convergence_ratio=0.30,
    end_padding_px=5,

    # Red root boundary search.
    min_half_width_ratio=0.040,
    max_half_width_ratio=0.36,
    expected_half_width_ratio=0.19,

    # Yellow PDL/lamina search.
    pdl_min_px=1,
    pdl_max_px=16,
    lamina_search_px=20,
    fallback_pdl_width_px=7,

    # Apex closure.
    root_taper_ratio=0.12,
    outer_taper_ratio=0.12,
    root_end_half_width_px=2,
    outer_end_half_width_px=5
):
    """
    Extract:
        1. Root region inside red boundary.
        2. PDL region between red and yellow boundaries.
        3. Final ROI = root + PDL + small outside margin.

    Debug color convention:
        Red    = root/tooth boundary.
        Yellow = outer PDL / lamina dura boundary.
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = ensure_uint8_gray(img)

    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None

    h, w = img.shape

    enhanced = preprocess_xray(img)

    root_like, img01, homogeneity, grad01 = build_root_likelihood(enhanced)

    edge_link = build_edge_link_map(enhanced)

    centerline = estimate_centerline(
        root_like,
        central_width_ratio=0.72
    )

    # ------------------------------------------------------------
    # First pass root boundary.
    # ------------------------------------------------------------

    left_response_1 = build_root_boundary_response(
        img_uint8=enhanced,
        root_like=root_like,
        img01=img01,
        edge_link=edge_link,
        side="left"
    )

    right_response_1 = build_root_boundary_response(
        img_uint8=enhanced,
        root_like=root_like,
        img01=img01,
        edge_link=edge_link,
        side="right"
    )

    left_root_1 = trace_boundary_dp(
        response=left_response_1,
        centerline=centerline,
        side="left",
        min_half_width_ratio=min_half_width_ratio,
        max_half_width_ratio=max_half_width_ratio,
        expected_half_width_ratio=expected_half_width_ratio
    )

    right_root_1 = trace_boundary_dp(
        response=right_response_1,
        centerline=centerline,
        side="right",
        min_half_width_ratio=min_half_width_ratio,
        max_half_width_ratio=max_half_width_ratio,
        expected_half_width_ratio=expected_half_width_ratio
    )

    left_root_1, right_root_1 = enforce_root_line_order(
        left_root=left_root_1,
        right_root=right_root_1,
        centerline=centerline,
        image_width=w,
        min_gap_px=max(8, int(w * 0.060)),
        fallback_half_width_ratio=expected_half_width_ratio
    )

    # ------------------------------------------------------------
    # Fill dark internal root area, then re-detect root boundary.
    # ------------------------------------------------------------

    root_like_filled, rough_root_mask = fill_internal_root_dark_regions(
        root_like=root_like,
        left_root=left_root_1,
        right_root=right_root_1
    )

    left_response_2 = build_root_boundary_response(
        img_uint8=enhanced,
        root_like=root_like_filled,
        img01=img01,
        edge_link=edge_link,
        side="left"
    )

    right_response_2 = build_root_boundary_response(
        img_uint8=enhanced,
        root_like=root_like_filled,
        img01=img01,
        edge_link=edge_link,
        side="right"
    )

    left_root = trace_boundary_dp(
        response=left_response_2,
        centerline=centerline,
        side="left",
        min_half_width_ratio=min_half_width_ratio,
        max_half_width_ratio=max_half_width_ratio,
        expected_half_width_ratio=expected_half_width_ratio
    )

    right_root = trace_boundary_dp(
        response=right_response_2,
        centerline=centerline,
        side="right",
        min_half_width_ratio=min_half_width_ratio,
        max_half_width_ratio=max_half_width_ratio,
        expected_half_width_ratio=expected_half_width_ratio
    )

    left_root, right_root = enforce_root_line_order(
        left_root=left_root,
        right_root=right_root,
        centerline=centerline,
        image_width=w,
        min_gap_px=max(8, int(w * 0.060)),
        fallback_half_width_ratio=expected_half_width_ratio
    )

    root_like_filled, filled_root_mask = fill_internal_root_dark_regions(
        root_like=root_like,
        left_root=left_root,
        right_root=right_root
    )

    # ------------------------------------------------------------
    # Yellow outer PDL / lamina dura boundary.
    # ------------------------------------------------------------

    left_outer, left_outer_score = detect_outer_pdl_boundary(
        img01=img01,
        edge_link=edge_link,
        root_boundary=left_root,
        side="left",
        pdl_min_px=pdl_min_px,
        pdl_max_px=pdl_max_px,
        lamina_search_px=lamina_search_px,
        fallback_pdl_width_px=fallback_pdl_width_px
    )

    right_outer, right_outer_score = detect_outer_pdl_boundary(
        img01=img01,
        edge_link=edge_link,
        root_boundary=right_root,
        side="right",
        pdl_min_px=pdl_min_px,
        pdl_max_px=pdl_max_px,
        lamina_search_px=lamina_search_px,
        fallback_pdl_width_px=fallback_pdl_width_px
    )

    left_outer, right_outer = enforce_outer_line_order(
        left_root=left_root,
        right_root=right_root,
        left_outer=left_outer,
        right_outer=right_outer,
        image_width=w,
        pdl_min_px=pdl_min_px,
        pdl_max_px=pdl_max_px,
        lamina_search_px=lamina_search_px
    )

    # ------------------------------------------------------------
    # Apex/end detection from yellow line convergence.
    # ------------------------------------------------------------

    end_y, root_profile, gap_profile = choose_lamina_convergence_end_y(
        left_outer=left_outer,
        right_outer=right_outer,
        left_root=left_root,
        right_root=right_root,
        root_like_filled=root_like_filled,
        min_end_ratio=min_end_ratio,
        max_end_ratio=max_end_ratio,
        convergence_ratio=convergence_ratio,
        end_padding_px=end_padding_px
    )

    # ------------------------------------------------------------
    # Taper both red and yellow lines smoothly to apex.
    # ------------------------------------------------------------

    left_root_tapered, right_root_tapered = taper_lines_to_apex(
        left_line=left_root,
        right_line=right_root,
        centerline=centerline,
        end_y=end_y,
        image_width=w,
        taper_ratio=root_taper_ratio,
        end_half_width_px=root_end_half_width_px
    )

    left_outer_tapered, right_outer_tapered = taper_lines_to_apex(
        left_line=left_outer,
        right_line=right_outer,
        centerline=centerline,
        end_y=end_y,
        image_width=w,
        taper_ratio=outer_taper_ratio,
        end_half_width_px=outer_end_half_width_px
    )

    # ------------------------------------------------------------
    # Masks.
    # ------------------------------------------------------------

    root_mask = create_polygon_mask(
        image_shape=img.shape,
        left_boundary=left_root_tapered,
        right_boundary=right_root_tapered,
        end_y=end_y
    )

    outer_mask = create_polygon_mask(
        image_shape=img.shape,
        left_boundary=left_outer_tapered,
        right_boundary=right_outer_tapered,
        end_y=end_y
    )

    pdl_mask = cv2.subtract(outer_mask, root_mask)

    final_mask = add_fixed_outer_space(
        outer_mask,
        outer_space_px=outer_space_px
    )

    final_mask = smooth_mask_edge(
        final_mask,
        smooth_px=edge_smooth_px
    )

    roi = cv2.bitwise_and(img, img, mask=final_mask)

    # ------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------

    safe_imwrite(mask_path, final_mask)
    safe_imwrite(roi_path, roi)
    safe_imwrite(root_mask_path, root_mask)
    safe_imwrite(pdl_mask_path, pdl_mask)
    safe_imwrite(outer_mask_path, outer_mask)

    if debug_path:
        debug = create_debug_image(
            original=img,
            enhanced=enhanced,
            root_like=root_like,
            root_like_filled=root_like_filled,
            edge_link=edge_link,
            root_mask=root_mask,
            pdl_mask=pdl_mask,
            outer_mask=outer_mask,
            final_mask=final_mask,
            left_root=left_root_tapered,
            right_root=right_root_tapered,
            left_outer=left_outer_tapered,
            right_outer=right_outer_tapered,
            centerline=centerline,
            end_y=end_y,
            root_profile=root_profile,
            gap_profile=gap_profile,
            left_outer_score=left_outer_score,
            right_outer_score=right_outer_score
        )

        safe_imwrite(debug_path, debug)

    print(
        f"Processed: {os.path.basename(image_path)} | "
        f"size={w}x{h} | end_y={end_y} | "
        f"root+PDL mask saved"
    )

    return {
        "final_mask": final_mask,
        "root_mask": root_mask,
        "pdl_mask": pdl_mask,
        "outer_mask": outer_mask,
        "roi": roi,
        "left_root": left_root_tapered,
        "right_root": right_root_tapered,
        "left_outer": left_outer_tapered,
        "right_outer": right_outer_tapered,
        "end_y": end_y
    }