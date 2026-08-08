import cv2
import numpy as np
import os


# ============================================================
# Utility functions
# ============================================================

def ensure_uint8_gray(img):
    """
    Ensure image is grayscale uint8.
    """

    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    return img


def safe_imwrite(path, image):
    """
    Save image safely.
    """

    dir_name = os.path.dirname(path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    cv2.imwrite(path, image)


def smooth_1d(signal, ksize=15):
    """
    Smooth 1D signal using Gaussian blur.
    """

    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(
        signal.reshape(-1, 1).astype(np.float32),
        (1, ksize),
        0
    ).flatten()


# ============================================================
# Step 1: Detect dark border-connected background
# ============================================================

def get_dark_border_component(img_uint8):
    """
    Detect large dark background region connected to image border.

    This is used only to decide where crown-search should begin.
    """

    h, w = img_uint8.shape

    p10 = int(np.percentile(img_uint8, 10))
    p20 = int(np.percentile(img_uint8, 20))

    dark_threshold = min(70, max(25, int((p10 + p20) / 2)))

    dark_mask = (img_uint8 <= dark_threshold).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, open_kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, 8)

    best_label = None
    best_area = 0

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        touches_border = (
            x <= 2 or
            y <= 2 or
            x + cw >= w - 2 or
            y + ch >= h - 2
        )

        if touches_border and area > best_area and area > h * w * 0.005:
            best_label = i
            best_area = area

    if best_label is None:
        return None

    border_component = (labels == best_label).astype(np.uint8) * 255

    return border_component


def detect_background_position(border_component):
    """
    Decide whether dark background is mainly at top or bottom.
    """

    h, w = border_component.shape

    top_score = np.sum(border_component[: int(h * 0.20), :] > 0)
    bottom_score = np.sum(border_component[int(h * 0.80):, :] > 0)

    if top_score >= bottom_score:
        return "top"

    return "bottom"


def extract_background_boundary_curve(border_component, position):
    """
    Extract the black-background / anatomy boundary curve.

    If background is top:
        lower edge of dark component.

    If background is bottom:
        upper edge of dark component.
    """

    h, w = border_component.shape
    boundary = np.full(w, np.nan, dtype=np.float32)

    for x in range(w):
        ys = np.where(border_component[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        if position == "top":
            boundary[x] = np.max(ys)
        else:
            boundary[x] = np.min(ys)

    valid = ~np.isnan(boundary)

    if np.sum(valid) < max(5, int(w * 0.15)):
        return None

    xs = np.arange(w)
    boundary = np.interp(xs, xs[valid], boundary[valid]).astype(np.float32)

    smooth_ksize = max(15, int(w * 0.18))

    if smooth_ksize % 2 == 0:
        smooth_ksize += 1

    boundary = cv2.GaussianBlur(
        boundary.reshape(1, -1),
        (smooth_ksize, 1),
        0
    ).flatten()

    boundary = np.clip(boundary, 0, h - 1)

    return boundary.astype(np.int32)


def get_robust_background_y(boundary, position):
    """
    Convert background boundary curve into stable horizontal reference.
    """

    if position == "top":
        return int(np.percentile(boundary, 70))

    return int(np.percentile(boundary, 30))


# ============================================================
# Step 2: Detect upper crown margin
# ============================================================

def detect_upper_crown_margin(
    img_uint8,
    start_y=0,
    central_width_ratio=0.75,
    bright_percentile=88,
    min_bright_ratio=0.16,
    consecutive_rows=6,
    smooth_ksize=17
):
    """
    Detect upper boundary of tooth crown by scanning from top to bottom.

    The first stable bright region is considered the upper crown margin.

    Important:
    - Uses central region only.
    - Uses dynamic brightness threshold.
    - Requires several consecutive bright rows to avoid noise.
    """

    h, w = img_uint8.shape

    # Focus only central part of ROI to avoid neighboring teeth / borders
    cx1 = int(w * (1.0 - central_width_ratio) / 2.0)
    cx2 = int(w * (1.0 + central_width_ratio) / 2.0)

    cx1 = max(0, cx1)
    cx2 = min(w, cx2)

    roi = img_uint8[:, cx1:cx2]

    # Mild smoothing
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    # Dynamic bright threshold
    bright_threshold = int(np.percentile(roi_blur, bright_percentile))

    bright_mask = (roi_blur >= bright_threshold).astype(np.uint8)

    # Row-wise bright ratio
    row_score = np.mean(bright_mask, axis=1)

    # Smooth row score
    row_score_smooth = smooth_1d(row_score, smooth_ksize)

    start_y = max(0, min(start_y, h - 1))

    upper_crown_y = None

    for y in range(start_y, h - consecutive_rows):
        window = row_score_smooth[y:y + consecutive_rows]

        if np.all(window >= min_bright_ratio):
            upper_crown_y = y
            break

    return upper_crown_y, row_score_smooth, bright_threshold, (cx1, cx2)


# ============================================================
# Step 3: Crop root using upper crown margin
# ============================================================

def crop_root_using_upper_crown_margin(
    img_uint8,
    upper_crown_y,
    crown_depth_ratio=0.18,
    min_crown_depth=30,
    max_crown_depth_ratio=0.28,
    min_output_height_ratio=0.38
):
    """
    Crop root + PDL + periapical bone region.

    upper_crown_y:
        detected upper boundary of crown.

    root_start_y:
        upper_crown_y + estimated crown depth.
    """

    h, w = img_uint8.shape

    crown_depth = int(h * crown_depth_ratio)
    crown_depth = max(crown_depth, min_crown_depth)

    max_crown_depth = int(h * max_crown_depth_ratio)
    crown_depth = min(crown_depth, max_crown_depth)

    root_start_y = upper_crown_y + crown_depth

    # Safety: keep enough root/periapical bone area
    max_start_y = int(h * (1.0 - min_output_height_ratio))
    root_start_y = min(root_start_y, max_start_y)

    root_start_y = max(0, min(root_start_y, h - 1))

    cropped = img_uint8[root_start_y:h, :]
    crop_box = (0, root_start_y, w, h)

    return cropped, crop_box, crown_depth, root_start_y


# ============================================================
# Fallback crop if crown margin is not detected
# ============================================================

def fallback_crop_using_background_boundary(
    img_uint8,
    background_y,
    position,
    fallback_shift_ratio=0.18,
    min_shift=35,
    min_output_height_ratio=0.38
):
    """
    Fallback crop method if upper crown margin detection fails.
    """

    h, w = img_uint8.shape

    shift = int(h * fallback_shift_ratio)
    shift = max(shift, min_shift)

    if position == "top":
        root_start_y = background_y + shift

        max_start_y = int(h * (1.0 - min_output_height_ratio))
        root_start_y = min(root_start_y, max_start_y)

        root_start_y = max(0, min(root_start_y, h - 1))

        cropped = img_uint8[root_start_y:h, :]
        crop_box = (0, root_start_y, w, h)

    else:
        root_end_y = background_y - shift

        min_end_y = int(h * min_output_height_ratio)
        root_end_y = max(root_end_y, min_end_y)

        root_end_y = max(1, min(root_end_y, h))

        cropped = img_uint8[0:root_end_y, :]
        crop_box = (0, 0, w, root_end_y)

        root_start_y = root_end_y

    return cropped, crop_box, shift, root_start_y


# ============================================================
# Debug visualization
# ============================================================

def create_debug_image(
    img_uint8,
    border_component=None,
    background_boundary=None,
    background_y=None,
    upper_crown_y=None,
    root_line_y=None,
    crop_box=None,
    central_region=None
):
    """
    Debug legend:

    Blue curve   = black-background / anatomy boundary
    Blue line    = stable background reference
    Red line     = detected upper crown margin
    Yellow line  = final root crop line
    Green box    = final root ROI
    Cyan lines   = central scan region
    Right image  = detected dark border component
    """

    debug = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    h, w = img_uint8.shape

    # Central scan region
    if central_region is not None:
        cx1, cx2 = central_region
        cv2.line(debug, (cx1, 0), (cx1, h - 1), (255, 255, 0), 1)
        cv2.line(debug, (cx2, 0), (cx2, h - 1), (255, 255, 0), 1)

    # Blue background boundary curve
    if background_boundary is not None:
        for x in range(w - 1):
            y1 = int(background_boundary[x])
            y2 = int(background_boundary[x + 1])
            cv2.line(debug, (x, y1), (x + 1, y2), (255, 0, 0), 2)

    # Blue background reference line
    if background_y is not None:
        cv2.line(
            debug,
            (0, int(background_y)),
            (w - 1, int(background_y)),
            (255, 0, 0),
            1
        )

    # Red upper crown margin
    if upper_crown_y is not None:
        cv2.line(
            debug,
            (0, int(upper_crown_y)),
            (w - 1, int(upper_crown_y)),
            (0, 0, 255),
            2
        )

    # Yellow final crop line
    if root_line_y is not None:
        cv2.line(
            debug,
            (0, int(root_line_y)),
            (w - 1, int(root_line_y)),
            (0, 255, 255),
            2
        )

    # Green crop box
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 0),
            2
        )

    if border_component is not None:
        component_vis = cv2.cvtColor(border_component, cv2.COLOR_GRAY2BGR)
    else:
        component_vis = np.zeros_like(debug)

    combined = np.hstack([debug, component_vis])

    return combined


# ============================================================
# Main extraction function
# ============================================================

def extract_anatomical_region(
    image_path,
    save_path=None,
    debug_path=None,

    # Crown-margin detection parameters
    central_width_ratio=0.75,
    bright_percentile=88,
    min_bright_ratio=0.16,
    consecutive_rows=6,

    # Crown removal depth
    crown_depth_ratio=0.18,
    min_crown_depth=30,
    max_crown_depth_ratio=0.28
):
    """
    Main anatomical root extraction pipeline.

    Input:
        selected tooth ROI

    Output:
        root + PDL + periapical bone ROI

    Method:
        1. Detect black-background edge.
        2. Start scanning below it.
        3. Detect upper crown margin using first stable bright region.
        4. Crop below estimated crown depth.
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = ensure_uint8_gray(img)

    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return None

    h, w = img.shape

    border_component = get_dark_border_component(img)

    background_position = "top"
    background_boundary = None
    background_y = 0

    if border_component is not None:
        background_position = detect_background_position(border_component)
        background_boundary = extract_background_boundary_curve(
            border_component,
            background_position
        )

        if background_boundary is not None:
            background_y = get_robust_background_y(
                background_boundary,
                background_position
            )

    # Search for crown only after black-background area
    if background_position == "top":
        search_start_y = background_y + 3
    else:
        search_start_y = 0

    upper_crown_y, row_score, bright_threshold, central_region = detect_upper_crown_margin(
        img_uint8=img,
        start_y=search_start_y,
        central_width_ratio=central_width_ratio,
        bright_percentile=bright_percentile,
        min_bright_ratio=min_bright_ratio,
        consecutive_rows=consecutive_rows,
        smooth_ksize=17
    )

    used_method = "upper_crown_margin"

    if upper_crown_y is not None:
        cropped, crop_box, crown_depth, root_line_y = crop_root_using_upper_crown_margin(
            img_uint8=img,
            upper_crown_y=upper_crown_y,
            crown_depth_ratio=crown_depth_ratio,
            min_crown_depth=min_crown_depth,
            max_crown_depth_ratio=max_crown_depth_ratio,
            min_output_height_ratio=0.38
        )
    else:
        used_method = "fallback_background_boundary"

        cropped, crop_box, crown_depth, root_line_y = fallback_crop_using_background_boundary(
            img_uint8=img,
            background_y=background_y,
            position=background_position,
            fallback_shift_ratio=0.18,
            min_shift=35,
            min_output_height_ratio=0.38
        )

    if save_path:
        safe_imwrite(save_path, cropped)

    if debug_path:
        debug_img = create_debug_image(
            img_uint8=img,
            border_component=border_component,
            background_boundary=background_boundary,
            background_y=background_y,
            upper_crown_y=upper_crown_y,
            root_line_y=root_line_y,
            crop_box=crop_box,
            central_region=central_region
        )

        safe_imwrite(debug_path, debug_img)

    print(
        f"Extracted: {os.path.basename(image_path)} | "
        f"method={used_method} | "
        f"background_y={background_y} | "
        f"upper_crown_y={upper_crown_y} | "
        f"crown_depth={crown_depth} | "
        f"root_line_y={root_line_y} | "
        f"bright_threshold={bright_threshold}"
    )

    return cropped