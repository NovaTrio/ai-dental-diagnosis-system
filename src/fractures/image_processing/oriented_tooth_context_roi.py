import cv2
import numpy as np
import os


def smooth_1d(signal, ksize=31):
    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(
        signal.reshape(1, -1).astype(np.float32),
        (ksize, 1),
        0
    ).flatten()


def estimate_orientation_pca(img_uint8):
    """
    Estimate dominant tooth/root direction using PCA on edge pixels.
    Returns angle in degrees.
    Vertical root direction should be close to 90 degrees.
    """

    blur = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    ys, xs = np.where(edges > 0)

    if len(xs) < 30:
        return 90.0, edges

    points = np.column_stack((xs, ys)).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(points, mean=None)

    vx, vy = eigenvectors[0]

    angle = np.degrees(np.arctan2(vy, vx))

    # Convert to 0-180 range
    if angle < 0:
        angle += 180

    # We care about root-like near-vertical direction
    if angle < 45:
        angle += 90

    if angle > 135:
        angle -= 90

    return angle, edges


def rotate_image(img_uint8, angle):
    """
    Rotate image so dominant root axis becomes vertical.
    """

    h, w = img_uint8.shape
    center = (w // 2, h // 2)

    rotation_angle = 90 - angle

    matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img_uint8,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated, matrix, rotation_angle


def find_root_context_zones(rotated_img):
    """
    After rotation, roots are approximately vertical.
    Use projection to find tooth/root context zones.
    """

    h, w = rotated_img.shape

    blur = cv2.GaussianBlur(rotated_img, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)

    intensity_projection = np.mean(blur, axis=0)
    edge_projection = np.sum(edges > 0, axis=0).astype(np.float32)

    intensity_projection = (
        intensity_projection - intensity_projection.min()
    ) / (intensity_projection.max() - intensity_projection.min() + 1e-6)

    edge_projection = (
        edge_projection - edge_projection.min()
    ) / (edge_projection.max() - edge_projection.min() + 1e-6)

    score = 0.55 * intensity_projection + 0.45 * edge_projection
    score = smooth_1d(score, 31)

    threshold = np.percentile(score, 45)

    active_cols = score > threshold

    # Smooth column activation
    active_cols = np.convolve(
        active_cols.astype(np.uint8),
        np.ones(11, dtype=np.uint8),
        mode="same"
    ) > 0

    zones = []
    start = None

    for x in range(w):
        if active_cols[x] and start is None:
            start = x

        if (not active_cols[x] or x == w - 1) and start is not None:
            end = x

            if end - start > 0.07 * w:
                zones.append((start, end))

            start = None

    return zones, score, edges


def split_merged_zones(rotated_img, zones):
    """
    Split wide zones using dark vertical valleys.
    """

    h, w = rotated_img.shape
    refined = []

    for x1, x2 in zones:
        zone_w = x2 - x1

        if zone_w < 0.30 * w:
            refined.append((x1, x2))
            continue

        roi = rotated_img[:, x1:x2]

        col_mean = np.mean(roi, axis=0)
        col_mean = smooth_1d(col_mean, 25)

        valley_threshold = np.percentile(col_mean, 35)
        valleys = np.where(col_mean < valley_threshold)[0]

        split_points = []

        if len(valleys) > 0:
            groups = np.split(
                valleys,
                np.where(np.diff(valleys) > 5)[0] + 1
            )

            for group in groups:
                if len(group) < 5:
                    continue

                sx = int(np.mean(group))

                if 0.20 * zone_w < sx < 0.80 * zone_w:
                    split_points.append(sx)

        split_points = sorted(list(set(split_points)))

        if len(split_points) == 0:
            refined.append((x1, x2))
        else:
            bounds = [0] + split_points + [zone_w]

            for i in range(len(bounds) - 1):
                sx1 = bounds[i]
                sx2 = bounds[i + 1]

                if sx2 - sx1 > 0.07 * w:
                    refined.append((x1 + sx1, x1 + sx2))

    return refined


def create_context_rois(rotated_img, zones):
    """
    Create diagnostic ROIs containing:
    tooth/root + PDL + surrounding bone.
    """

    h, w = rotated_img.shape

    rois = []
    boxes = []

    for x1, x2 in zones:
        zone_w = x2 - x1

        margin_x = int(0.30 * zone_w)
        margin_y = int(0.05 * h)

        cx1 = max(0, x1 - margin_x)
        cx2 = min(w, x2 + margin_x)

        cy1 = margin_y
        cy2 = h - margin_y

        roi = rotated_img[cy1:cy2, cx1:cx2]

        if roi.shape[0] < 40 or roi.shape[1] < 25:
            continue

        rois.append(roi)
        boxes.append((cx1, cy1, cx2, cy2))

    return rois, boxes


def create_debug_image(original, rotated, edges_before, edges_after, boxes, zones, score, angle):
    """
    Debug output:
    original edges, rotated image with boxes, projection, rotated edges.
    """

    h, w = rotated.shape

    rotated_debug = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(rotated_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            rotated_debug,
            f"ROI {idx + 1}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    for x1, x2 in zones:
        cv2.line(rotated_debug, (x1, 0), (x1, h - 1), (0, 0, 255), 1)
        cv2.line(rotated_debug, (x2, 0), (x2, h - 1), (0, 0, 255), 1)

    cv2.putText(
        rotated_debug,
        f"Estimated angle: {angle:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    projection_img = np.zeros_like(rotated)

    score_scaled = (score * (h - 1)).astype(np.int32)

    for x in range(min(len(score_scaled) - 1, w - 1)):
        y1 = h - 1 - score_scaled[x]
        y2 = h - 1 - score_scaled[x + 1]
        cv2.line(projection_img, (x, y1), (x + 1, y2), 255, 1)

    original_vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    edges_before_vis = cv2.cvtColor(edges_before, cv2.COLOR_GRAY2BGR)
    projection_vis = cv2.cvtColor(projection_img, cv2.COLOR_GRAY2BGR)
    edges_after_vis = cv2.cvtColor(edges_after, cv2.COLOR_GRAY2BGR)

    # Resize original-side images to rotated height
    original_vis = cv2.resize(original_vis, (w, h))
    edges_before_vis = cv2.resize(edges_before_vis, (w, h))

    combined = np.hstack([
        original_vis,
        edges_before_vis,
        rotated_debug,
        projection_vis,
        edges_after_vis
    ])

    return combined


def extract_oriented_tooth_context_rois(image_path, output_dir=None, debug_path=None):
    """
    Main function.

    Input:
    anatomical ROI image

    Output:
    rotated tooth/root context ROIs containing:
    tooth + PDL + surrounding bone
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    angle, edges_before = estimate_orientation_pca(img)

    rotated, matrix, rotation_angle = rotate_image(img, angle)

    zones, score, edges_after = find_root_context_zones(rotated)

    zones = split_merged_zones(rotated, zones)

    rois, boxes = create_context_rois(rotated, zones)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        for idx, roi in enumerate(rois):
            save_path = os.path.join(output_dir, f"oriented_context_roi_{idx + 1}.png")
            cv2.imwrite(save_path, roi)

    if debug_path:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        debug_img = create_debug_image(
            original=img,
            rotated=rotated,
            edges_before=edges_before,
            edges_after=edges_after,
            boxes=boxes,
            zones=zones,
            score=score,
            angle=angle
        )

        cv2.imwrite(debug_path, debug_img)

    return rois, boxes, angle