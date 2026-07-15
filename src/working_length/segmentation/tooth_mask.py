"""Classical image-processing algorithms for constructing a tooth mask."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_roi(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)


def adaptive_tooth_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -2,
    )


def clear_side_margins(binary: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
    width = binary.shape[1]
    margin = int(width * margin_ratio)
    cleaned = binary.copy()
    cleaned[:, :margin] = 0
    cleaned[:, width - margin:] = 0
    return cleaned


def morphology_cleanup(binary: np.ndarray) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)


def suppress_top_broad_artifacts(binary: np.ndarray) -> np.ndarray:
    """Restrict broad upper clamp/crown regions to the estimated root corridor."""
    height, width = binary.shape
    lower_widths: list[int] = []
    lower_centers: list[float] = []

    for y in range(int(0.35 * height), height):
        xs = np.where(binary[y, :] > 0)[0]
        if len(xs) >= 5:
            lower_widths.append(int(xs.max() - xs.min() + 1))
            lower_centers.append(float((xs.min() + xs.max()) / 2.0))

    if len(lower_widths) < 5:
        return binary

    root_center = float(np.median(lower_centers))
    root_width = float(np.percentile(lower_widths, 70))
    if root_width <= 0:
        return binary

    corridor_half_width = max(0.80 * root_width, 0.12 * width, 8)
    corridor_left = max(0, int(round(root_center - corridor_half_width)))
    corridor_right = min(width, int(round(root_center + corridor_half_width + 1)))
    wide_row_threshold = max(1.55 * root_width, 0.35 * width)
    cleaned = binary.copy()

    for y in range(int(0.55 * height)):
        xs = np.where(cleaned[y, :] > 0)[0]
        if len(xs) >= 5 and xs.max() - xs.min() + 1 >= wide_row_threshold:
            row = np.zeros(width, dtype=np.uint8)
            row[corridor_left:corridor_right] = cleaned[y, corridor_left:corridor_right]
            cleaned[y, :] = row

    return cleaned


def extract_tooth_core(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract a stable core using a robust, ROI-adaptive distance threshold."""
    distance_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if distance_transform.max() == 0:
        return np.zeros_like(binary), distance_transform

    positive_distances = distance_transform[distance_transform > 0]
    robust_threshold = float(np.percentile(positive_distances, 65))
    maximum_threshold = 0.30 * float(distance_transform.max())
    core_threshold = max(1.0, min(robust_threshold, maximum_threshold))

    _, core = cv2.threshold(
        distance_transform,
        core_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    core = core.astype(np.uint8)
    core = cv2.morphologyEx(
        core,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return core, distance_transform


def normalize_distance_transform(distance_transform: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        distance_transform,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)


def _component_confidence(
    stats: np.ndarray,
    centroid: np.ndarray,
    image_shape: tuple[int, int],
    component_mask: np.ndarray,
) -> float | None:
    x, y, box_width, box_height, area = stats
    height, width = image_shape

    if area < 20 or box_width < max(6, int(0.025 * width)):
        return None

    center_x, _ = centroid
    center_score = np.clip(
        1.0 - abs(center_x - width / 2.0) / (width / 2.0),
        0.0,
        1.0,
    )
    height_score = np.clip(box_height / (0.70 * height), 0.0, 1.0)
    bottom_score = np.clip((y + box_height) / height, 0.0, 1.0)
    vertical_score = np.clip(box_height / (2.5 * box_width + 1), 0.0, 1.0)
    area_score = np.clip(area / (0.18 * height * width), 0.0, 1.0)
    row_occupancy = np.count_nonzero(np.any(component_mask, axis=1)) / max(1, box_height)
    fill_score = np.clip(area / max(1, box_width * box_height), 0.0, 1.0)
    aspect_ratio = max(box_width, box_height) / (min(box_width, box_height) + 1)

    shape_penalty = 0.0
    if aspect_ratio > 6.0:
        shape_penalty += 0.35
    if box_width < max(10, int(0.05 * width)) and box_height > 0.35 * height:
        shape_penalty += 0.35
    if box_width > 0.55 * width and y < 0.35 * height:
        shape_penalty += 0.45

    confidence = (
        0.25 * center_score
        + 0.18 * height_score
        + 0.14 * bottom_score
        + 0.14 * vertical_score
        + 0.10 * area_score
        + 0.10 * row_occupancy
        + 0.09 * fill_score
        - shape_penalty
    )
    return float(np.clip(confidence, 0.0, 1.0))


def select_central_component(
    core_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score core components and return the most tooth-like central component."""
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        core_mask,
        connectivity=8,
    )
    selected = np.zeros_like(core_mask)
    visualization = cv2.cvtColor(core_mask, cv2.COLOR_GRAY2BGR)
    best_label: int | None = None
    best_score = float("-inf")
    scores: dict[int, float] = {}

    for label in range(1, label_count):
        score = _component_confidence(
            stats[label],
            centroids[label],
            core_mask.shape,
            labels == label,
        )
        if score is None:
            continue
        scores[label] = score
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is not None:
        selected[labels == best_label] = 255

    for label, score in scores.items():
        x, y, box_width, box_height, _ = stats[label]
        color = (0, 255, 0) if label == best_label else (0, 165, 255)
        cv2.rectangle(
            visualization,
            (int(x), int(y)),
            (int(x + box_width - 1), int(y + box_height - 1)),
            color,
            1,
        )
        cv2.putText(
            visualization,
            f"{label}:{score:.2f}",
            (int(x), max(10, int(y) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )

    return selected, visualization


def fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def geometry_constrained_region_expansion(
    selected_core: np.ndarray,
    candidate_mask: np.ndarray,
    boundary_inset_ratio: float = 0.15,
) -> np.ndarray:
    """Expand a ranked core into a vertically coherent, tooth-shaped region.

    Candidate foreground is evaluated row by row inside a corridor centered on
    the selected distance-transform core. The outermost responses estimate the
    anatomical envelope, while a small inset prevents expansion into adjacent
    bone and ROI borders.
    """
    height, width = candidate_mask.shape
    core_y, core_x = np.where(selected_core > 0)

    if len(core_x):
        center_x = float(np.median(core_x))
        core_width = int(core_x.max() - core_x.min() + 1)
    else:
        center_x = width / 2.0
        core_width = max(6, int(0.15 * width))

    corridor_half_width = min(
        int(0.46 * width),
        max(int(0.35 * width), int(2.5 * core_width)),
    )
    corridor_left = max(0, int(round(center_x - corridor_half_width)))
    corridor_right = min(width, int(round(center_x + corridor_half_width + 1)))
    expanded = np.zeros_like(candidate_mask)

    for y in range(height):
        row_x = np.where(candidate_mask[y, corridor_left:corridor_right] > 0)[0]
        if len(row_x) < 3:
            continue

        left = int(row_x.min() + corridor_left)
        right = int(row_x.max() + corridor_left)
        inset = int(round((right - left + 1) * boundary_inset_ratio))
        left += inset
        right -= inset
        if right >= left:
            expanded[y, left:right + 1] = 255

    expanded = cv2.bitwise_or(expanded, selected_core)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
    return cv2.morphologyEx(
        expanded,
        cv2.MORPH_CLOSE,
        vertical_kernel,
        iterations=1,
    )


def refine_tooth_boundary(expanded_mask: np.ndarray) -> np.ndarray:
    """Smooth row boundaries, close small gaps, and produce a binary mask."""
    if cv2.countNonZero(expanded_mask) == 0:
        return expanded_mask

    refined = cv2.medianBlur(expanded_mask, 5)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )
    refined = fill_holes(refined)

    # Remove only very small disconnected regions; preserve separated roots.
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined,
        connectivity=8,
    )
    minimum_area = max(12, int(0.005 * refined.size))
    cleaned = np.zeros_like(refined)
    for label in range(1, label_count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255

    return cleaned


def crop_tooth_region(
    image: np.ndarray,
    mask: np.ndarray,
    padding_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image, mask, (0, 0)

    height, width = image.shape
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    pad_x = int((x_max - x_min + 1) * padding_ratio)
    pad_y = int((y_max - y_min + 1) * padding_ratio)
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(width, x_max + pad_x + 1)
    y1 = min(height, y_max + pad_y + 1)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], (x0, y0)


def remove_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)


def make_tooth_white(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def mask_from_isolated_tooth(image: np.ndarray) -> np.ndarray:
    """Build a binary mask from an image whose background is already black."""
    return (image > 0).astype(np.uint8) * 255
