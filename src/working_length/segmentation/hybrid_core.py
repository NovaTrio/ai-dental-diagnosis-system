"""Hybrid global and row-adaptive distance-transform tooth core extraction."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def extract_row_adaptive_core(
    candidate_mask: np.ndarray,
    distance_map: np.ndarray,
    row_ratio: float = 0.55,
    expected_center_x: float | None = None,
) -> np.ndarray:
    height, width = candidate_mask.shape
    center_x = width / 2.0 if expected_center_x is None else expected_center_x
    half_width = int(0.40 * width)
    left = max(0, int(round(center_x)) - half_width)
    right = min(width, int(round(center_x)) + half_width + 1)
    core = np.zeros_like(candidate_mask, dtype=np.uint8)

    for y in range(height):
        foreground = candidate_mask[y, left:right] > 0
        if not np.any(foreground):
            continue
        distances = distance_map[y, left:right]
        row_maximum = float(distances[foreground].max())
        if row_maximum <= 0:
            continue
        keep = foreground & (distances >= row_ratio * row_maximum)
        core[y, left:right][keep] = 255

    return core


def select_reliable_core(
    core_mask: np.ndarray,
    distance_map: np.ndarray,
    expected_center_x: float | None = None,
) -> np.ndarray:
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (core_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    height, width = core_mask.shape
    center_x = width / 2.0 if expected_center_x is None else expected_center_x
    maximum_distance = max(float(distance_map.max()), 1e-6)
    best_label = None
    best_score = float("-inf")

    for label in range(1, label_count):
        x, y, box_width, box_height, area = stats[label]
        if area < max(10, int(0.0005 * core_mask.size)):
            continue
        centroid_x, _ = centroids[label]
        component = labels == label
        centrality = 1.0 - min(
            abs(centroid_x - center_x) / max(width / 2.0, 1.0),
            1.0,
        )
        vertical_span = box_height / height
        aspect_ratio = box_height / max(box_width, 1)
        normalized_depth = float(distance_map[component].mean()) / maximum_distance
        area_ratio = area / core_mask.size
        row_continuity = np.count_nonzero(np.any(component, axis=1)) / max(box_height, 1)
        border_penalty = 0.0
        if x == 0 or x + box_width >= width:
            border_penalty += 0.5
        if y == 0 or y + box_height >= height:
            border_penalty += 0.2

        score = (
            0.27 * centrality
            + 0.22 * min(vertical_span / 0.6, 1.0)
            + 0.17 * min(aspect_ratio / 4.0, 1.0)
            + 0.18 * normalized_depth
            + 0.06 * min(area_ratio / 0.10, 1.0)
            + 0.10 * row_continuity
            - border_penalty
        )
        if score > best_score:
            best_score = score
            best_label = label

    selected = np.zeros_like(core_mask, dtype=np.uint8)
    if best_label is not None:
        selected[labels == best_label] = 255
    return selected


def validate_core(core_mask: np.ndarray) -> bool:
    ys, xs = np.where(core_mask > 0)
    if len(xs) == 0:
        return False
    height, width = core_mask.shape
    vertical_span = (ys.max() - ys.min() + 1) / height
    horizontal_span = (xs.max() - xs.min() + 1) / width
    return vertical_span >= 0.25 and horizontal_span <= 0.80


def extract_reliable_tooth_core(
    candidate_mask: np.ndarray,
    expected_center_x: float | None = None,
) -> dict[str, np.ndarray]:
    binary = (candidate_mask > 0).astype(np.uint8)
    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    empty = np.zeros_like(candidate_mask, dtype=np.uint8)
    if distance_map.max() <= 0:
        return {
            "distance_map": distance_map,
            "global_core": empty,
            "row_core": empty,
            "combined_core": empty,
            "selected_core": empty,
        }

    positive = distance_map[distance_map > 0]
    global_threshold = max(
        0.35 * float(distance_map.max()),
        float(np.percentile(positive, 65)),
    )
    global_core = (distance_map >= global_threshold).astype(np.uint8) * 255
    row_core = extract_row_adaptive_core(
        candidate_mask,
        distance_map,
        row_ratio=0.55,
        expected_center_x=expected_center_x,
    )
    combined = cv2.bitwise_or(global_core, row_core)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9)),
        iterations=1,
    )
    selected = select_reliable_core(
        combined,
        distance_map,
        expected_center_x,
    )
    if not validate_core(selected):
        fallback = select_reliable_core(global_core, distance_map, expected_center_x)
        if cv2.countNonZero(fallback) > 0:
            selected = fallback

    return {
        "distance_map": distance_map,
        "global_core": global_core,
        "row_core": row_core,
        "combined_core": combined,
        "selected_core": selected,
    }


def reconstruct_from_core(
    core_mask: np.ndarray,
    candidate_mask: np.ndarray,
    max_iterations: int = 300,
) -> np.ndarray:
    marker = (core_mask > 0).astype(np.uint8)
    constraint = (candidate_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_iterations):
        previous = marker.copy()
        marker = cv2.dilate(marker, kernel, iterations=1)
        marker = cv2.bitwise_and(marker, constraint)
        if np.array_equal(marker, previous):
            break
    return marker * 255


def core_guided_region_growing(
    gray: np.ndarray,
    core_mask: np.ndarray,
    candidate_mask: np.ndarray,
    distance_map: np.ndarray | None = None,
    expected_center_x: float | None = None,
    core_tolerance: float = 35.0,
    region_tolerance: float = 20.0,
    local_tolerance: float = 20.0,
) -> dict[str, np.ndarray]:
    """Grow the core using global, dynamic, local, edge and mask constraints."""
    if gray.ndim != 2:
        raise ValueError("gray must be a grayscale image")
    if gray.shape != core_mask.shape or gray.shape != candidate_mask.shape:
        raise ValueError("All inputs must have identical dimensions")

    gray_float = gray.astype(np.float32)
    height, width = gray.shape
    seed = (core_mask > 0) & (candidate_mask > 0)
    candidate = candidate_mask > 0
    empty = np.zeros_like(gray, dtype=np.uint8)
    if not np.any(seed):
        return {
            "intensity_allowed": empty,
            "gradient_map": empty,
            "edge_allowed": empty,
            "geometry_allowed": empty,
            "distance_confidence": empty,
            "grown_mask": empty,
        }

    core_values = gray_float[seed]
    initial_mean = float(core_values.mean())
    initial_std = float(core_values.std())
    anchored_tolerance = float(
        np.clip(max(core_tolerance, 1.5 * initial_std), 25.0, 55.0)
    )
    dynamic_tolerance = float(np.clip(region_tolerance, 12.0, 40.0))
    neighbourhood_tolerance = float(np.clip(local_tolerance, 10.0, 35.0))

    intensity_allowed = (
        candidate
        & (gray_float >= initial_mean - anchored_tolerance)
        & (gray_float <= initial_mean + anchored_tolerance)
    )

    gradient_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    candidate_gradients = gradient[candidate]
    gradient_limit = (
        float(np.percentile(candidate_gradients, 80))
        if candidate_gradients.size
        else 80.0
    )
    gradient_limit = float(np.clip(gradient_limit, 50.0, 120.0))
    edge_allowed = gradient <= gradient_limit
    local_mean = cv2.blur(gray_float, (5, 5))

    if expected_center_x is None:
        expected_center_x = float(np.median(np.where(seed)[1]))

    # Infer which end is broader so crown/root orientation is not hard-coded.
    row_widths = np.zeros(height, dtype=np.float32)
    for row in range(height):
        row_x = np.where(candidate[row])[0]
        if len(row_x):
            row_widths[row] = row_x.max() - row_x.min() + 1
    section = max(1, int(0.35 * height))
    top_values = row_widths[:section][row_widths[:section] > 0]
    bottom_values = row_widths[-section:][row_widths[-section:] > 0]
    top_width = float(np.median(top_values)) if top_values.size else 0.0
    bottom_width = float(np.median(bottom_values)) if bottom_values.size else 0.0
    crown_at_top = top_width >= bottom_width

    geometry_allowed = np.zeros_like(candidate)
    root_half_width = 0.25 * width
    crown_half_width = 0.45 * width
    for row in range(height):
        relative_y = row / max(height - 1, 1)
        crown_weight = 1.0 - relative_y if crown_at_top else relative_y
        half_width = (
            root_half_width
            + crown_weight * (crown_half_width - root_half_width)
        )
        left = max(0, int(round(expected_center_x - half_width)))
        right = min(width, int(round(expected_center_x + half_width + 1)))
        geometry_allowed[row, left:right] = True

    if distance_map is None or distance_map.shape != gray.shape:
        distance_map = cv2.distanceTransform(
            candidate.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
    distance_confidence = cv2.normalize(
        distance_map,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
    )
    allowed = candidate & intensity_allowed & edge_allowed & geometry_allowed

    grown = np.zeros_like(gray, dtype=np.uint8)
    visited = np.zeros_like(gray, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    seed_y, seed_x = np.where(seed)

    for y, x in zip(seed_y, seed_x):
        queue.append((int(y), int(x)))
        grown[y, x] = 1
        visited[y, x] = True

    region_sum = float(core_values.sum())
    region_count = int(core_values.size)
    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    )

    while queue:
        y, x = queue.popleft()
        region_mean = region_sum / max(region_count, 1)

        for delta_y, delta_x in neighbours:
            neighbour_y = y + delta_y
            neighbour_x = x + delta_x
            if not (0 <= neighbour_y < height and 0 <= neighbour_x < width):
                continue
            if visited[neighbour_y, neighbour_x]:
                continue
            visited[neighbour_y, neighbour_x] = True
            if not allowed[neighbour_y, neighbour_x]:
                continue

            pixel_value = float(gray_float[neighbour_y, neighbour_x])
            confidence = float(distance_confidence[neighbour_y, neighbour_x])
            # Interior pixels receive a larger tolerance; boundary pixels must
            # agree more strongly. The sign is positive by design.
            distance_tolerance = 12.0 + 25.0 * confidence
            accepted_core_tolerance = min(anchored_tolerance, distance_tolerance)
            accepted_region_tolerance = min(dynamic_tolerance, distance_tolerance)
            accepted_local_tolerance = min(neighbourhood_tolerance, distance_tolerance)

            if abs(pixel_value - initial_mean) > accepted_core_tolerance:
                continue
            if abs(pixel_value - region_mean) > accepted_region_tolerance:
                continue
            if (
                abs(pixel_value - float(local_mean[neighbour_y, neighbour_x]))
                > accepted_local_tolerance
            ):
                continue

            grown[neighbour_y, neighbour_x] = 1
            queue.append((neighbour_y, neighbour_x))
            region_sum += pixel_value
            region_count += 1

    gradient_debug = cv2.normalize(
        gradient,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    return {
        "intensity_allowed": intensity_allowed.astype(np.uint8) * 255,
        "gradient_map": gradient_debug,
        "edge_allowed": edge_allowed.astype(np.uint8) * 255,
        "geometry_allowed": geometry_allowed.astype(np.uint8) * 255,
        "distance_confidence": (distance_confidence * 255).astype(np.uint8),
        "grown_mask": grown * 255,
    }


def proximity_constrained_completion(
    dynamic_mask: np.ndarray,
    morphological_mask: np.ndarray,
    width_ratio: float = 0.30,
) -> np.ndarray:
    """Recover nearby candidate pixels without permitting unlimited leakage."""
    iterations = max(3, int(round(dynamic_mask.shape[1] * width_ratio)))
    neighbourhood = cv2.dilate(
        (dynamic_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=iterations,
    )
    nearby_candidate = cv2.bitwise_and(
        (morphological_mask > 0).astype(np.uint8) * 255,
        neighbourhood,
    )
    return cv2.bitwise_or(
        (dynamic_mask > 0).astype(np.uint8) * 255,
        nearby_candidate,
    )


def layered_core_growth(
    gray: np.ndarray,
    core_mask: np.ndarray,
    candidate_mask: np.ndarray,
    max_iterations: int = 200,
) -> np.ndarray:
    """Grow complete morphological layers under a core-intensity constraint."""
    if gray.ndim != 2:
        raise ValueError("gray must be a grayscale image")
    if gray.shape != core_mask.shape or gray.shape != candidate_mask.shape:
        raise ValueError("All inputs must have identical dimensions")

    candidate = (candidate_mask > 0).astype(np.uint8)
    grown = ((core_mask > 0) & (candidate > 0)).astype(np.uint8)
    if cv2.countNonZero(grown) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    core_values = gray[grown > 0].astype(np.float32)
    core_mean = float(core_values.mean())
    core_std = float(core_values.std())
    # Retain a minimum range for nearly uniform seeds while keeping the range
    # anchored to the original reliable core.
    intensity_half_range = float(np.clip(2.5 * core_std, 20.0, 70.0))
    lower = max(0.0, core_mean - intensity_half_range)
    upper = min(255.0, core_mean + intensity_half_range)
    intensity_allowed = ((gray >= lower) & (gray <= upper)).astype(np.uint8)
    allowed = cv2.bitwise_and(candidate, intensity_allowed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_iterations):
        previous = grown.copy()
        frontier = cv2.dilate(grown, kernel, iterations=1)
        frontier = cv2.bitwise_and(frontier, allowed)
        grown = cv2.bitwise_or(grown, frontier)
        if np.array_equal(grown, previous):
            break

    return grown * 255
