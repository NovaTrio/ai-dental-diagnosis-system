"""Extract a branch-free, smoothed centreline from a root-canal mask."""

from __future__ import annotations

import heapq
import math

import cv2
import numpy as np
from scipy.interpolate import splprep, splev


NEIGHBOURS = (
    (-1, -1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, 1, 1.0),
    (1, 1, math.sqrt(2.0)),
)


def detect_bright_canal_material(
    image: np.ndarray,
    tooth_mask: np.ndarray,
    dark_canal_mask: np.ndarray,
) -> np.ndarray:
    """Detect narrow central radiopaque filling or instruments in the canal."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    interior_values = gray[tooth_mask > 0]
    if interior_values.size == 0:
        return np.zeros_like(gray)

    threshold = max(180.0, float(np.percentile(interior_values, 88)))
    bright = np.logical_and(gray >= threshold, tooth_mask > 0).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7)),
        iterations=1,
    )

    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (bright > 0).astype(np.uint8),
        connectivity=8,
    )
    height, width = bright.shape
    tooth_x = np.where(tooth_mask > 0)[1]
    expected_x = float(np.median(tooth_x)) if tooth_x.size else width / 2.0
    dark_dilated = cv2.dilate(
        dark_canal_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    selected = np.zeros_like(bright)
    for label in range(1, label_count):
        x, _, box_width, box_height, area = (int(v) for v in stats[label])
        center_x = float(centroids[label, 0])
        centrality = 1.0 - min(
            abs(center_x - expected_x) / max(width / 2.0, 1.0),
            1.0,
        )
        verticality = box_height / max(box_width, 1)
        overlaps_dark = bool(
            np.any(np.logical_and(labels == label, dark_dilated > 0))
        )
        canal_like = (
            area >= 5
            and box_height >= max(7, int(0.05 * height))
            and box_width <= max(8, int(0.22 * width))
            and verticality >= 1.8
            and centrality >= 0.45
        )
        if canal_like and (overlaps_dark or centrality >= 0.72):
            selected[labels == label] = 255
    return selected


def combine_canal_evidence(
    dark_canal_mask: np.ndarray,
    bright_material_mask: np.ndarray,
    tooth_mask: np.ndarray,
) -> np.ndarray:
    combined = cv2.bitwise_or(dark_canal_mask, bright_material_mask)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9)),
        iterations=1,
    )
    return cv2.bitwise_and(combined, tooth_mask)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Create a connectivity-preserving one-pixel Zhang-Suen skeleton."""
    image = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for first_step in (True, False):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = np.sum(
                np.stack(
                    (
                        (p2 == 0) & (p3 == 1),
                        (p3 == 0) & (p4 == 1),
                        (p4 == 0) & (p5 == 1),
                        (p5 == 0) & (p6 == 1),
                        (p6 == 0) & (p7 == 1),
                        (p7 == 0) & (p8 == 1),
                        (p8 == 0) & (p9 == 1),
                        (p9 == 0) & (p2 == 1),
                    )
                ),
                axis=0,
                dtype=np.uint8,
            )
            if first_step:
                triplet_a = p2 * p4 * p6
                triplet_b = p4 * p6 * p8
            else:
                triplet_a = p2 * p4 * p8
                triplet_b = p2 * p6 * p8
            remove = (
                (image == 1)
                & (neighbours >= 2)
                & (neighbours <= 6)
                & (transitions == 1)
                & (triplet_a == 0)
                & (triplet_b == 0)
            )
            if np.any(remove):
                image[remove] = 0
                changed = True
    return image * 255


def _graph(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], float]]]:
    ys, xs = np.where(skeleton > 0)
    nodes = {(int(x), int(y)) for x, y in zip(xs, ys)}
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {
        node: [] for node in nodes
    }
    for x, y in nodes:
        for dx, dy, weight in NEIGHBOURS:
            neighbour = (x + dx, y + dy)
            if neighbour in nodes:
                graph[(x, y)].append((neighbour, weight))
    return graph


def _shortest_paths(
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], float]]],
    start: tuple[int, int],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]]]:
    distances = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                parents[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return distances, parents


def longest_skeleton_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Return the graph diameter path, discarding all shorter branches."""
    graph = _graph(skeleton)
    if not graph:
        return []
    components: list[set[tuple[int, int]]] = []
    unseen = set(graph)
    while unseen:
        start = next(iter(unseen))
        component = {start}
        stack = [start]
        unseen.remove(start)
        while stack:
            node = stack.pop()
            for neighbour, _ in graph[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    component = max(components, key=len)
    subgraph = {
        node: [(n, w) for n, w in graph[node] if n in component]
        for node in component
    }
    endpoints = [node for node in component if len(subgraph[node]) == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(component)

    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_distance = -1.0
    best_parents: dict[tuple[int, int], tuple[int, int]] = {}
    for start in candidates:
        distances, parents = _shortest_paths(subgraph, start)
        finish = max(candidates, key=lambda node: distances.get(node, -1.0))
        if distances.get(finish, -1.0) > best_distance:
            best_distance = distances[finish]
            best_pair = (start, finish)
            best_parents = parents
    if best_pair is None:
        return [next(iter(component))]

    start, node = best_pair
    path = [node]
    while node != start:
        node = best_parents[node]
        path.append(node)
    path.reverse()
    if path[0][1] > path[-1][1]:
        path.reverse()
    return path


def smooth_path(
    path: list[tuple[int, int]],
    smoothing: float = 3.0,
) -> list[tuple[float, float]]:
    """Fit a parametric spline so curved canals need not be x=f(y)."""
    if len(path) < 4:
        return [(float(x), float(y)) for x, y in path]
    points = np.asarray(path, dtype=np.float64)
    keep = np.concatenate(([True], np.any(np.diff(points, axis=0) != 0, axis=1)))
    points = points[keep]
    if len(points) < 4:
        return [(float(x), float(y)) for x, y in points]
    try:
        spline, _ = splprep(
            [points[:, 0], points[:, 1]],
            s=smoothing * len(points),
            k=min(3, len(points) - 1),
        )
        parameter = np.linspace(0.0, 1.0, len(points))
        smooth_x, smooth_y = splev(parameter, spline)
        return list(zip(np.asarray(smooth_x), np.asarray(smooth_y)))
    except (TypeError, ValueError):
        return [(float(x), float(y)) for x, y in points]


def path_length(path: list[tuple[float, float]]) -> float:
    return float(
        sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(path, path[1:])
        )
    )


def path_to_mask(
    shape: tuple[int, int],
    path: list[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pixels = [
        (
            int(np.clip(round(x), 0, shape[1] - 1)),
            int(np.clip(round(y), 0, shape[0] - 1)),
        )
        for x, y in path
    ]
    for first, second in zip(pixels, pixels[1:]):
        cv2.line(mask, first, second, 255, 1, cv2.LINE_AA)
    if len(pixels) == 1:
        mask[pixels[0][1], pixels[0][0]] = 255
    return mask


def extract_canal_midline(
    image: np.ndarray,
    dark_canal_mask: np.ndarray,
    tooth_mask: np.ndarray | None = None,
    *,
    smoothing: float = 3.0,
) -> dict[str, object]:
    """Combine dark/bright canal evidence and extract its longest centreline."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if tooth_mask is None:
        tooth_mask = (gray > 0).astype(np.uint8) * 255
    bright = detect_bright_canal_material(gray, tooth_mask, dark_canal_mask)
    combined = combine_canal_evidence(dark_canal_mask, bright, tooth_mask)
    skeleton = skeletonize(combined)
    raw_path = longest_skeleton_path(skeleton)
    smooth = smooth_path(raw_path, smoothing)
    midline_mask = path_to_mask(gray.shape, smooth)
    length = path_length(smooth)
    tooth_rows = np.flatnonzero(np.any(tooth_mask > 0, axis=1))
    tooth_height = (
        int(tooth_rows[-1] - tooth_rows[0] + 1) if tooth_rows.size else gray.shape[0]
    )
    minimum_length = max(10.0, 0.15 * tooth_height)
    detected = len(smooth) >= 2 and length >= minimum_length
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[midline_mask > 0] = (0, 0, 255)
    return {
        "bright_material_mask": bright,
        "combined_canal_mask": combined,
        "skeleton": skeleton,
        "raw_path": raw_path,
        "midline_path": smooth,
        "midline_mask": midline_mask,
        "midline_length_px": length,
        "minimum_valid_length_px": minimum_length,
        "overlay": overlay,
        "detected": detected,
        "validation_reason": "valid" if detected else "path_too_short",
    }
