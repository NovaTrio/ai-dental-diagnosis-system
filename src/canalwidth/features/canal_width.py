"""Measure apical canal width perpendicular to a smoothed canal centreline."""

from __future__ import annotations

import math

import cv2
import numpy as np


DEFAULT_FRACTIONS = (0.95, 0.96, 0.97, 0.98, 0.99)


def _arc_length_path(
    path: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must contain at least two (x, y) points")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-9))
    points = points[keep]
    cumulative = cumulative[keep]
    if len(points) < 2 or cumulative[-1] <= 0:
        raise ValueError("path must have positive length")
    return points, cumulative


def point_and_tangent_at_fraction(
    path: list[tuple[float, float]],
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a point and unit tangent at an arc-length fraction."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    points, cumulative = _arc_length_path(path)
    target = fraction * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="right") - 1)
    index = int(np.clip(index, 0, len(points) - 2))
    start_length, end_length = cumulative[index : index + 2]
    ratio = (target - start_length) / max(end_length - start_length, 1e-9)
    point = points[index] + ratio * (points[index + 1] - points[index])

    # Average adjacent directions for a more stable local normal.
    left = max(0, index - 1)
    right = min(len(points) - 1, index + 2)
    tangent = points[right] - points[left]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        tangent = points[index + 1] - points[index]
        norm = float(np.linalg.norm(tangent))
    return point, tangent / norm


def measure_perpendicular_width(
    canal_mask: np.ndarray,
    centre: np.ndarray,
    tangent: np.ndarray,
    *,
    sample_step: float = 0.25,
    maximum_distance: float | None = None,
) -> dict[str, object]:
    """Measure the contiguous mask intersection centred on a normal line."""
    if canal_mask.ndim != 2:
        raise ValueError("canal_mask must be single-channel")
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")
    height, width = canal_mask.shape
    if maximum_distance is None:
        maximum_distance = float(math.hypot(height, width))
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    distances = np.arange(
        -maximum_distance,
        maximum_distance + sample_step,
        sample_step,
    )
    samples = centre[None, :] + distances[:, None] * normal[None, :]
    xs = np.rint(samples[:, 0]).astype(int)
    ys = np.rint(samples[:, 1]).astype(int)
    in_image = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    inside = np.zeros(len(distances), dtype=bool)
    inside[in_image] = canal_mask[ys[in_image], xs[in_image]] > 0
    inside_indices = np.flatnonzero(inside)
    if inside_indices.size == 0:
        return {"valid": False, "reason": "normal_does_not_intersect_canal"}

    zero_index = int(np.argmin(np.abs(distances)))
    nearest_inside = int(inside_indices[np.argmin(np.abs(inside_indices - zero_index))])
    # Reject a spline that has moved materially outside the canal mask.
    if abs(float(distances[nearest_inside])) > max(2.0, 4.0 * sample_step):
        return {"valid": False, "reason": "centre_outside_canal"}

    left_index = nearest_inside
    while left_index > 0 and inside[left_index - 1]:
        left_index -= 1
    right_index = nearest_inside
    while right_index + 1 < len(inside) and inside[right_index + 1]:
        right_index += 1

    left_distance = float(distances[left_index] - sample_step / 2.0)
    right_distance = float(distances[right_index] + sample_step / 2.0)
    left_wall = centre + left_distance * normal
    right_wall = centre + right_distance * normal
    return {
        "valid": True,
        "reason": "valid",
        "width_px": right_distance - left_distance,
        "left_wall": (float(left_wall[0]), float(left_wall[1])),
        "centre": (float(centre[0]), float(centre[1])),
        "right_wall": (float(right_wall[0]), float(right_wall[1])),
        "normal": (float(normal[0]), float(normal[1])),
    }


def measure_apical_canal_widths(
    canal_mask: np.ndarray,
    centreline_path: list[tuple[float, float]],
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> dict[str, object]:
    """Measure 95–99% widths and return their robust median."""
    measurements: list[dict[str, object]] = []
    for fraction in fractions:
        centre, tangent = point_and_tangent_at_fraction(
            centreline_path,
            fraction,
        )
        measurement = measure_perpendicular_width(
            canal_mask,
            centre,
            tangent,
        )
        measurement["fraction"] = fraction
        measurements.append(measurement)
    valid_widths = [
        float(measurement["width_px"])
        for measurement in measurements
        if measurement["valid"]
    ]
    median_width = float(np.median(valid_widths)) if valid_widths else None
    return {
        "measurements": measurements,
        "valid_measurement_count": len(valid_widths),
        "median_width_px": median_width,
        "valid": len(valid_widths) >= 3,
        "reason": "valid" if len(valid_widths) >= 3 else "fewer_than_three_valid_widths",
    }


def create_width_overlay(
    image: np.ndarray,
    centreline_path: list[tuple[float, float]],
    measurements: list[dict[str, object]],
) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    path_pixels = np.rint(np.asarray(centreline_path)).astype(np.int32)
    if len(path_pixels) >= 2:
        cv2.polylines(overlay, [path_pixels], False, (0, 0, 255), 1, cv2.LINE_AA)
    for measurement in measurements:
        if not measurement["valid"]:
            continue
        left = tuple(np.rint(measurement["left_wall"]).astype(int))
        centre = tuple(np.rint(measurement["centre"]).astype(int))
        right = tuple(np.rint(measurement["right_wall"]).astype(int))
        cv2.line(overlay, left, right, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, centre, 1, (255, 0, 0), -1)
    return overlay
