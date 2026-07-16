"""Shared geometric measurements for manual and predicted midlines."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def polyline_length(points: np.ndarray | list | tuple) -> float:
    """Return the sum of Euclidean distances between consecutive [x, y] points."""
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.size == 0:
        return 0.0
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2) using [x, y] coordinates")
    if len(points_array) < 2:
        return 0.0
    differences = np.diff(points_array, axis=0)
    return float(np.linalg.norm(differences, axis=1).sum())


def orient_path_coronal_to_apex(
    points: np.ndarray | list | tuple,
    tooth_mask: np.ndarray,
    end_fraction: float = 0.20,
) -> list[tuple[int, int]]:
    """Orient a rowwise path from the broader coronal end to the narrower apex."""
    if not 0.0 < end_fraction <= 0.5:
        raise ValueError("end_fraction must be in the interval (0, 0.5]")
    path = [(int(point[0]), int(point[1])) for point in points]
    if len(path) < 2:
        return path
    if tooth_mask.ndim != 2:
        raise ValueError("tooth_mask must be a single-channel image")

    widths = np.count_nonzero(tooth_mask > 0, axis=1).astype(np.float32)
    occupied_rows = np.flatnonzero(widths > 0)
    if occupied_rows.size < 2:
        return path
    band_size = max(1, int(np.ceil(end_fraction * occupied_rows.size)))
    top_width = float(np.median(widths[occupied_rows[:band_size]]))
    bottom_width = float(np.median(widths[occupied_rows[-band_size:]]))
    coronal_y = int(occupied_rows[0] if top_width >= bottom_width else occupied_rows[-1])
    if abs(path[-1][1] - coronal_y) < abs(path[0][1] - coronal_y):
        path.reverse()
    return path


def endpoint_errors(
    predicted_points: np.ndarray | list | tuple,
    ground_truth_points: np.ndarray | list | tuple,
) -> tuple[float, float]:
    """Return coronal-reference and apex errors for coronal-to-apex paths."""
    predicted = np.asarray(predicted_points, dtype=np.float32)
    ground_truth = np.asarray(ground_truth_points, dtype=np.float32)
    for name, points_array in (
        ("predicted_points", predicted),
        ("ground_truth_points", ground_truth),
    ):
        if points_array.ndim != 2 or points_array.shape[1:] != (2,):
            raise ValueError(f"{name} must have shape (N, 2)")
        if len(points_array) < 2:
            raise ValueError(f"{name} must contain at least two points")
    coronal_error = np.linalg.norm(predicted[0] - ground_truth[0])
    apex_error = np.linalg.norm(predicted[-1] - ground_truth[-1])
    return float(coronal_error), float(apex_error)


def resample_polyline(
    points: np.ndarray | list | tuple,
    spacing: float = 1.0,
) -> np.ndarray:
    """Sample a polyline uniformly by arc length, including both endpoints."""
    if spacing <= 0:
        raise ValueError("spacing must be greater than zero")
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1:] != (2,):
        raise ValueError("points must have shape (N, 2) using [x, y] coordinates")
    if len(points_array) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points_array) == 1:
        return points_array.copy()

    segment_lengths = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 0:
        return points_array[:1].copy()

    sample_count = max(2, int(np.ceil(total_length / spacing)) + 1)
    sample_distances = np.linspace(0.0, total_length, sample_count)
    x_values = np.interp(sample_distances, cumulative, points_array[:, 0])
    y_values = np.interp(sample_distances, cumulative, points_array[:, 1])
    return np.column_stack((x_values, y_values)).astype(np.float32)


def nearest_path_distances(
    source_points: np.ndarray | list | tuple,
    target_points: np.ndarray | list | tuple,
    sample_spacing: float = 1.0,
) -> np.ndarray:
    """Return each source-path sample's distance to the nearest target sample."""
    source = resample_polyline(source_points, spacing=sample_spacing)
    target = resample_polyline(target_points, spacing=sample_spacing)
    if len(source) == 0 or len(target) == 0:
        raise ValueError("Both paths must contain at least one point")
    distances, _ = cKDTree(target).query(source, k=1)
    return np.asarray(distances, dtype=np.float64)


def average_path_error(
    predicted_points: np.ndarray | list | tuple,
    ground_truth_points: np.ndarray | list | tuple,
    sample_spacing: float = 1.0,
) -> float:
    """Directed mean distance from the predicted path to ground truth."""
    distances = nearest_path_distances(
        predicted_points,
        ground_truth_points,
        sample_spacing=sample_spacing,
    )
    return float(np.mean(distances))


def symmetric_average_path_error(
    predicted_points: np.ndarray | list | tuple,
    ground_truth_points: np.ndarray | list | tuple,
    sample_spacing: float = 1.0,
) -> float:
    """Mean of predicted-to-GT and GT-to-predicted average distances."""
    predicted_to_gt = nearest_path_distances(
        predicted_points,
        ground_truth_points,
        sample_spacing=sample_spacing,
    )
    gt_to_predicted = nearest_path_distances(
        ground_truth_points,
        predicted_points,
        sample_spacing=sample_spacing,
    )
    return float(0.5 * (np.mean(predicted_to_gt) + np.mean(gt_to_predicted)))


def hausdorff_distance(
    predicted_points: np.ndarray | list | tuple,
    ground_truth_points: np.ndarray | list | tuple,
    sample_spacing: float = 1.0,
) -> float:
    """Return the symmetric maximum nearest-path disagreement."""
    predicted_to_gt = nearest_path_distances(
        predicted_points,
        ground_truth_points,
        sample_spacing=sample_spacing,
    )
    gt_to_predicted = nearest_path_distances(
        ground_truth_points,
        predicted_points,
        sample_spacing=sample_spacing,
    )
    return float(max(np.max(predicted_to_gt), np.max(gt_to_predicted)))
