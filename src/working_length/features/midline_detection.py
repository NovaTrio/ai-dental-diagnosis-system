"""Midline extraction and measurement for segmented tooth masks."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline


def compute_path_length(path):
    return float(
        sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(path, path[1:])
        )
    )


def remove_outlier_center_points(center_points, max_jump=25):
    if len(center_points) < 3:
        return center_points

    filtered = [center_points[0]]
    for point in center_points[1:]:
        if abs(point[0] - filtered[-1][0]) <= max_jump:
            filtered.append(point)
    return filtered


def extract_rowwise_midline(mask, smooth_factor=60, min_row_pixels=5):
    center_points = []
    for y in range(mask.shape[0]):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) >= min_row_pixels:
            center_points.append(((xs.min() + xs.max()) / 2.0, y))

    center_points = remove_outlier_center_points(center_points)
    if len(center_points) < 5:
        return [], 0.0

    points = np.asarray(center_points, dtype=np.float32)
    x_values = points[:, 0]
    y_values = points[:, 1]

    try:
        spline = UnivariateSpline(y_values, x_values, s=smooth_factor)
        smooth_y = np.linspace(y_values.min(), y_values.max(), len(y_values))
        smooth_x = spline(smooth_y)
        path = [
            (int(round(x_value)), int(round(y_value)))
            for x_value, y_value in zip(smooth_x, smooth_y)
        ]
    except Exception:
        path = [
            (int(round(x_value)), int(round(y_value)))
            for x_value, y_value in points
        ]

    return path, compute_path_length(path)


def overlay_midline_on_mask(mask, path):
    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    dash_length = 6
    gap_length = 5
    for index in range(0, len(path) - 1, dash_length + gap_length):
        segment = path[index:index + dash_length]
        for first, second in zip(segment, segment[1:]):
            cv2.line(visualization, first, second, (0, 0, 0), 1)
    return visualization
