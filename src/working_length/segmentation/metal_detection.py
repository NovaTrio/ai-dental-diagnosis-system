"""Metal artifact detection and suppression for dental X-ray ROIs."""

from __future__ import annotations

import cv2
import numpy as np


def _robust_bright_threshold(
    image: np.ndarray,
    percentile: float,
    minimum: int,
) -> int:
    """Estimate a bright-pixel threshold from each ROI's intensity distribution."""
    values = image.astype(np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_high = median + 3.5 * 1.4826 * mad
    percentile_high = float(np.percentile(values, percentile))
    return int(np.clip(max(robust_high, percentile_high, minimum), minimum, 250))


def detect_artifacts_adaptive(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect radiopaque artifacts using image-adaptive intensity and shape cues."""
    threshold = _robust_bright_threshold(image, percentile=97.5, minimum=195)
    _, candidates = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates,
        connectivity=8,
    )

    artifact_mask = np.zeros_like(candidates)
    image_area = image.size

    for label in range(1, label_count):
        x, y, width, height, area = stats[label]
        if area < 8:
            continue

        component = labels == label
        aspect_ratio = max(width, height) / max(1, min(width, height))
        fill_ratio = area / max(1, width * height)
        mean_intensity = float(image[component].mean())
        touches_border = (
            x == 0
            or y == 0
            or x + width >= image.shape[1]
            or y + height >= image.shape[0]
        )

        elongated_metal = aspect_ratio >= 4.0 and fill_ratio >= 0.18
        dense_border_fixture = (
            touches_border
            and area >= 0.015 * image_area
            and fill_ratio >= 0.45
        )
        saturated_component = mean_intensity >= 248 and area >= 12

        if elongated_metal or dense_border_fixture or saturated_component:
            artifact_mask[component] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    artifact_mask = cv2.dilate(artifact_mask, kernel, iterations=1)
    return artifact_mask, threshold


def remove_metal_artifacts(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptively detect radiopaque artifacts and replace them by inpainting."""
    metal_mask, _ = detect_artifacts_adaptive(image)
    cleaned_image = cv2.inpaint(image, metal_mask, 3, cv2.INPAINT_TELEA)
    return cleaned_image, metal_mask


def remove_metal_fixtures(
    enhanced_image: np.ndarray,
    binary_mask: np.ndarray,
    threshold: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove very bright fixtures before distance-transform processing."""
    if threshold is None:
        threshold = _robust_bright_threshold(
            enhanced_image,
            percentile=98.5,
            minimum=220,
        )

    _, fixture_candidates = cv2.threshold(
        enhanced_image,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )
    fixture_mask = cv2.bitwise_and(fixture_candidates, binary_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fixture_mask = cv2.dilate(fixture_mask, kernel, iterations=1)
    cleaned_binary = cv2.bitwise_and(binary_mask, cv2.bitwise_not(fixture_mask))
    return cleaned_binary, fixture_mask


def create_metal_detection_visualization(
    image: np.ndarray,
    metal_mask: np.ndarray,
    fixture_mask: np.ndarray,
) -> np.ndarray:
    """Overlay artifact and fixture detections as separate colors."""
    visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    visualization[metal_mask > 0] = (0, 0, 255)
    visualization[fixture_mask > 0] = (0, 255, 255)
    return visualization
