"""Adaptive multi-evidence detection of endodontic metal files."""

from __future__ import annotations

import cv2
import numpy as np


def filter_metal_components(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Retain bright, thin, long, central and reasonably solid components."""
    binary = (mask > 0).astype(np.uint8) * 255
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    output = np.zeros_like(binary)
    image_height, image_width = gray.shape
    brightness_reference = float(np.percentile(gray, 90))

    for label in range(1, label_count):
        x, y, width, height, area = stats[label]
        if area < max(5, int(0.0005 * gray.size)):
            continue

        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0
        aspect_ratio = height / max(width, 1)
        vertical_span = height / image_height
        width_ratio = width / image_width
        mean_intensity = float(gray[labels == label].mean())
        center_x = x + width / 2.0
        center_distance = abs(center_x - image_width / 2.0) / max(
            image_width / 2.0,
            1.0,
        )
        bounding_fill = area / max(width * height, 1)

        if (
            width_ratio <= 0.18
            and vertical_span >= 0.15
            and aspect_ratio >= 2.5
            and mean_intensity >= brightness_reference
            and center_distance <= 0.75
            and bounding_fill >= 0.10
            and solidity >= 0.20
        ):
            output[labels == label] = 255

    return output


def detect_adaptive_metal(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Detect and inpaint metal using intensity, ridges, lines and geometry."""
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale image")

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    height, width = gray.shape

    intensity_threshold = max(
        float(np.percentile(gray, 98.0)),
        float(np.mean(gray) + 2.3 * np.std(gray)),
    )
    bright_mask = (gray >= intensity_threshold).astype(np.uint8) * 255

    top_hat_response = np.zeros_like(gray)
    for kernel_size in ((3, 15), (3, 25), (5, 21)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        response = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        top_hat_response = np.maximum(top_hat_response, response)

    ridge_threshold = max(
        float(np.percentile(top_hat_response, 97.0)),
        float(np.mean(top_hat_response) + 2.0 * np.std(top_hat_response)),
        1.0,
    )
    ridge_mask = (top_hat_response >= ridge_threshold).astype(np.uint8) * 255

    median_intensity = float(np.median(gray))
    canny_low = int(np.clip(0.66 * median_intensity, 0, 255))
    canny_high = int(np.clip(1.33 * median_intensity, canny_low + 1, 255))
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(10, int(0.08 * height)),
        minLineLength=max(10, int(0.20 * height)),
        maxLineGap=max(3, int(0.04 * height)),
    )

    line_mask = np.zeros_like(gray)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle - 90.0) <= 20.0:
                cv2.line(
                    line_mask,
                    (x1, y1),
                    (x2, y2),
                    255,
                    thickness=max(3, int(0.04 * width)),
                )

    line_region = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    geometric_evidence = cv2.bitwise_or(ridge_mask, line_region)
    candidates = cv2.bitwise_and(bright_mask, geometric_evidence)
    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        iterations=1,
    )

    detection_mask = filter_metal_components(candidates, gray)
    removal_mask = cv2.dilate(
        detection_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    inpaint_mask = (removal_mask > 0).astype(np.uint8) * 255
    metal_removed = cv2.inpaint(
        gray,
        inpaint_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )

    return {
        "bright_mask": bright_mask,
        "ridge_response": top_hat_response,
        "ridge_mask": ridge_mask,
        "edge_mask": edges,
        "line_mask": line_mask,
        "metal_candidates": candidates,
        "metal_detection_mask": detection_mask,
        "metal_removal_mask": inpaint_mask,
        "metal_removed": metal_removed,
    }
