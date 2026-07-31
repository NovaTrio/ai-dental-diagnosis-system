"""Detect a dark root canal inside an already isolated tooth image."""

from __future__ import annotations

import cv2
import numpy as np


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, conversion)
    raise ValueError("image must be grayscale, BGR, or BGRA")


def create_tooth_interior_mask(
    image: np.ndarray,
    erosion_iterations: int = 1,
) -> np.ndarray:
    """Derive an interior mask while excluding the black saved background."""
    gray = _as_grayscale(image)
    tooth_mask = (gray > 0).astype(np.uint8) * 255
    if erosion_iterations > 0:
        tooth_mask = cv2.erode(
            tooth_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=erosion_iterations,
        )
    return tooth_mask


def enhance_dark_canal(
    image: np.ndarray,
    tooth_mask: np.ndarray,
    blackhat_kernel_size: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply CLAHE and black-hat filtering inside the tooth."""
    if blackhat_kernel_size < 3:
        raise ValueError("blackhat_kernel_size must be at least 3")
    if blackhat_kernel_size % 2 == 0:
        blackhat_kernel_size += 1

    gray = _as_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    clahe[tooth_mask == 0] = 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (blackhat_kernel_size, blackhat_kernel_size),
    )
    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel)
    blackhat[tooth_mask == 0] = 0
    normalized = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    return clahe, blackhat, normalized.astype(np.uint8)


def create_canal_candidates(
    blackhat_response: np.ndarray,
    tooth_mask: np.ndarray,
    adaptive_block_size: int = 31,
    adaptive_c: float = -2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold enhanced dark structures and clean thin vertical responses."""
    if adaptive_block_size < 3:
        raise ValueError("adaptive_block_size must be at least 3")
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1

    thresholded = cv2.adaptiveThreshold(
        blackhat_response,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c,
    )
    thresholded = cv2.bitwise_and(thresholded, tooth_mask)
    opened = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 9)),
        iterations=1,
    )
    cleaned = cv2.bitwise_and(cleaned, tooth_mask)
    return thresholded, cleaned


def select_central_canal_component(
    candidates: np.ndarray,
    tooth_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    """Select the most central, vertically continuous dark component."""
    binary = (candidates > 0).astype(np.uint8)
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    height, width = binary.shape
    tooth_y, tooth_x = np.where(tooth_mask > 0)
    expected_center = (
        float(np.median(tooth_x)) if tooth_x.size else width / 2.0
    )
    minimum_area = max(5, int(round(0.001 * binary.size)))
    best_label: int | None = None
    best_score = float("-inf")
    reports: list[dict[str, float | int | bool]] = []

    for label in range(1, label_count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[label]
        )
        center_x = float(centroids[label, 0])
        area_ratio = area / max(binary.size, 1)
        height_ratio = box_height / max(height, 1)
        width_ratio = box_width / max(width, 1)
        centrality = 1.0 - min(
            abs(center_x - expected_center) / max(width / 2.0, 1.0),
            1.0,
        )
        verticality = min(box_height / max(3.0 * box_width, 1.0), 1.0)
        row_continuity = (
            np.count_nonzero(np.any(labels == label, axis=1))
            / max(box_height, 1)
        )
        rejected = bool(
            area < minimum_area
            or height_ratio < 0.08
            or width_ratio > 0.45
            or area_ratio > 0.25
        )
        score = (
            0.30 * centrality
            + 0.25 * height_ratio
            + 0.20 * verticality
            + 0.15 * row_continuity
            + 0.10 * min(area_ratio / 0.04, 1.0)
        )
        reports.append(
            {
                "label": label,
                "score": float(score),
                "area": area,
                "area_ratio": float(area_ratio),
                "height_ratio": float(height_ratio),
                "width_ratio": float(width_ratio),
                "centrality": float(centrality),
                "verticality": float(verticality),
                "row_continuity": float(row_continuity),
                "rejected": rejected,
                "selected": False,
            }
        )
        if not rejected and score > best_score:
            best_score = score
            best_label = label

    selected = np.zeros_like(candidates)
    if best_label is not None:
        selected[labels == best_label] = 255
        for report in reports:
            report["selected"] = report["label"] == best_label
    return selected, reports


def create_canal_overlay(
    image: np.ndarray,
    canal_mask: np.ndarray,
) -> np.ndarray:
    gray = _as_grayscale(image)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(overlay)
    color[canal_mask > 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, 1.0, color, 0.55, 0)


def detect_canal(
    image: np.ndarray,
    *,
    blackhat_kernel_size: int = 15,
    adaptive_block_size: int = 31,
    adaptive_c: float = -2.0,
) -> dict[str, object]:
    """Run the complete dark-canal detection pipeline."""
    gray = _as_grayscale(image)
    tooth_interior = create_tooth_interior_mask(gray)
    clahe, blackhat, normalized = enhance_dark_canal(
        gray,
        tooth_interior,
        blackhat_kernel_size,
    )
    thresholded, cleaned = create_canal_candidates(
        normalized,
        tooth_interior,
        adaptive_block_size,
        adaptive_c,
    )
    canal_mask, component_reports = select_central_canal_component(
        cleaned,
        tooth_interior,
    )
    return {
        "original": gray,
        "tooth_interior_mask": tooth_interior,
        "clahe": clahe,
        "blackhat": blackhat,
        "blackhat_normalized": normalized,
        "thresholded": thresholded,
        "cleaned_candidates": cleaned,
        "canal_mask": canal_mask,
        "overlay": create_canal_overlay(gray, canal_mask),
        "component_reports": component_reports,
        "detected": cv2.countNonZero(canal_mask) > 0,
    }
