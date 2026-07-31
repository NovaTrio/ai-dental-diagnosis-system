"""Conservative, narrow-band boundary refinement for selected tooth masks."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import chan_vese

from src.working_length.segmentation.shape_component_selection import (
    select_shape_constrained_component,
)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    # Padding guarantees that the flood-fill seed is background even when the
    # selected component legitimately touches an image corner.
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(padded, holes)[1:-1, 1:-1]


def smooth_mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)


def create_boundary_band(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be at least 1")
    binary = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return (dilated - eroded).astype(np.uint8)


def refine_with_chan_vese(
    gray: np.ndarray,
    initial_mask: np.ndarray,
    max_iterations: int = 30,
    mu: float = 0.20,
) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image")
    if gray.shape != initial_mask.shape:
        raise ValueError("Image and initial mask dimensions must match")
    if cv2.countNonZero((initial_mask > 0).astype(np.uint8)) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    refined = chan_vese(
        gray.astype(np.float32) / 255.0,
        mu=mu,
        lambda1=1.0,
        lambda2=1.0,
        tol=1e-3,
        max_num_iter=max_iterations,
        dt=0.5,
        init_level_set=initial_mask > 0,
        extended_output=False,
    )
    return refined.astype(np.uint8) * 255


def constrained_chan_vese_refinement(
    gray: np.ndarray,
    initial_mask: np.ndarray,
    band_radius: int = 4,
    max_iterations: int = 30,
    mu: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refined = refine_with_chan_vese(gray, initial_mask, max_iterations, mu)
    band = create_boundary_band(initial_mask, radius=band_radius)
    final = initial_mask > 0
    final = final.copy()
    final[band > 0] = refined[band > 0] > 0
    return final.astype(np.uint8) * 255, band * 255, refined


def compute_gradient_map(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    return cv2.normalize(gradient, None, 0.0, 1.0, cv2.NORM_MINMAX)


def suppress_edge_crossing(
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    gradient: np.ndarray,
    gradient_threshold: float = 0.85,
) -> np.ndarray:
    initial = initial_mask > 0
    refined = refined_mask > 0
    newly_added = refined & ~initial
    final = initial | (newly_added & (gradient < gradient_threshold))
    final[initial & ~refined] = False
    return final.astype(np.uint8) * 255


def get_apex_protection_mask(
    mask: np.ndarray,
    lower_fraction: float = 0.15,
) -> np.ndarray:
    if not 0.0 <= lower_fraction <= 1.0:
        raise ValueError("lower_fraction must be between 0 and 1")
    binary = mask > 0
    ys = np.where(binary)[0]
    protection = np.zeros_like(binary)
    if ys.size == 0:
        return protection
    y_min, y_max = int(ys.min()), int(ys.max())
    component_height = y_max - y_min + 1
    apex_start = y_max - int(lower_fraction * component_height)
    protection[apex_start : y_max + 1] = binary[apex_start : y_max + 1]
    return protection


def create_boundary_overlay(
    gray: np.ndarray,
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:
    visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    initial_contours, _ = cv2.findContours(
        (initial_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    refined_contours, _ = cv2.findContours(
        (refined_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(visual, initial_contours, -1, (255, 0, 0), 1)
    cv2.drawContours(visual, refined_contours, -1, (0, 0, 255), 1)
    return visual


def refine_tooth_boundary_stages(
    gray: np.ndarray,
    initial_mask: np.ndarray,
    band_radius: int = 4,
    max_iterations: int = 30,
    mu: float = 0.20,
    gradient_threshold: float = 0.85,
    preserve_apex: bool = True,
    expected_center_x: float | None = None,
) -> dict[str, object]:
    """Refine only the neighbourhood of an already selected tooth boundary."""
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image")
    if gray.shape != initial_mask.shape:
        raise ValueError("Image and mask dimensions must match")

    selected = (initial_mask > 0).astype(np.uint8) * 255
    filled = fill_mask_holes(selected)
    smoothed = smooth_mask_boundary(filled)
    apex = get_apex_protection_mask(smoothed) if preserve_apex else None
    constrained, band, unconstrained = constrained_chan_vese_refinement(
        gray,
        smoothed,
        band_radius=band_radius,
        max_iterations=max_iterations,
        mu=mu,
    )
    gradient = compute_gradient_map(gray)
    edge_guided = suppress_edge_crossing(
        smoothed,
        constrained,
        gradient,
        gradient_threshold=gradient_threshold,
    )
    refined = fill_mask_holes(edge_guided)
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if apex is not None:
        refined_bool = refined > 0
        refined_bool |= apex
        refined = refined_bool.astype(np.uint8) * 255

    final_mask, final_reports = select_shape_constrained_component(
        refined,
        expected_center_x=expected_center_x,
    )
    final_validation_fallback = cv2.countNonZero(final_mask) == 0
    if final_validation_fallback:
        # A rejected refinement must not alter a previously plausible tooth.
        # Restore the exact input rather than retaining even mild smoothing.
        final_mask = selected.copy()
    overlay = create_boundary_overlay(gray, selected, final_mask)
    return {
        "selected_component": selected,
        "filled_mask": filled,
        "smoothed_mask": smoothed,
        "boundary_band": band,
        "chan_vese_result": unconstrained,
        "constrained_result": constrained,
        "gradient_map": (gradient * 255).astype(np.uint8),
        "edge_guided_result": edge_guided,
        "final_refined_mask": final_mask,
        "refined_overlay": overlay,
        "final_component_reports": final_reports,
        "final_validation_fallback": final_validation_fallback,
    }


def refine_tooth_boundary(
    gray: np.ndarray,
    initial_mask: np.ndarray,
    band_radius: int = 4,
    max_iterations: int = 30,
    preserve_apex: bool = True,
) -> np.ndarray:
    return refine_tooth_boundary_stages(
        gray,
        initial_mask,
        band_radius=band_radius,
        max_iterations=max_iterations,
        preserve_apex=preserve_apex,
    )["final_refined_mask"]
