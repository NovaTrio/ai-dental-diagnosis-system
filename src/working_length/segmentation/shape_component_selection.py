"""Anatomy-aware connected-component selection for tooth masks."""

from __future__ import annotations

import cv2
import numpy as np


def component_solidity(component_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        (component_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return 0.0 if hull_area <= 0 else float(area / hull_area)


def row_occupancy_score(component_mask: np.ndarray) -> float:
    occupied_rows = np.any(component_mask > 0, axis=1)
    ys = np.flatnonzero(occupied_rows)
    if ys.size == 0:
        return 0.0
    row_range = int(ys[-1] - ys[0] + 1)
    occupied_count = int(np.count_nonzero(occupied_rows[ys[0] : ys[-1] + 1]))
    return float(occupied_count / max(row_range, 1))


def width_profile(component_mask: np.ndarray) -> np.ndarray:
    widths = np.zeros(component_mask.shape[0], dtype=np.float32)
    for row_id, row in enumerate(component_mask):
        xs = np.flatnonzero(row > 0)
        if xs.size:
            widths[row_id] = float(xs[-1] - xs[0] + 1)
    return widths


def width_smoothness_score(component_mask: np.ndarray) -> float:
    widths = width_profile(component_mask)
    occupied = np.flatnonzero(widths > 0)
    if occupied.size < 3:
        return 0.0
    consecutive = np.diff(occupied) == 1
    differences = np.abs(np.diff(widths[occupied]))[consecutive]
    if differences.size == 0:
        return 0.0
    mean_width = float(np.mean(widths[occupied]))
    if mean_width <= 0:
        return 0.0
    return float(np.exp(-3.0 * float(np.mean(differences) / mean_width)))


def crown_root_profile_score(component_mask: np.ndarray) -> float:
    occupied = width_profile(component_mask)
    occupied = occupied[occupied > 0]
    if occupied.size < 10:
        return 0.0

    # Treat the broader end as the crown so vertically flipped ROIs remain valid.
    end_count = max(2, int(round(0.20 * occupied.size)))
    if np.median(occupied[-end_count:]) > np.median(occupied[:end_count]):
        occupied = occupied[::-1]

    crown_end = max(1, int(round(0.35 * occupied.size)))
    middle_end = max(crown_end + 1, int(round(0.70 * occupied.size)))
    middle_end = min(middle_end, occupied.size - 1)
    crown_width = float(np.median(occupied[:crown_end]))
    middle_width = float(np.median(occupied[crown_end:middle_end]))
    apex_width = float(np.median(occupied[middle_end:]))
    crown_score = np.clip(crown_width / max(middle_width, 1e-6) / 1.3, 0.0, 1.0)
    taper_score = np.clip(middle_width / max(apex_width, 1e-6) / 1.3, 0.0, 1.0)
    return float(0.5 * crown_score + 0.5 * taper_score)


def centerline_smoothness_score(component_mask: np.ndarray) -> float:
    centers = []
    for row in component_mask:
        xs = np.flatnonzero(row > 0)
        if xs.size:
            centers.append(float(np.mean(xs)))
    if len(centers) < 3:
        return 0.0

    # Curvature penalizes abrupt bends but permits a consistently tilted tooth.
    curvature = np.abs(np.diff(np.asarray(centers, dtype=np.float32), n=2))
    normalized = float(np.mean(curvature) / max(component_mask.shape[1], 1))
    return float(np.exp(-12.0 * normalized))


def _horizontal_run_count(row: np.ndarray) -> int:
    binary = (row > 0).astype(np.int8)
    transitions = np.diff(np.pad(binary, (1, 1), mode="constant"))
    return int(np.count_nonzero(transitions == 1))


def score_tooth_candidate(
    mask: np.ndarray,
    core_mask: np.ndarray,
    metal_mask: np.ndarray,
    expected_center_x: float | None = None,
) -> tuple[float, dict]:
    """Score a complete tooth candidate using anatomy and contamination cues."""
    if mask.shape != core_mask.shape or mask.shape != metal_mask.shape:
        raise ValueError("Candidate, core, and metal masks must have identical shapes")
    binary = mask > 0
    image_height, image_width = binary.shape
    if expected_center_x is None:
        expected_center_x = image_width / 2.0
    if not np.any(binary):
        return float("-inf"), {"score": None, "rejected": True, "reason": "empty"}

    ys, xs = np.where(binary)
    area = int(np.count_nonzero(binary))
    area_ratio = area / max(image_height * image_width, 1)
    vertical_span = (int(ys.max()) - int(ys.min()) + 1) / max(image_height, 1)
    core = core_mask > 0
    core_overlap = float(np.logical_and(binary, core).sum() / max(core.sum(), 1))
    centrality = 1.0 - min(
        abs(float(np.mean(xs)) - expected_center_x) / max(image_width / 2.0, 1.0),
        1.0,
    )
    vertical_score = float(np.clip(vertical_span / 0.75, 0.0, 1.0))
    area_score = float(np.exp(-abs(area_ratio - 0.25) / 0.22))
    row_score = row_occupancy_score(binary.astype(np.uint8))
    width_score = width_smoothness_score(binary.astype(np.uint8))
    taper_score = crown_root_profile_score(binary.astype(np.uint8))

    occupied_rows = np.flatnonzero(np.any(binary, axis=1))
    multiple_run_fraction = float(
        np.mean([_horizontal_run_count(binary[y]) > 1 for y in occupied_rows])
    )
    widths = np.count_nonzero(binary, axis=1)
    positive_widths = widths[widths > 0]
    width_spread = max(
        0.0,
        float(
            np.percentile(positive_widths, 95)
            / max(float(np.median(positive_widths)), 1.0)
            - 1.8
        ),
    )
    border_penalty = (
        0.10 * float(np.any(binary[:, 0]))
        + 0.10 * float(np.any(binary[:, -1]))
    )
    metal_overlap = float(
        np.logical_and(binary, metal_mask > 0).sum() / max(area, 1)
    )
    branch_penalty = 0.30 * multiple_run_fraction
    abnormal_extension_penalty = 0.12 * width_spread
    metal_penalty = 0.50 * metal_overlap
    total_penalty = (
        branch_penalty
        + abnormal_extension_penalty
        + border_penalty
        + metal_penalty
    )
    score = (
        0.20 * core_overlap
        + 0.15 * vertical_score
        + 0.15 * centrality
        + 0.15 * width_score
        + 0.15 * taper_score
        + 0.10 * area_score
        + 0.10 * row_score
        - total_penalty
    )
    return float(score), {
        "score": float(score),
        "core_overlap": core_overlap,
        "vertical_span_score": vertical_score,
        "centrality": float(centrality),
        "width_smoothness": float(width_score),
        "root_taper": float(taper_score),
        "area_plausibility": area_score,
        "row_continuity": float(row_score),
        "area_ratio": float(area_ratio),
        "multiple_run_fraction": multiple_run_fraction,
        "width_spread": width_spread,
        "metal_overlap": metal_overlap,
        "branch_penalty": branch_penalty,
        "abnormal_extension_penalty": abnormal_extension_penalty,
        "border_penalty": border_penalty,
        "metal_penalty": metal_penalty,
        "total_penalty": total_penalty,
        "rejected": False,
    }


def select_reconstruction_candidate(
    candidates: dict[str, np.ndarray],
    core_mask: np.ndarray,
    metal_mask: np.ndarray,
    expected_center_x: float | None = None,
) -> tuple[str, np.ndarray, list[dict]]:
    """Select the highest-scoring reconstruction without using ground truth."""
    if not candidates:
        raise ValueError("At least one reconstruction candidate is required")
    best_name = ""
    best_mask = None
    best_score = float("-inf")
    reports = []
    for candidate_name, candidate_mask in candidates.items():
        score, report = score_tooth_candidate(
            candidate_mask,
            core_mask,
            metal_mask,
            expected_center_x=expected_center_x,
        )
        report["candidate"] = candidate_name
        report["selected"] = False
        reports.append(report)
        if score > best_score:
            best_name = candidate_name
            best_mask = candidate_mask
            best_score = score
    if best_mask is None:
        best_name, best_mask = next(iter(candidates.items()))
    for report in reports:
        report["selected"] = report["candidate"] == best_name
    return best_name, best_mask.copy(), reports


def prune_side_branches(
    mask: np.ndarray,
    core_mask: np.ndarray,
) -> np.ndarray:
    """Keep the horizontal mask run following the reliable core axis per row."""
    if mask.shape != core_mask.shape:
        raise ValueError("Mask and core mask must have identical shapes")
    binary = mask > 0
    core = core_mask > 0
    image_height, image_width = binary.shape
    core_rows = np.flatnonzero(np.any(core, axis=1))
    if core_rows.size:
        core_centers = np.asarray(
            [np.mean(np.flatnonzero(core[y])) for y in core_rows],
            dtype=np.float32,
        )
        axis = np.interp(np.arange(image_height), core_rows, core_centers)
    else:
        axis = np.full(image_height, image_width / 2.0, dtype=np.float32)

    pruned = np.zeros_like(binary)
    for y in range(image_height):
        xs = np.flatnonzero(binary[y])
        if xs.size == 0:
            continue
        split_indices = np.flatnonzero(np.diff(xs) > 1)
        starts = np.concatenate(([0], split_indices + 1))
        ends = np.concatenate((split_indices, [len(xs) - 1]))
        runs = [(int(xs[start]), int(xs[end])) for start, end in zip(starts, ends)]
        containing = [run for run in runs if run[0] <= axis[y] <= run[1]]
        selected_run = containing[0] if containing else min(
            runs,
            key=lambda run: min(abs(axis[y] - run[0]), abs(axis[y] - run[1])),
        )
        pruned[y, selected_run[0] : selected_run[1] + 1] = True

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        pruned.astype(np.uint8),
        connectivity=8,
    )
    if component_count > 2:
        best_label = max(
            range(1, component_count),
            key=lambda label: (
                int(np.logical_and(labels == label, core).sum()),
                -abs(
                    float(stats[label, cv2.CC_STAT_LEFT])
                    + float(stats[label, cv2.CC_STAT_WIDTH]) / 2.0
                    - image_width / 2.0
                ),
                int(stats[label, cv2.CC_STAT_AREA]),
            ),
        )
        pruned = labels == best_label
    return pruned.astype(np.uint8) * 255


def reject_component(
    x: int,
    y: int,
    width: int,
    height: int,
    area: int,
    image_shape: tuple[int, int],
) -> tuple[bool, list[str]]:
    del y
    image_height, image_width = image_shape
    area_ratio = area / max(image_height * image_width, 1)
    vertical_span = height / max(image_height, 1)
    horizontal_span = width / max(image_width, 1)
    touches_left = x <= 1
    touches_right = x + width >= image_width - 1

    reasons = []
    if area_ratio < 0.01:
        reasons.append("area_too_small")
    if area_ratio > 0.65:
        reasons.append("area_too_large")
    if vertical_span < 0.25:
        reasons.append("vertical_span_too_short")
    if horizontal_span > 0.90:
        reasons.append("horizontal_span_too_wide")
    if touches_left and touches_right:
        reasons.append("touches_both_side_borders")
    return bool(reasons), reasons


def draw_component_boxes(
    image: np.ndarray,
    reports: list[dict],
    labels: np.ndarray,
    stats: np.ndarray,
) -> np.ndarray:
    """Draw component bounds, selection state, and anatomical score."""
    if image.ndim == 2:
        visual = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        visual = image.copy()
    else:
        raise ValueError("image must be grayscale or BGR")

    if labels.shape != image.shape[:2]:
        raise ValueError("labels and image must have identical dimensions")

    for report in reports:
        label_id = int(report["label"])
        if label_id <= 0 or label_id >= len(stats):
            continue
        x, y, width, height, _ = (int(value) for value in stats[label_id])
        score = report.get("score")
        selected = bool(report.get("selected", False))
        rejected = bool(report.get("rejected", score is None))

        if selected:
            color = (0, 255, 0)
            text = f"{label_id}: selected {score:.2f}"
        elif rejected:
            color = (0, 0, 255)
            text = f"{label_id}: rejected"
        else:
            color = (255, 255, 255)
            text = f"{label_id}: {score:.2f}"

        cv2.rectangle(
            visual,
            (x, y),
            (min(x + width - 1, visual.shape[1] - 1),
             min(y + height - 1, visual.shape[0] - 1)),
            color,
            1,
        )
        cv2.putText(
            visual,
            text,
            (x, max(y - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return visual


def select_shape_constrained_component(
    mask: np.ndarray,
    expected_center_x: float | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Return the most tooth-like connected component and its score reports."""
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")
    binary = (mask > 0).astype(np.uint8)
    image_height, image_width = binary.shape
    if expected_center_x is None:
        expected_center_x = image_width / 2.0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    best_label = None
    best_score = float("-inf")
    reports: list[dict] = []

    for label_id in range(1, num_labels):
        x, y, width, height, area = (int(value) for value in stats[label_id])
        rejected, reasons = reject_component(
            x, y, width, height, area, binary.shape
        )
        report = {
            "label": label_id,
            "score": None,
            "rejected": rejected,
            "rejection_reasons": reasons,
            "selected": False,
        }
        if rejected:
            reports.append(report)
            continue

        component = np.zeros_like(binary)
        component[labels == label_id] = 255
        area_ratio = area / max(image_height * image_width, 1)
        vertical_span = height / max(image_height, 1)
        center_x = float(centroids[label_id][0])
        centrality = 1.0 - min(
            abs(center_x - expected_center_x) / max(image_width / 2.0, 1.0), 1.0
        )
        vertical_score = float(np.clip(vertical_span / 0.75, 0.0, 1.0))
        area_score = float(np.exp(-abs(area_ratio - 0.20) / 0.18))
        row_score = row_occupancy_score(component)
        solidity_score = component_solidity(component)
        width_score = width_smoothness_score(component)
        taper_score = crown_root_profile_score(component)
        centerline_score = centerline_smoothness_score(component)

        touches_left = x <= 1
        touches_right = x + width >= image_width - 1
        touches_top = y <= 1
        touches_bottom = y + height >= image_height - 1
        border_penalty = (
            0.10 * float(touches_left)
            + 0.10 * float(touches_right)
            + 0.50 * float(touches_left and touches_right)
            + 0.03 * float(touches_top)
            + 0.03 * float(touches_bottom)
        )
        score = (
            0.20 * centrality
            + 0.15 * vertical_score
            + 0.10 * area_score
            + 0.15 * row_score
            + 0.10 * solidity_score
            + 0.10 * width_score
            + 0.10 * taper_score
            + 0.10 * centerline_score
            - border_penalty
        )
        report.update(
            {
                "score": float(score),
                "area_ratio": float(area_ratio),
                "vertical_span": float(vertical_span),
                "centrality": float(centrality),
                "row_continuity": float(row_score),
                "solidity": float(solidity_score),
                "width_smoothness": float(width_score),
                "taper_score": float(taper_score),
                "centerline_smoothness": float(centerline_score),
                "border_penalty": float(border_penalty),
            }
        )
        reports.append(report)
        if score > best_score:
            best_score = score
            best_label = label_id

    selected = np.zeros_like(binary)
    if best_label is not None:
        selected[labels == best_label] = 255
        for report in reports:
            report["selected"] = report["label"] == best_label
    return selected, reports
