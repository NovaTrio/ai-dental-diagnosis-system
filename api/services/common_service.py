"""Shared case upload, preprocessing, and doctor tooth-selection operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from api.schemas import ToothSelectionRequest
from src.common.preprocessing.base_preprocess import contrast_stretch, normalize_image
from src.common.preprocessing.selected_tooth_roi import (
    calculate_manual_rotation_angle,
    find_black_gap_boundaries,
    rotate_image_and_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = PROJECT_ROOT / "data" / "api" / "cases"


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message


def preprocess_uploaded_image(image: np.ndarray) -> np.ndarray:
    """Create the normalized 256x256 image shown in the tooth-selection UI."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Unsupported image dimensions")
    denoised = cv2.medianBlur(gray, 5)
    normalized = denoised.astype(np.float32) / 255.0
    normalized = contrast_stretch(normalized, 2, 98)
    normalized = normalize_image(normalized)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def extract_selected_roi(
    selection_image: np.ndarray,
    selection: ToothSelectionRequest,
) -> np.ndarray:
    """Rotate from the doctor's axis and extract the shared selected-tooth ROI."""
    height, width = selection_image.shape[:2]
    if (
        selection.selected_x >= width
        or selection.direction_x >= width
        or selection.selected_y >= height
        or selection.direction_y >= height
    ):
        raise PipelineError(
            "tooth_selection",
            f"Coordinates must be inside the {width}x{height} selection image",
        )
    if (
        selection.selected_x == selection.direction_x
        and selection.selected_y == selection.direction_y
    ):
        raise PipelineError(
            "tooth_selection",
            "Tooth centre and direction points must be different",
        )

    coordinates = (
        selection.selected_x,
        selection.selected_y,
        selection.direction_x,
        selection.direction_y,
    )
    bgr = cv2.cvtColor(selection_image, cv2.COLOR_GRAY2BGR)
    angle = calculate_manual_rotation_angle(*coordinates)
    rotated, rotated_points = rotate_image_and_points(
        bgr,
        [
            (selection.selected_x, selection.selected_y),
            (selection.direction_x, selection.direction_y),
        ],
        angle,
    )
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    crop_box, _, _, _ = find_black_gap_boundaries(
        rotated_gray, rotated_points[0][0]
    )
    x1, y1, x2, y2 = crop_box
    roi = rotated_gray[y1:y2, x1:x2]
    if roi.size == 0:
        raise PipelineError("tooth_selection", "Selected tooth ROI is empty")
    return roi


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0 or not cv2.imwrite(str(path), image):
        raise PipelineError("artifact_storage", f"Could not save {path.name}")


def artifact_url(case_id: str, filename: str) -> str:
    return f"/artifacts/{case_id}/{filename}"


def save_tooth_selection(
    case_id: str,
    selection: ToothSelectionRequest,
) -> dict[str, Any]:
    """Persist the doctor's JSON selection and reusable selected-tooth ROI."""
    case_dir = CASE_ROOT / case_id
    selection_path = case_dir / "selection_image.png"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Unknown case: {case_id}")
    selection_image = cv2.imread(str(selection_path), cv2.IMREAD_GRAYSCALE)
    if selection_image is None:
        raise PipelineError("common_preprocessing", "Selection image is unreadable")

    selected_roi = extract_selected_roi(selection_image, selection)
    selection_document = {
        "case_id": case_id,
        "selected_x": selection.selected_x,
        "selected_y": selection.selected_y,
        "direction_x": selection.direction_x,
        "direction_y": selection.direction_y,
        "coordinate_image": "selection_image.png",
        "coordinate_image_width": int(selection_image.shape[1]),
        "coordinate_image_height": int(selection_image.shape[0]),
    }
    with (case_dir / "tooth_selection.json").open("w", encoding="utf-8") as file:
        json.dump(selection_document, file, indent=2)
    _write_image(case_dir / "selected_tooth_roi.png", selected_roi)
    return {
        "case_id": case_id,
        "status": "selected",
        "selection": selection.model_dump(),
        "selection_json_url": artifact_url(case_id, "tooth_selection.json"),
        "selected_tooth_roi_url": artifact_url(case_id, "selected_tooth_roi.png"),
        "available_module_endpoints": {
            "working_length": f"/api/v1/cases/{case_id}/modules/working-length",
            "fracture": f"/api/v1/cases/{case_id}/modules/fracture",
            "lesion": f"/api/v1/cases/{case_id}/modules/lesion",
        },
        "next_action": "Call one or more module endpoints for this case.",
    }


def load_saved_tooth_roi(case_id: str) -> np.ndarray:
    """Load the selected-tooth ROI shared by all three diagnosis modules."""
    case_dir = CASE_ROOT / case_id
    selection_json = case_dir / "tooth_selection.json"
    roi_path = case_dir / "selected_tooth_roi.png"
    if not selection_json.is_file() or not roi_path.is_file():
        raise PipelineError(
            "tooth_selection",
            "No saved tooth selection exists; call the common tooth-selection endpoint first",
        )
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise PipelineError("tooth_selection", "Saved selected-tooth ROI is unreadable")
    return roi
