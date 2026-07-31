"""Single-case lesion detection and calibrated diameter estimation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from api.services.common_service import (
    CASE_ROOT,
    PipelineError,
    artifact_url,
    load_saved_tooth_roi,
    load_scale_calibration,
)
from src.abscess.Segmentation.lesion_segmentation import segment_lesion_image
from src.abscess.Segmentation.tooth_segmentation import segment_tooth_image
from src.abscess.preprocessing.crop_croun import extract_anatomical_region
from src.abscess.preprocessing.lesion_preprocess import preprocess_single_image
from src.abscess.training.lesion_measurement_pca_mm import (
    add_mm_measurements,
    compute_pca_measurements,
    draw_visualization,
    find_largest_contour,
)


@lru_cache(maxsize=1)
def _load_detector_models():
    try:
        from src.abscess.Segmentation.lesion_detection import load_lesion_detectors

        return load_lesion_detectors()
    except ModuleNotFoundError as error:
        raise PipelineError(
            "lesion_detection",
            f"Lesion detector dependency is unavailable: {error}",
        ) from error
    except FileNotFoundError as error:
        raise PipelineError(
            "lesion_detection",
            f"Lesion detector model is unavailable: {error}",
        ) from error


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0 or not cv2.imwrite(str(path), image):
        raise PipelineError("artifact_storage", f"Could not save {path.name}")


def _artifact_map(case_id: str, measured: bool) -> dict[str, str | None]:
    names: dict[str, str | None] = {
        "preprocessed_tooth_roi": "lesion_preprocessed_tooth_roi.png",
        "periapical_crop": "lesion_periapical_crop.png",
        "tooth_mask": "lesion_tooth_mask.png",
        "lesion_mask": "lesion_mask.png",
        "lesion_overlay": "lesion_overlay.png",
        "measurement_visualization": (
            "lesion_measurement.png" if measured else None
        ),
    }
    return {
        key: artifact_url(case_id, filename) if filename else None
        for key, filename in names.items()
    }


def process_lesion_case(case_id: str) -> dict[str, Any]:
    """Run lesion inference using the saved tooth ROI and scale calibration."""
    case_dir = CASE_ROOT / case_id
    selected_roi = load_saved_tooth_roi(case_id)
    calibration = load_scale_calibration(case_id)
    mm_per_pixel = float(calibration["mm_per_pixel"])

    selected_path = case_dir / "selected_tooth_roi.png"
    preprocessed_path = case_dir / "lesion_preprocessed_tooth_roi.png"
    crop_path = case_dir / "lesion_periapical_crop.png"
    preprocess_single_image(str(selected_path), str(preprocessed_path))
    crop = extract_anatomical_region(
        str(preprocessed_path),
        save_path=str(crop_path),
        debug_path=None,
    )
    if crop is None or crop.size == 0:
        raise PipelineError(
            "lesion_preprocessing", "Periapical region could not be extracted"
        )

    pipeline_warnings: list[str] = []
    tooth_mask_for_inference: np.ndarray | None
    try:
        tooth_mask = segment_tooth_image(str(crop_path))
    except FileNotFoundError as error:
        # Match run_pipeline.py: the lesion feature extractor supports a
        # missing tooth mask and substitutes neutral anatomy features.
        tooth_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        tooth_mask_for_inference = None
        pipeline_warnings.append(
            "Tooth segmentation model was unavailable; lesion inference used "
            "neutral tooth-anatomy features."
        )
    else:
        if tooth_mask is None or cv2.countNonZero(tooth_mask) == 0:
            tooth_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            tooth_mask_for_inference = None
            pipeline_warnings.append(
                "No tooth region was segmented; lesion inference used neutral "
                "tooth-anatomy features."
            )
        else:
            tooth_mask_for_inference = tooth_mask
    tooth_mask_path = case_dir / "lesion_tooth_mask.png"
    _write_image(tooth_mask_path, tooth_mask)

    try:
        candidate_mask = segment_lesion_image(
            str(crop_path),
            tooth_mask=tooth_mask_for_inference,
        )
    except FileNotFoundError as error:
        raise PipelineError(
            "lesion_segmentation",
            f"Lesion segmentation model is unavailable: {error}",
        ) from error
    if candidate_mask is None:
        raise PipelineError(
            "lesion_segmentation", "Lesion segmentation did not produce a mask"
        )

    try:
        from src.abscess.Segmentation.lesion_detection import detect_lesion_single
    except ModuleNotFoundError as error:
        raise PipelineError(
            "lesion_detection",
            f"Lesion detector dependency is unavailable: {error}",
        ) from error
    detection = detect_lesion_single(
        str(crop_path),
        fcm_mask=candidate_mask,
        tooth_mask=tooth_mask_for_inference,
        models=_load_detector_models(),
    )
    if detection is None:
        raise PipelineError("lesion_detection", "Lesion detector could not read the crop")

    detected = bool(detection["detected"])
    lesion_mask = (
        detection.get("best_mask")
        if detected and detection.get("best_mask") is not None
        else np.zeros(candidate_mask.shape[:2], dtype=np.uint8)
    )
    lesion_mask = (np.asarray(lesion_mask) > 0).astype(np.uint8) * 255
    _write_image(case_dir / "lesion_mask.png", lesion_mask)

    crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop_bgr is None:
        raise PipelineError("lesion_detection", "Periapical crop is unreadable")
    overlay = crop_bgr.copy()
    overlay[lesion_mask > 0] = (
        0.45 * overlay[lesion_mask > 0] + 0.55 * np.array([0, 0, 255])
    ).astype(np.uint8)
    _write_image(case_dir / "lesion_overlay.png", overlay)

    probability = detection.get("ensemble_prob")
    base_response: dict[str, Any] = {
        "case_id": case_id,
        "status": "no_lesion" if not detected else "completed",
        "lesion_detected": detected,
        "detection_probability": (
            float(probability) if probability is not None else None
        ),
        "mm_per_pixel": mm_per_pixel,
        "lesion_diameter_px": None,
        "lesion_diameter_mm": None,
        "major_diameter_px": None,
        "minor_diameter_px": None,
        "horizontal_diameter_px": None,
        "vertical_diameter_px": None,
        "major_diameter_mm": None,
        "minor_diameter_mm": None,
        "horizontal_diameter_mm": None,
        "vertical_diameter_mm": None,
        "area_px2": None,
        "area_mm2": None,
        "orientation_angle_degrees": None,
        "artifacts": _artifact_map(case_id, measured=False),
        "warnings": [
            "Research-use output; clinician review is required.",
            *pipeline_warnings,
        ],
    }
    if not detected:
        with (case_dir / "lesion_result.json").open("w", encoding="utf-8") as file:
            json.dump(base_response, file, indent=2)
        return base_response

    contour = find_largest_contour(lesion_mask)
    if contour is None or len(contour) < 5:
        raise PipelineError(
            "lesion_measurement", "Detected lesion has no measurable contour"
        )
    measurements = compute_pca_measurements(contour)
    add_mm_measurements(measurements, mm_per_pixel)
    measurement_image = draw_visualization(lesion_mask, contour, measurements)
    _write_image(case_dir / "lesion_measurement.png", measurement_image)
    base_response.update(
        {
            # The runtime pipeline defines lesion diameter as the PCA major
            # diameter. Keep the explicit major field as a compatibility alias.
            "lesion_diameter_px": float(measurements["major_diameter_pixels"]),
            "lesion_diameter_mm": float(measurements["major_diameter_mm"]),
            "major_diameter_px": float(measurements["major_diameter_pixels"]),
            "minor_diameter_px": float(measurements["minor_diameter_pixels"]),
            "horizontal_diameter_px": float(
                measurements["horizontal_diameter_pixels"]
            ),
            "vertical_diameter_px": float(
                measurements["vertical_diameter_pixels"]
            ),
            "major_diameter_mm": float(measurements["major_diameter_mm"]),
            "minor_diameter_mm": float(measurements["minor_diameter_mm"]),
            "horizontal_diameter_mm": float(
                measurements["horizontal_diameter_mm"]
            ),
            "vertical_diameter_mm": float(measurements["vertical_diameter_mm"]),
            "area_px2": float(measurements["area_pixels"]),
            "area_mm2": float(measurements["area_mm2"]),
            "orientation_angle_degrees": float(
                measurements["orientation_angle_degrees"]
            ),
            "artifacts": _artifact_map(case_id, measured=True),
        }
    )
    with (case_dir / "lesion_result.json").open("w", encoding="utf-8") as file:
        json.dump(base_response, file, indent=2)
    return base_response
