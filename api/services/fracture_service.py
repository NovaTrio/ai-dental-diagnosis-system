"""Single-case orchestration for fracture-risk assessment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from api.services.common_service import (
    CASE_ROOT,
    PipelineError,
    artifact_url,
    extract_selected_roi,
    load_saved_tooth_roi,
)
from api.schemas import ToothSelectionRequest
from src.fractures.feature_extraction.classify_pdl_patterns_v2 import classify_row
from src.fractures.feature_extraction.pdl_width_features import (
    PDLFeatureConfig,
    extract_pdl_features,
    features_to_dict,
)
from src.fractures.image_processing.anatomy_segmentation import extract_anatomical_region
from src.fractures.image_processing.fracture_preprocess import (
    fracture_specific_preprocess_image,
)
from src.fractures.image_processing.pdl_from_final_mask_polynomial import (
    extract_polynomial_root_and_dark_pdl_from_final_mask,
)
from src.fractures.image_processing.root_pdl_layered_roi import extract_root_pdl_layered_roi
from src.fractures.risk_assessment.fracture_risk_assessment import assess_fracture_risk


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0 or not cv2.imwrite(str(path), image):
        raise PipelineError("artifact_storage", f"Could not save {path.name}")


def _overlay_mask(
    base: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if base.ndim != 3:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    binary_mask = mask > 0
    if not np.any(binary_mask):
        return base
    colored = np.zeros_like(base)
    colored[binary_mask] = color_bgr
    return cv2.addWeighted(base, 1.0, colored, alpha, 0)


def _build_pdl_width_overlay(
    anatomical_region: np.ndarray,
    root_mask: np.ndarray,
    pdl_mask: np.ndarray,
) -> np.ndarray:
    base = cv2.cvtColor(anatomical_region, cv2.COLOR_GRAY2BGR)
    overlay = _overlay_mask(base, root_mask, (0, 0, 255), 0.30)
    overlay = _overlay_mask(overlay, pdl_mask, (0, 255, 255), 0.55)
    return overlay


def _fracture_preprocess_then_extract_roi(
    case_dir: Path,
    selected_roi_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Reproduce the offline ordering:
    full radiograph -> fracture preprocessing -> selected-tooth ROI.
    """
    selection_path = case_dir / "tooth_selection.json"
    original_paths = sorted(case_dir.glob("original.*"))
    if not selection_path.is_file() or not original_paths:
        raise PipelineError(
            "fracture_preprocessing",
            "The original radiograph or saved tooth selection is missing",
        )

    try:
        with selection_path.open(encoding="utf-8") as file:
            selection_document = json.load(file)
        coordinate_width = int(selection_document["coordinate_image_width"])
        coordinate_height = int(selection_document["coordinate_image_height"])
        if coordinate_width <= 0 or coordinate_height <= 0:
            raise ValueError
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise PipelineError(
            "fracture_preprocessing",
            "Saved tooth-selection metadata is invalid",
        ) from error

    original = cv2.imread(str(original_paths[0]), cv2.IMREAD_COLOR)
    if original is None:
        raise PipelineError(
            "fracture_preprocessing",
            "The original radiograph is unreadable",
        )
    preprocessed_full = fracture_specific_preprocess_image(original)
    height, width = preprocessed_full.shape

    scale_x = width / coordinate_width
    scale_y = height / coordinate_height
    mapped_selection = ToothSelectionRequest(
        selected_x=int(round(float(selection_document["selected_x"]) * scale_x)),
        selected_y=int(round(float(selection_document["selected_y"]) * scale_y)),
        direction_x=int(round(float(selection_document["direction_x"]) * scale_x)),
        direction_y=int(round(float(selection_document["direction_y"]) * scale_y)),
    )
    mapped_values = mapped_selection.model_dump()
    mapped_values["selected_x"] = min(mapped_values["selected_x"], width - 1)
    mapped_values["direction_x"] = min(mapped_values["direction_x"], width - 1)
    mapped_values["selected_y"] = min(mapped_values["selected_y"], height - 1)
    mapped_values["direction_y"] = min(mapped_values["direction_y"], height - 1)

    processed_roi = extract_selected_roi(
        preprocessed_full,
        ToothSelectionRequest.model_validate(mapped_values),
    )
    if processed_roi.ndim != 2 or processed_roi.size == 0:
        raise PipelineError(
            "fracture_preprocessing",
            "The preprocessed selected-tooth ROI is empty",
        )

    # With equal coordinate/original dimensions, ROI geometry must remain
    # identical to the common selection. For legacy resized cases, it scales
    # proportionally to the full-resolution original.
    if (
        coordinate_width == width
        and coordinate_height == height
        and processed_roi.shape != selected_roi_shape[:2]
    ):
        raise PipelineError(
            "fracture_preprocessing",
            "Replayed tooth selection produced inconsistent ROI dimensions",
        )
    return processed_roi


def process_fracture_case(case_id: str) -> dict[str, Any]:
    case_dir = CASE_ROOT / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Unknown case: {case_id}")

    try:
        selected_roi = load_saved_tooth_roi(case_id)

        preprocessed_roi = _fracture_preprocess_then_extract_roi(
            case_dir,
            selected_roi.shape,
        )
        preprocessed_roi_path = case_dir / "fracture_preprocessed_roi.png"
        _write_image(preprocessed_roi_path, preprocessed_roi)

        anatomical_region_path = case_dir / "anatomical_region.png"
        anatomical_region_debug_path = case_dir / "anatomical_region_debug.png"
        extract_anatomical_region(
            image_path=str(preprocessed_roi_path),
            save_path=str(anatomical_region_path),
            debug_path=str(anatomical_region_debug_path),
            central_width_ratio=0.75,
            bright_percentile=88,
            min_bright_ratio=0.16,
            consecutive_rows=6,
            crown_depth_ratio=0.18,
            min_crown_depth=30,
            max_crown_depth_ratio=0.28,
        )

        anatomical_region = cv2.imread(str(anatomical_region_path), cv2.IMREAD_GRAYSCALE)
        if anatomical_region is None:
            raise PipelineError("anatomy_segmentation", "Anatomical region image is unreadable")

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)

            final_mask_path = temp_dir / "bone_pdl_root_region_mask.png"
            root_mask_path = temp_dir / "root_mask.png"
            pdl_mask_path = temp_dir / "pdl_mask.png"
            outer_mask_path = temp_dir / "outer_mask.png"
            root_layered_roi_path = temp_dir / "root_pdl_layered_roi.png"
            root_layered_debug_path = temp_dir / "root_pdl_layered_roi_debug.png"

            extract_root_pdl_layered_roi(
                image_path=str(anatomical_region_path),
                mask_path=str(final_mask_path),
                roi_path=str(root_layered_roi_path),
                debug_path=str(root_layered_debug_path),
                root_mask_path=str(root_mask_path),
                pdl_mask_path=str(pdl_mask_path),
                outer_mask_path=str(outer_mask_path),
                outer_space_px=4,
                edge_smooth_px=5,
                min_end_ratio=0.50,
                max_end_ratio=0.93,
                convergence_ratio=0.36,
                end_padding_px=5,
                min_half_width_ratio=0.040,
                max_half_width_ratio=0.36,
                expected_half_width_ratio=0.19,
                pdl_min_px=1,
                pdl_max_px=16,
                lamina_search_px=20,
                fallback_pdl_width_px=7,
                root_taper_ratio=0.12,
                outer_taper_ratio=0.12,
                root_end_half_width_px=2,
                outer_end_half_width_px=5,
            )

            final_mask = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE)
            root_mask = cv2.imread(str(root_mask_path), cv2.IMREAD_GRAYSCALE)
            pdl_mask = cv2.imread(str(pdl_mask_path), cv2.IMREAD_GRAYSCALE)
            if final_mask is None or root_mask is None or pdl_mask is None:
                raise PipelineError(
                    "root_pdl_segmentation",
                    "One or more segmentation masks is unreadable",
                )

            polynomial_root_mask_path = case_dir / "polynomial_root_mask.png"
            polynomial_pdl_mask_path = case_dir / "polynomial_pdl_mask.png"
            polynomial_debug_path = case_dir / "polynomial_root_pdl_debug.png"
            polynomial_result = extract_polynomial_root_and_dark_pdl_from_final_mask(
                image_path=str(anatomical_region_path),
                final_mask_path=str(final_mask_path),
                root_mask_path=str(polynomial_root_mask_path),
                pdl_mask_path=str(polynomial_pdl_mask_path),
                debug_path=str(polynomial_debug_path),
                root_percentile=8,
                polynomial_degree=3,
                root_end_half_width_px=2,
                apex_synthetic_weight=10,
                root_taper_ratio=0.13,
                apex_padding_px=4,
                min_pdl_search_px=1,
                max_pdl_search_px=24,
                expected_pdl_px=6,
                max_pdl_px=60,
                dark_percentile=70,
                adaptive_offset=0.035,
                probability_threshold=0.10,
                gap_tolerance_px=2,
                min_component_area_px=8,
            )

            if polynomial_result is None:
                raise PipelineError(
                    "polynomial_pdl_segmentation",
                    "Polynomial root and PDL segmentation returned no result",
                )

            root_mask = cv2.imread(str(polynomial_root_mask_path), cv2.IMREAD_GRAYSCALE)
            pdl_mask = cv2.imread(str(polynomial_pdl_mask_path), cv2.IMREAD_GRAYSCALE)
            if root_mask is None or pdl_mask is None:
                raise PipelineError("polynomial_pdl_segmentation", "Polynomial masks are unreadable")

            if cv2.countNonZero(root_mask) == 0 or cv2.countNonZero(pdl_mask) == 0:
                raise PipelineError("pdl_feature_extraction", "The root and PDL masks are empty")

            features, _ = extract_pdl_features(
                root_mask=root_mask,
                pdl_mask=pdl_mask,
                image_name=case_id,
                config=PDLFeatureConfig(),
            )
            feature_row = pd.Series(features_to_dict(features))
            pattern_result = classify_row(feature_row)

            assessed = assess_fracture_risk(int(pattern_result["predicted_pdl_pattern_score"]))

        response: dict[str, Any] = {
            "case_id": case_id,
            "status": "completed",
            "pdl_pattern": {
                "score": int(pattern_result["predicted_pdl_pattern_score"]),
                "label": pattern_result["predicted_pdl_pattern_label"],
                "confidence_margin": float(pattern_result["prediction_confidence_margin"]),
                "uniform_score": float(pattern_result["uniform_score"]),
                "side_score": float(pattern_result["side_score"]),
                "irregular_score": float(pattern_result["irregular_score"]),
            },
            "fracture_risk": {
                "score": assessed.fracture_risk_score,
                "label": assessed.fracture_risk_label,
                "explanation": assessed.explanation,
            },
            "artifacts": {
                "fracture_preprocessed_roi": artifact_url(case_id, "fracture_preprocessed_roi.png"),
                "anatomical_region": artifact_url(case_id, "anatomical_region.png"),
                "anatomical_region_debug": artifact_url(case_id, "anatomical_region_debug.png"),
                "polynomial_root_mask": artifact_url(case_id, "polynomial_root_mask.png"),
                "polynomial_pdl_mask": artifact_url(case_id, "polynomial_pdl_mask.png"),
                "polynomial_root_pdl_debug": artifact_url(case_id, "polynomial_root_pdl_debug.png"),
            },
        }

        with (case_dir / "result.json").open("w", encoding="utf-8") as file:
            json.dump(response, file, indent=2)

        return response
    except ValueError as error:
        raise PipelineError("pdl_feature_extraction", str(error)) from error
    except Exception as error:  # pragma: no cover - defensive fallback for unsupported inputs
        raise PipelineError("fracture_pipeline", str(error)) from error
