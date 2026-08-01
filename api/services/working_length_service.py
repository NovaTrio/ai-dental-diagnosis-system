"""Single-case orchestration for working length and GP recommendation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from api.services.common_service import (
    CASE_ROOT,
    PipelineError,
    artifact_url,
    load_saved_tooth_roi,
)
from src.canalwidth.features.canal_midline import extract_canal_midline
from src.canalwidth.features.canal_width import (
    create_width_overlay,
    measure_apical_canal_widths,
)
from src.canalwidth.models.train_binding_file_model import nearest_iso_size
from src.canalwidth.recommend_gp import PROTOCOLS, recommend_gp
from src.canalwidth.segmentation.canal_detection import detect_canal
from src.working_length.features.extract_features import segment_tooth
from src.working_length.preprocessing.roi_extraction import (
    enhance as enhance_wl_roi,
    remove_black_borders,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WL_MODEL_PATH = PROJECT_ROOT / "src/canalwidth/models/working_length/linear_regression.joblib"
BINDING_MODEL_PATH = PROJECT_ROOT / "src/canalwidth/models/binding_file/linear_regression.joblib"


def preprocess_wl_roi(roi: np.ndarray) -> np.ndarray:
    normalized = roi.astype(np.float32) / 255.0
    cropped = remove_black_borders(normalized)
    enhanced = enhance_wl_roi(cropped, debug=False)
    return (np.clip(enhanced, 0.0, 1.0) * 255).astype(np.uint8)


@lru_cache(maxsize=1)
def load_wl_model():
    if not WL_MODEL_PATH.is_file():
        raise PipelineError("working_length_prediction", f"Missing model: {WL_MODEL_PATH}")
    return joblib.load(WL_MODEL_PATH)


@lru_cache(maxsize=1)
def load_binding_model():
    if not BINDING_MODEL_PATH.is_file():
        raise PipelineError("binding_file_prediction", f"Missing model: {BINDING_MODEL_PATH}")
    return joblib.load(BINDING_MODEL_PATH)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0 or not cv2.imwrite(str(path), image):
        raise PipelineError("artifact_storage", f"Could not save {path.name}")


def process_working_length_case(
    case_id: str,
    gp_protocol: str,
) -> dict[str, Any]:
    case_dir = CASE_ROOT / case_id
    selected_roi = load_saved_tooth_roi(case_id)
    wl_roi = preprocess_wl_roi(selected_roi)
    tooth_result = segment_tooth(wl_roi, debug=False)
    tooth_mask = tooth_result["tooth_mask"]
    isolated_tooth = tooth_result["tooth_only"]
    tooth_path = tooth_result["midline_path"]
    tooth_length = float(tooth_result["midline_length"])
    if cv2.countNonZero(tooth_mask) == 0 or isolated_tooth.size == 0:
        raise PipelineError("tooth_segmentation", "Selected tooth could not be segmented")

    canal_result = detect_canal(isolated_tooth)
    if not canal_result["detected"]:
        raise PipelineError("canal_detection", "No reliable canal was detected")
    canal_midline = extract_canal_midline(isolated_tooth, canal_result["canal_mask"])
    if not canal_midline["detected"]:
        raise PipelineError("canal_midline", "Canal centreline is incomplete")

    canal_length = float(canal_midline["midline_length_px"])
    predicted_wl = float(
        load_wl_model().predict(np.asarray([[canal_length]], dtype=float))[0]
    )
    widths = measure_apical_canal_widths(
        canal_midline["combined_canal_mask"],
        canal_midline["midline_path"],
    )
    if not widths["valid"] or widths["median_width_px"] is None:
        raise PipelineError(
            "canal_width",
            "Fewer than three valid apical width measurements were available",
        )
    apical_diameter = float(widths["median_width_px"])
    continuous_file_size = float(
        load_binding_model().predict(
            np.asarray([[predicted_wl, apical_diameter]], dtype=float)
        )[0]
    )
    nearest_size = int(nearest_iso_size(continuous_file_size)[0])
    _, gp_label = recommend_gp(
        continuous_file_size,
        PROTOCOLS[gp_protocol],
    )

    tooth_midline_overlay = cv2.cvtColor(
        tooth_result["cropped_img"], cv2.COLOR_GRAY2BGR
    )
    for first, second in zip(tooth_path, tooth_path[1:]):
        cv2.line(tooth_midline_overlay, first, second, (0, 0, 255), 1)
    width_overlay = create_width_overlay(
        isolated_tooth,
        canal_midline["midline_path"],
        widths["measurements"],
    )
    artifacts = {
        "selected_tooth_roi": "selected_tooth_roi.png",
        "preprocessed_tooth_roi": "preprocessed_tooth_roi.png",
        "tooth_mask": "tooth_mask.png",
        "isolated_tooth": "isolated_tooth.png",
        "tooth_midline": "tooth_midline.png",
        "canal_mask": "canal_mask.png",
        "canal_overlay": "canal_overlay.png",
        "canal_midline": "canal_midline.png",
        "width_overlay": "width_overlay.png",
    }
    images = {
        "selected_tooth_roi": selected_roi,
        "preprocessed_tooth_roi": wl_roi,
        "tooth_mask": tooth_mask,
        "isolated_tooth": isolated_tooth,
        "tooth_midline": tooth_midline_overlay,
        "canal_mask": canal_result["canal_mask"],
        "canal_overlay": canal_result["overlay"],
        "canal_midline": canal_midline["overlay"],
        "width_overlay": width_overlay,
    }
    for key, filename in artifacts.items():
        _write_image(case_dir / filename, images[key])

    response: dict[str, Any] = {
        "case_id": case_id,
        "status": "completed",
        "gp_protocol": gp_protocol,
        "tooth_midline_length_px": tooth_length,
        "canal_midline_length_px": canal_length,
        "predicted_working_length_mm": predicted_wl,
        "apical_diameter_px": apical_diameter,
        "valid_apical_width_measurements": int(widths["valid_measurement_count"]),
        "continuous_file_size_k": continuous_file_size,
        "recommended_iso_file_size_k": nearest_size,
        "master_gp_recommendation": gp_label,
        "artifacts": {
            key: artifact_url(case_id, filename)
            for key, filename in artifacts.items()
        },
        "warnings": [
            "Research-use output; clinician review is required.",
            "Apical diameter is in pixels because no scale was supplied.",
            "GP protocol was selected by the requester, not inferred.",
        ],
    }
    with (case_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(response, file, indent=2)
    return response
