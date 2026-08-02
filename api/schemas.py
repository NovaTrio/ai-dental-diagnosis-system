"""Request and response schemas for the dental API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


GPProtocol = Literal["nearest-04", "upsize-04", "nearest-06", "upsize-06"]


class ToothSelectionRequest(BaseModel):
    selected_x: int = Field(ge=0)
    selected_y: int = Field(ge=0)
    direction_x: int = Field(ge=0)
    direction_y: int = Field(ge=0)


class ScaleSelectionRequest(BaseModel):
    start_x: int = Field(ge=0)
    start_y: int = Field(ge=0)
    end_x: int = Field(ge=0)
    end_y: int = Field(ge=0)
    known_length_mm: float = Field(default=10.0, gt=0)


class WorkingLengthAnalysisRequest(BaseModel):
    gp_protocol: GPProtocol = "nearest-06"


class LegacyWorkingLengthRequest(ToothSelectionRequest):
    gp_protocol: GPProtocol = "nearest-06"


class UploadResponse(BaseModel):
    case_id: str
    filename: str
    image_width: int
    image_height: int
    selection_image_url: str
    raw_image_url: str
    raw_image_width: int
    raw_image_height: int
    next_action: str


class ToothSelectionResponse(BaseModel):
    case_id: str
    status: Literal["selected"]
    selection: ToothSelectionRequest
    selection_json_url: str
    selected_tooth_roi_url: str
    available_module_endpoints: dict[str, str]
    next_action: str


class ScaleSelectionResponse(BaseModel):
    case_id: str
    status: Literal["calibrated"]
    selection: ScaleSelectionRequest
    raw_scale_length_px: float
    scale_length_px: float
    mm_per_pixel: float
    calibration_json_url: str
    calibration_image_url: str
    next_action: str


class ArtifactLinks(BaseModel):
    selected_tooth_roi: str
    preprocessed_tooth_roi: str
    tooth_mask: str
    isolated_tooth: str
    tooth_midline: str
    canal_mask: str
    canal_overlay: str
    canal_midline: str
    width_overlay: str


class LesionArtifactLinks(BaseModel):
    preprocessed_tooth_roi: str
    periapical_crop: str
    tooth_mask: str
    lesion_mask: str
    lesion_overlay: str
    measurement_visualization: str | None = None


class LesionResponse(BaseModel):
    case_id: str
    status: Literal["completed", "no_lesion"]
    lesion_detected: bool
    detection_probability: float | None = None
    mm_per_pixel: float
    lesion_diameter_px: float | None = None
    lesion_diameter_mm: float | None = None
    major_diameter_px: float | None = None
    minor_diameter_px: float | None = None
    horizontal_diameter_px: float | None = None
    vertical_diameter_px: float | None = None
    major_diameter_mm: float | None = None
    minor_diameter_mm: float | None = None
    horizontal_diameter_mm: float | None = None
    vertical_diameter_mm: float | None = None
    area_px2: float | None = None
    area_mm2: float | None = None
    orientation_angle_degrees: float | None = None
    artifacts: LesionArtifactLinks
    warnings: list[str]


class WorkingLengthResponse(BaseModel):
    case_id: str
    status: Literal["completed"]
    gp_protocol: GPProtocol
    tooth_midline_length_px: float
    canal_midline_length_px: float
    predicted_working_length_mm: float
    apical_diameter_px: float
    valid_apical_width_measurements: int
    continuous_file_size_k: float
    recommended_iso_file_size_k: int
    master_gp_recommendation: str
    artifacts: ArtifactLinks
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    working_length_model_available: bool
    binding_file_model_available: bool
