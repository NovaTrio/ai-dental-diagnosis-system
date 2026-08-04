"""Common upload and tooth-selection endpoints used by every diagnosis module."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api.schemas import (
    ScaleSelectionRequest,
    ScaleSelectionResponse,
    ToothSelectionRequest,
    ToothSelectionResponse,
    UploadResponse,
)
from api.services.common_service import (
    CASE_ROOT,
    PipelineError,
    prepare_diagnostic_image,
    preprocess_working_length_image,
    save_scale_selection,
    save_tooth_selection,
)


router = APIRouter(prefix="/api/v1/cases", tags=["Common case and tooth selection"])
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/tiff", "image/bmp"}


def validated_case_id(case_id: str) -> str:
    try:
        return str(UUID(case_id))
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Case not found") from error


async def _create_case(image: UploadFile, flow: str) -> UploadResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a PNG, JPEG, TIFF, or BMP radiograph",
        )
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB limit")
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=422, detail="File is not a readable image")

    processor = (
        preprocess_working_length_image
        if flow == "working-length"
        else prepare_diagnostic_image
    )
    selection_image = await run_in_threadpool(processor, decoded)
    case_id = str(uuid4())
    case_dir = CASE_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(image.filename or "radiograph.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        suffix = ".png"
    original_filename = f"original{suffix}"
    if not cv2.imwrite(str(case_dir / original_filename), decoded):
        raise HTTPException(status_code=500, detail="Could not store uploaded image")
    if not cv2.imwrite(str(case_dir / "selection_image.png"), selection_image):
        raise HTTPException(status_code=500, detail="Could not store selection image")
    height, width = selection_image.shape
    return UploadResponse(
        case_id=case_id,
        filename=image.filename or "radiograph",
        image_width=width,
        image_height=height,
        selection_image_url=f"/artifacts/{case_id}/selection_image.png",
        raw_image_url=f"/artifacts/{case_id}/{original_filename}",
        raw_image_width=int(decoded.shape[1]),
        raw_image_height=int(decoded.shape[0]),
        next_action=(
            f"Display the {flow} selection image and submit tooth centre and "
            "direction coordinates to the tooth-selection endpoint."
        ),
    )


@router.post(
    "/working-length", response_model=UploadResponse, status_code=status.HTTP_201_CREATED
)
async def create_working_length_case(image: UploadFile = File(...)) -> UploadResponse:
    return await _create_case(image, "working-length")


@router.post(
    "/diagnostic", response_model=UploadResponse, status_code=status.HTTP_201_CREATED
)
async def create_diagnostic_case(image: UploadFile = File(...)) -> UploadResponse:
    return await _create_case(image, "fracture-lesion")


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def create_case(image: UploadFile = File(...)) -> UploadResponse:
    """Compatibility upload endpoint retaining the original WL behavior."""
    return await _create_case(image, "working-length")


@router.post("/{case_id}/tooth-selection", response_model=ToothSelectionResponse)
async def select_tooth(
    case_id: str,
    selection: ToothSelectionRequest,
) -> ToothSelectionResponse:
    case_id = validated_case_id(case_id)
    try:
        result = await run_in_threadpool(save_tooth_selection, case_id, selection)
        return ToothSelectionResponse.model_validate(result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PipelineError as error:
        raise HTTPException(
            status_code=422,
            detail={"stage": error.stage, "message": error.message},
        ) from error


@router.post(
    "/{case_id}/scale-selection",
    response_model=ScaleSelectionResponse,
)
async def select_scale(
    case_id: str,
    selection: ScaleSelectionRequest,
) -> ScaleSelectionResponse:
    case_id = validated_case_id(case_id)
    try:
        result = await run_in_threadpool(save_scale_selection, case_id, selection)
        return ScaleSelectionResponse.model_validate(result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PipelineError as error:
        raise HTTPException(
            status_code=422,
            detail={"stage": error.stage, "message": error.message},
        ) from error
