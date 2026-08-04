"""Periapical lesion analysis endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.routes.common import validated_case_id
from api.schemas import LesionResponse
from api.services.common_service import PipelineError
from api.services.lesion_service import process_lesion_case


router = APIRouter(prefix="/api/v1/cases", tags=["Lesion analysis"])


@router.post("/{case_id}/modules/lesion", response_model=LesionResponse)
async def analyze_lesion(case_id: str) -> LesionResponse:
    case_id = validated_case_id(case_id)
    try:
        result = await run_in_threadpool(process_lesion_case, case_id)
        return LesionResponse.model_validate(result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PipelineError as error:
        raise HTTPException(
            status_code=422,
            detail={"stage": error.stage, "message": error.message},
        ) from error
