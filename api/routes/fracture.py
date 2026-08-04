"""Fracture-risk module router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.schemas import FractureResponse
from api.services.common_service import PipelineError
from api.services.fracture_service import process_fracture_case

router = APIRouter(prefix="/api/v1/cases", tags=["Fracture risk"])


@router.post("/{case_id}/modules/fracture", response_model=FractureResponse)
async def analyze_fracture(case_id: str) -> FractureResponse:
    try:
        result = await run_in_threadpool(process_fracture_case, case_id)
        return FractureResponse.model_validate(result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PipelineError as error:
        raise HTTPException(
            status_code=422,
            detail={"stage": error.stage, "message": error.message},
        ) from error
