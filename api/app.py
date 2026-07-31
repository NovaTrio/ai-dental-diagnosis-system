"""FastAPI entry point for dental radiograph analysis."""

from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.common import router as common_router
from api.schemas import (
    HealthResponse,
    LegacyWorkingLengthRequest,
    ToothSelectionRequest,
    WorkingLengthAnalysisRequest,
    WorkingLengthResponse,
)
from api.services.common_service import (
    CASE_ROOT,
    PipelineError,
    save_tooth_selection,
)
from api.services.working_length_service import (
    BINDING_MODEL_PATH,
    WL_MODEL_PATH,
    process_working_length_case,
)


CASE_ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(
    title="AI Dental Diagnosis API",
    version="1.0.0",
    description=(
        "Human-in-the-loop dental radiograph analysis. The current route "
        "implements working-length prediction and GP recommendation."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(CASE_ROOT)), name="artifacts")
app.include_router(common_router)


def _validated_case_id(case_id: str) -> str:
    try:
        return str(UUID(case_id))
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Case not found") from error


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        working_length_model_available=WL_MODEL_PATH.is_file(),
        binding_file_model_available=BINDING_MODEL_PATH.is_file(),
    )


@app.post(
    "/api/v1/cases/{case_id}/modules/working-length",
    response_model=WorkingLengthResponse,
)
async def analyze_working_length(
    case_id: str,
    request: WorkingLengthAnalysisRequest,
) -> WorkingLengthResponse:
    validated_case_id = _validated_case_id(case_id)
    try:
        result = await run_in_threadpool(
            process_working_length_case,
            validated_case_id,
            request.gp_protocol,
        )
        return WorkingLengthResponse.model_validate(result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PipelineError as error:
        raise HTTPException(
            status_code=422,
            detail={"stage": error.stage, "message": error.message},
        ) from error


@app.post(
    "/api/v1/cases/{case_id}/working-length",
    response_model=WorkingLengthResponse,
    deprecated=True,
)
async def analyze_working_length_legacy(
    case_id: str,
    request: LegacyWorkingLengthRequest,
) -> WorkingLengthResponse:
    """Compatibility route; new clients should use selection then module routes."""
    validated_case_id = _validated_case_id(case_id)
    selection = ToothSelectionRequest.model_validate(request.model_dump())
    try:
        await run_in_threadpool(
            save_tooth_selection,
            validated_case_id,
            selection,
        )
        result = await run_in_threadpool(
            process_working_length_case,
            validated_case_id,
            request.gp_protocol,
        )
        return WorkingLengthResponse.model_validate(result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PipelineError as error:
        raise HTTPException(
            status_code=422,
            detail={"stage": error.stage, "message": error.message},
        ) from error


@app.post("/api/v1/cases/{case_id}/modules/fracture", status_code=501)
async def analyze_fracture(case_id: str) -> None:
    _validated_case_id(case_id)
    raise HTTPException(
        status_code=501,
        detail=(
            "Fracture analysis is not connected yet. It will consume the saved "
            "tooth_selection.json and selected_tooth_roi.png for this case."
        ),
    )


@app.post("/api/v1/cases/{case_id}/modules/lesion", status_code=501)
async def analyze_lesion(case_id: str) -> None:
    _validated_case_id(case_id)
    raise HTTPException(
        status_code=501,
        detail=(
            "Lesion analysis is not connected yet. It will consume the saved "
            "tooth_selection.json and selected_tooth_roi.png for this case."
        ),
    )
