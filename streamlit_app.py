"""Streamlit client for working-length and lesion-analysis workflows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


DEFAULT_API_URL = os.getenv("DENTAL_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 180
GP_PROTOCOLS = ("nearest-04", "upsize-04", "nearest-06", "upsize-06")
ANALYSIS_MODULES = {
    "working_length": "working-length",
    "lesion": "lesion",
}


def api_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        stage = detail.get("stage")
        message = detail.get("message", detail)
        return f"{stage}: {message}" if stage else str(message)
    return str(detail)


def absolute_url(api_url: str, path: str) -> str:
    if "://" in path:
        return path
    return f"{api_url.rstrip('/')}/{path.lstrip('/')}"


def post(
    url: str,
    *,
    files: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> requests.Response:
    try:
        return requests.post(
            url, files=files, json=json, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Could not contact the API. Start it with "
            "`python -m uvicorn api.app:app --reload`."
        ) from error


def run_analysis_modules(
    api_url: str,
    case_id: str,
    gp_protocol: str,
) -> dict[str, requests.Response | Exception]:
    """Run working-length and lesion requests concurrently."""

    def call_module(name: str, endpoint: str) -> requests.Response:
        payload = {"gp_protocol": gp_protocol} if name == "working_length" else None
        return post(
            f"{api_url}/api/v1/cases/{case_id}/modules/{endpoint}",
            json=payload,
        )

    results: dict[str, requests.Response | Exception] = {}
    with ThreadPoolExecutor(max_workers=len(ANALYSIS_MODULES)) as executor:
        futures = {
            executor.submit(call_module, name, endpoint): name
            for name, endpoint in ANALYSIS_MODULES.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:
                results[name] = error
    return results


def reset_case() -> None:
    for key in (
        "case",
        "selection_image",
        "tooth_points",
        "tooth_last_click",
        "tooth_click_generation",
        "selection_result",
        "scale_points",
        "scale_last_click",
        "scale_click_generation",
        "scale_result",
        "analysis_result",
        "fracture_result",
        "lesion_result",
        "module_results",
    ):
        st.session_state.pop(key, None)


def annotated_image(
    points_key: str,
    colors: tuple[str, str],
) -> Image.Image:
    image = Image.open(BytesIO(st.session_state.selection_image)).convert("RGB")
    draw = ImageDraw.Draw(image)
    points = st.session_state.get(points_key, [])
    for index, (x, y) in enumerate(points):
        radius = 5
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colors[index],
            outline="white",
            width=2,
        )
    if len(points) == 2:
        draw.line((points[0], points[1]), fill="lime", width=2)
    return image


def collect_two_points(
    *,
    case_id: str,
    state_prefix: str,
    colors: tuple[str, str],
) -> list[tuple[int, int]]:
    points_key = f"{state_prefix}_points"
    last_key = f"{state_prefix}_last_click"
    generation_key = f"{state_prefix}_click_generation"
    points = st.session_state.setdefault(points_key, [])
    click = streamlit_image_coordinates(
        annotated_image(points_key, colors),
        key=(
            f"{state_prefix}_selection_{case_id}_"
            f"{st.session_state.get(generation_key, 0)}"
        ),
    )
    if click:
        current = (int(click["x"]), int(click["y"]))
        if current != st.session_state.get(last_key) and len(points) < 2:
            points.append(current)
            st.session_state[last_key] = current
            st.rerun()
    return points


def reset_points(state_prefix: str) -> None:
    st.session_state[f"{state_prefix}_points"] = []
    st.session_state[f"{state_prefix}_last_click"] = None
    generation_key = f"{state_prefix}_click_generation"
    st.session_state[generation_key] = st.session_state.get(generation_key, 0) + 1


st.set_page_config(page_title="AI Dental Diagnosis", page_icon="🦷", layout="wide")
api_url = DEFAULT_API_URL

st.header("1. Upload radiograph")
uploaded_file = st.file_uploader(
    "PNG, JPEG, TIFF, or BMP (maximum API size: 20 MB)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
)
if st.button("Upload and create case", disabled=uploaded_file is None):
    reset_case()
    assert uploaded_file is not None
    with st.spinner("Uploading and preprocessing the radiograph..."):
        try:
            response = post(
                f"{api_url}/api/v1/cases",
                files={
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
            )
            if not response.ok:
                st.error(api_error(response))
            else:
                case_data = response.json()
                image_response = requests.get(
                    absolute_url(api_url, case_data["selection_image_url"]),
                    timeout=30,
                )
                image_response.raise_for_status()
                st.session_state.case = case_data
                st.session_state.selection_image = image_response.content
                st.rerun()
        except (requests.RequestException, RuntimeError) as error:
            st.error(str(error))

case = st.session_state.get("case")
if case:
    case_id = case["case_id"]
    st.caption(f"Current case ID: `{case_id}`")
    st.header("2. Select the tooth and direction")
    st.write("Click the tooth center, then a point along its root direction.")
    left, right = st.columns([2, 1])
    with left:
        tooth_points = collect_two_points(
            case_id=case_id,
            state_prefix="tooth",
            colors=("red", "dodgerblue"),
        )
    with right:
        if len(tooth_points) >= 1:
            st.write(f"Tooth center: **{tooth_points[0]}**")
        if len(tooth_points) == 2:
            st.write(f"Direction: **{tooth_points[1]}**")
        if st.button("Reset tooth points", disabled=not tooth_points):
            reset_points("tooth")
            st.rerun()
        if st.button("Save tooth selection", disabled=len(tooth_points) != 2):
            payload = {
                "selected_x": tooth_points[0][0],
                "selected_y": tooth_points[0][1],
                "direction_x": tooth_points[1][0],
                "direction_y": tooth_points[1][1],
            }
            with st.spinner("Creating the selected-tooth ROI..."):
                try:
                    response = post(
                        f"{api_url}/api/v1/cases/{case_id}/tooth-selection",
                        json=payload,
                    )
                    if response.ok:
                        st.session_state.selection_result = response.json()
                        st.session_state.pop("analysis_result", None)
                        st.session_state.pop("fracture_result", None)
                        st.session_state.pop("module_results", None)
                        for key in (
                            "scale_result",
                            "analysis_result",
                            "lesion_result",
                            "module_results",
                        ):
                            st.session_state.pop(key, None)
                        st.rerun()
                    else:
                        st.error(api_error(response))
                except RuntimeError as error:
                    st.error(str(error))

selection_result = st.session_state.get("selection_result")
if case and selection_result:
    case_id = case["case_id"]
    st.header("3. Select the scale bar")
    st.write("Click both endpoints of the scale bar shown in the radiograph.")
    left, right = st.columns([2, 1])
    with left:
        scale_points = collect_two_points(
            case_id=case_id,
            state_prefix="scale",
            colors=("orange", "magenta"),
        )
    with right:
        known_scale_mm = st.number_input(
            "Known scale-bar length (mm)",
            min_value=0.1,
            value=10.0,
            step=0.5,
        )
        if len(scale_points) >= 1:
            st.write(f"Scale start: **{scale_points[0]}**")
        if len(scale_points) == 2:
            st.write(f"Scale end: **{scale_points[1]}**")
        if st.button("Reset scale points", disabled=not scale_points):
            reset_points("scale")
            st.rerun()
        if st.button("Save scale calibration", disabled=len(scale_points) != 2):
            payload = {
                "start_x": scale_points[0][0],
                "start_y": scale_points[0][1],
                "end_x": scale_points[1][0],
                "end_y": scale_points[1][1],
                "known_length_mm": known_scale_mm,
            }
            with st.spinner("Calculating the radiograph scale..."):
                try:
                    response = post(
                        f"{api_url}/api/v1/cases/{case_id}/scale-selection",
                        json=payload,
                    )
                    if response.ok:
                        st.session_state.scale_result = response.json()
                        for key in (
                            "analysis_result",
                            "lesion_result",
                            "module_results",
                        ):
                            st.session_state.pop(key, None)
                        st.rerun()
                    else:
                        st.error(api_error(response))
                except RuntimeError as error:
                    st.error(str(error))

scale_result = st.session_state.get("scale_result")
if case and selection_result and scale_result:
    case_id = case["case_id"]
    st.success(f"Scale: {scale_result['mm_per_pixel']:.6f} mm/pixel")
    st.header("4. Analyze radiograph")
    gp_protocol = st.selectbox(
        "GP protocol",
        GP_PROTOCOLS,
        index=2,
        help="Choose the gutta-percha recommendation protocol.",
    )
    if st.button("Analysis", type="primary"):
        with st.spinner("Running both analysis modules in parallel..."):
            module_responses = run_analysis_modules(api_url, case_id, gp_protocol)
            stored: dict[str, Any] = {}
            for module_name, response in module_responses.items():
                if isinstance(response, Exception):
                    stored[module_name] = {
                        "status": "error",
                        "detail": str(response),
                    }
                elif response.ok:
                    stored[module_name] = {
                        "status": "completed",
                        "data": response.json(),
                    }
                else:
                    stored[module_name] = {
                        "status": "unavailable",
                        "http_status": response.status_code,
                        "detail": api_error(response),
                    }
            st.session_state.module_results = stored
            for module_name, result_key in (
                ("working_length", "analysis_result"),
                ("lesion", "lesion_result"),
            ):
                module_result = stored[module_name]
                if module_result["status"] == "completed":
                    st.session_state[result_key] = module_result["data"]
                else:
                    st.error(
                        f"{module_name.replace('_', ' ').title()}: "
                        f"{module_result['detail']}"
                    )
            if any(item["status"] == "completed" for item in stored.values()):
                st.rerun()

lesion = st.session_state.get("lesion_result")
working_length = st.session_state.get("analysis_result")
if working_length or lesion:
    st.header("Analysis results")
    working_length_column, lesion_column = st.columns(2)

    with working_length_column:
        with st.container(border=True):
            st.subheader("Working length")
            if working_length:
                st.metric(
                    "Predicted working length",
                    f"{working_length['predicted_working_length_mm']:.2f} mm",
                )
                st.metric(
                    "Master GP",
                    working_length["master_gp_recommendation"],
                )
                st.metric(
                    "Recommended ISO file",
                    f"#{working_length['recommended_iso_file_size_k']}",
                )
            else:
                st.warning("Working-length analysis was unavailable.")

    with lesion_column:
        with st.container(border=True):
            st.subheader("Lesion diameter")
            if not lesion:
                module_results = st.session_state.get("module_results", {})
                lesion_failure = module_results.get("lesion", {})
                detail = lesion_failure.get(
                    "detail", "Lesion analysis was unavailable."
                )
                st.warning(f"Lesion analysis unavailable: {detail}")
            elif not lesion["lesion_detected"]:
                st.metric("Lesion diameter", "No lesion detected")
            else:
                st.metric(
                    "Lesion diameter",
                    f"{lesion['lesion_diameter_mm']:.2f} mm",
                )
