"""Streamlit client for the dental working-length API workflow."""

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
    "fracture": "fracture",
    "lesion": "lesion",
}


def api_error(response: requests.Response) -> str:
    """Return the useful FastAPI error message when one is available."""
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
    if path.startswith(("http://", "https://")):
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
            url,
            files=files,
            json=json,
            timeout=REQUEST_TIMEOUT_SECONDS,
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
    """Run the independent diagnosis modules concurrently."""

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
        "points",
        "last_click",
        "click_generation",
        "selection_result",
        "analysis_result",
        "module_results",
    ):
        st.session_state.pop(key, None)


def annotated_selection_image() -> Image.Image:
    image = Image.open(BytesIO(st.session_state.selection_image)).convert("RGB")
    draw = ImageDraw.Draw(image)
    points = st.session_state.get("points", [])
    colors = ("red", "dodgerblue")
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


st.set_page_config(page_title="Dental Working Length", page_icon="🦷", layout="wide")
api_url = DEFAULT_API_URL

st.header("Upload radiograph")
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
                case = response.json()
                image_response = requests.get(
                    absolute_url(api_url, case["selection_image_url"]),
                    timeout=30,
                )
                image_response.raise_for_status()
                st.session_state.case = case
                st.session_state.selection_image = image_response.content
                st.session_state.points = []
                st.session_state.last_click = None
                st.session_state.click_generation = 0
                st.rerun()
        except (requests.RequestException, RuntimeError) as error:
            st.error(str(error))

case = st.session_state.get("case")
if case:
    st.caption(f"Current case ID: `{case['case_id']}`")
    st.header("2. Select the tooth and direction")
    st.write(
        "Click the center of the tooth first. Then click a second point along "
        "the tooth's vertical direction."
    )

    left, right = st.columns([2, 1])
    with left:
        click = streamlit_image_coordinates(
            annotated_selection_image(),
            key=(
                f"tooth_selection_{case['case_id']}_"
                f"{st.session_state.get('click_generation', 0)}"
            ),
        )
        if click:
            current_click = (int(click["x"]), int(click["y"]))
            if (
                current_click != st.session_state.get("last_click")
                and len(st.session_state.points) < 2
            ):
                st.session_state.points.append(current_click)
                st.session_state.last_click = current_click
                st.rerun()

    with right:
        points = st.session_state.points
        if len(points) >= 1:
            st.write(f"Tooth center: **({points[0][0]}, {points[0][1]})**")
        if len(points) == 2:
            st.write(f"Direction: **({points[1][0]}, {points[1][1]})**")
        if st.button("Reset points", disabled=not points):
            st.session_state.points = []
            st.session_state.last_click = None
            st.session_state.click_generation = (
                st.session_state.get("click_generation", 0) + 1
            )
            st.rerun()

        if st.button("Save tooth selection", disabled=len(points) != 2):
            payload = {
                "selected_x": points[0][0],
                "selected_y": points[0][1],
                "direction_x": points[1][0],
                "direction_y": points[1][1],
            }
            with st.spinner("Creating the selected-tooth ROI..."):
                try:
                    response = post(
                        f"{api_url}/api/v1/cases/{case['case_id']}/tooth-selection",
                        json=payload,
                    )
                    if not response.ok:
                        st.error(api_error(response))
                    else:
                        st.session_state.selection_result = response.json()
                        st.session_state.pop("analysis_result", None)
                        st.session_state.pop("module_results", None)
                        st.rerun()
                except RuntimeError as error:
                    st.error(str(error))

selection_result = st.session_state.get("selection_result")
if selection_result:
    st.header("3. Run working-length analysis")
    gp_protocol = st.selectbox(
        "GP protocol",
        GP_PROTOCOLS,
        index=2,
        help="Choose the gutta-percha recommendation protocol.",
    )
    if st.button("Analyze radiograph", type="primary"):
        with st.spinner("Analyzing the radiograph..."):
            module_responses = run_analysis_modules(
                api_url,
                case["case_id"],
                gp_protocol,
            )
            stored_results: dict[str, Any] = {}
            for module_name, response in module_responses.items():
                if isinstance(response, Exception):
                    stored_results[module_name] = {
                        "status": "error",
                        "detail": str(response),
                    }
                elif response.ok:
                    stored_results[module_name] = {
                        "status": "completed",
                        "data": response.json(),
                    }
                else:
                    stored_results[module_name] = {
                        "status": "unavailable",
                        "http_status": response.status_code,
                        "detail": api_error(response),
                    }

            st.session_state.module_results = stored_results
            working_length = stored_results["working_length"]
            if working_length["status"] == "completed":
                st.session_state.analysis_result = working_length["data"]
                st.rerun()
            else:
                st.error(working_length["detail"])

result = st.session_state.get("analysis_result")
if result:
    st.header("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Predicted working length",
        f"{result['predicted_working_length_mm']:.2f} mm",
    )
    col2.metric("Master GP", result["master_gp_recommendation"])
    col3.metric(
        "Recommended ISO file",
        f"#{result['recommended_iso_file_size_k']}",
    )
