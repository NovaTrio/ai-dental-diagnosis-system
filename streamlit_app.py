"""Two independent Streamlit workflows for dental radiograph analysis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


API_URL = os.getenv("DENTAL_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 180
IMAGE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
GP_PROTOCOLS = ("nearest-04", "upsize-04", "nearest-06", "upsize-06")


def absolute_url(path: str) -> str:
    return path if path.startswith(("http://", "https://")) else f"{API_URL}/{path.lstrip('/')}"


def api_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.json())
    except (ValueError, AttributeError):
        return response.text or f"HTTP {response.status_code}"
    if isinstance(detail, dict):
        return f"{detail.get('stage')}: {detail.get('message')}"
    return str(detail)


def post(path: str, **kwargs: Any) -> requests.Response:
    try:
        return requests.post(f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as error:
        raise RuntimeError("Could not contact the dental API.") from error


def clear_flow(prefix: str) -> None:
    for key in list(st.session_state):
        if key.startswith(f"{prefix}_") and key not in {
            f"{prefix}_upload",
            f"{prefix}_upload_button",
        }:
            del st.session_state[key]


def load_case_images(prefix: str, case: dict[str, Any]) -> None:
    selection = requests.get(absolute_url(case["selection_image_url"]), timeout=30)
    selection.raise_for_status()
    raw = requests.get(absolute_url(case["raw_image_url"]), timeout=30)
    raw.raise_for_status()
    st.session_state[f"{prefix}_case"] = case
    st.session_state[f"{prefix}_selection_image"] = selection.content
    st.session_state[f"{prefix}_raw_image"] = raw.content


def annotated_image(image_key: str, points_key: str, colors: tuple[str, str]) -> Image.Image:
    image = Image.open(BytesIO(st.session_state[image_key])).convert("RGB")
    draw = ImageDraw.Draw(image)
    points = st.session_state.get(points_key, [])
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[index], outline="white", width=2)
    if len(points) == 2:
        draw.line((points[0], points[1]), fill="lime", width=2)
    return image


def collect_points(prefix: str, case_id: str, image_key: str, colors: tuple[str, str]) -> list[tuple[int, int]]:
    points_key = f"{prefix}_points"
    points = st.session_state.setdefault(points_key, [])
    generation = st.session_state.get(f"{prefix}_generation", 0)
    click = streamlit_image_coordinates(
        annotated_image(image_key, points_key, colors),
        key=f"{prefix}_{case_id}_{generation}",
    )
    if click and len(points) < 2:
        point = (int(click["x"]), int(click["y"]))
        if point != st.session_state.get(f"{prefix}_last"):
            points.append(point)
            st.session_state[f"{prefix}_last"] = point
            st.rerun()
    return points


def reset_points(prefix: str) -> None:
    st.session_state[f"{prefix}_points"] = []
    st.session_state[f"{prefix}_last"] = None
    st.session_state[f"{prefix}_generation"] = st.session_state.get(f"{prefix}_generation", 0) + 1


def upload_section(prefix: str, endpoint: str, description: str) -> dict[str, Any] | None:
    upload = st.file_uploader(description, type=IMAGE_TYPES, key=f"{prefix}_upload")
    if upload is not None:
        st.caption(f"Ready to upload: {upload.name}")
    if st.button(
        "Upload and start",
        disabled=upload is None,
        key=f"{prefix}_upload_button",
        use_container_width=True,
    ):
        assert upload is not None
        clear_flow(prefix)
        with st.spinner("Uploading radiograph..."):
            response = post(
                endpoint,
                files={"image": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")},
            )
            if response.ok:
                load_case_images(prefix, response.json())
                st.rerun()
            else:
                st.error(api_error(response))
    return st.session_state.get(f"{prefix}_case")


def tooth_selection(prefix: str, case: dict[str, Any]) -> bool:
    case_id = case["case_id"]
    st.markdown("#### Step 2 · Select the tooth")
    st.caption(f"Case ID: `{case_id}`")
    st.write("Click the tooth center, then a point along its root direction.")
    points = collect_points(
        f"{prefix}_tooth", case_id, f"{prefix}_selection_image", ("red", "dodgerblue")
    )
    point_status = min(len(points), 2)
    st.progress(point_status / 2, text=f"Selection points: {point_status}/2")
    if st.button("↺ Reset points", disabled=not points, key=f"{prefix}_reset_tooth"):
        reset_points(f"{prefix}_tooth")
        st.rerun()
    if st.button(
        "Confirm tooth selection",
        disabled=len(points) != 2,
        key=f"{prefix}_save_tooth",
        use_container_width=True,
    ):
        response = post(
            f"/api/v1/cases/{case_id}/tooth-selection",
            json={
                "selected_x": points[0][0], "selected_y": points[0][1],
                "direction_x": points[1][0], "direction_y": points[1][1],
            },
        )
        if response.ok:
            st.session_state[f"{prefix}_tooth_saved"] = True
            st.rerun()
        st.error(api_error(response))
    return bool(st.session_state.get(f"{prefix}_tooth_saved"))


def render_working_length_flow() -> None:
    st.markdown("##  Select Your Master GP Cone")
    st.write("Upload your radiograph to estimate working length and receive a master GP cone recommendation.")
    st.markdown("#### Step 1 · Upload the WL radiograph")
    case = upload_section("wl", "/api/v1/cases/working-length", "Choose a dental radiograph")
    if not case or not tooth_selection("wl", case):
        return
    st.markdown("#### Step 3 · Configure and analyze")
    protocol = st.selectbox("GP protocol", GP_PROTOCOLS, index=2, key="wl_gp_protocol")
    if st.button("Estimate WL and select master GP", type="primary", key="wl_analyze", use_container_width=True):
        with st.spinner("Estimating working length..."):
            response = post(
                f"/api/v1/cases/{case['case_id']}/modules/working-length",
                json={"gp_protocol": protocol},
            )
            if response.ok:
                st.session_state.wl_result = response.json()
            else:
                st.error(api_error(response))
    result = st.session_state.get("wl_result")
    if result:
        st.success("Working-length analysis completed")
        st.metric("Predicted working length", f"{result['predicted_working_length_mm']:.2f} mm")
        st.metric("Master GP", result["master_gp_recommendation"])
        st.metric("Recommended ISO file", f"#{result['recommended_iso_file_size_k']}")


def run_diagnostic_modules(case_id: str) -> dict[str, Any]:
    endpoints = {"fracture": "fracture", "lesion": "lesion"}
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(post, f"/api/v1/cases/{case_id}/modules/{endpoint}"): name for name, endpoint in endpoints.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                response = future.result()
                results[name] = response.json() if response.ok else {"error": api_error(response)}
            except Exception as error:
                results[name] = {"error": str(error)}
    return results


def render_diagnostic_flow() -> None:
    st.markdown("##  Check RCT Suitability Indicators")
    st.write("Upload your radiograph to screen for fracture and periapical lesion indicators relevant to RCT planning.")
    st.markdown("#### Step 1 · Upload the diagnostic radiograph")
    case = upload_section("diagnostic", "/api/v1/cases/diagnostic", "Choose a dental radiograph")
    if not case or not tooth_selection("diagnostic", case):
        return
    case_id = case["case_id"]
    st.markdown("#### Step 3 · Calibrate the scale")
    st.write("Click both endpoints of the scale bar.")
    points = collect_points(
        "diagnostic_scale", case_id, "diagnostic_raw_image", ("orange", "magenta")
    )
    known_mm = st.number_input("Known scale length (mm)", min_value=0.1, value=10.0, step=0.5, key="diagnostic_scale_mm")
    st.progress(min(len(points), 2) / 2, text=f"Scale points: {min(len(points), 2)}/2")
    if st.button("↺ Reset scale points", disabled=not points, key="diagnostic_reset_scale"):
        reset_points("diagnostic_scale")
        st.rerun()
    if st.button("Confirm scale calibration", disabled=len(points) != 2, key="diagnostic_save_scale", use_container_width=True):
        response = post(
            f"/api/v1/cases/{case_id}/scale-selection",
            json={"start_x": points[0][0], "start_y": points[0][1], "end_x": points[1][0], "end_y": points[1][1], "known_length_mm": known_mm},
        )
        if response.ok:
            st.session_state.diagnostic_scale_saved = response.json()
            st.rerun()
        st.error(api_error(response))
    calibration = st.session_state.get("diagnostic_scale_saved")
    if not calibration:
        return
    st.success(f"Scale: {calibration['mm_per_pixel']:.6f} mm/pixel")
    st.markdown("#### Step 4 · Run suitability screening")
    if st.button("Check RCT suitability indicators", type="primary", key="diagnostic_analyze", use_container_width=True):
        with st.spinner("Running diagnostic modules..."):
            st.session_state.diagnostic_results = run_diagnostic_modules(case_id)
    results = st.session_state.get("diagnostic_results")
    if not results:
        return
    fracture = results["fracture"]
    if "error" in fracture:
        st.warning(f"Fracture analysis unavailable: {fracture['error']}")
    lesion = results["lesion"]
    if "error" in lesion:
        st.warning(f"Lesion analysis unavailable: {lesion['error']}")
    elif lesion["lesion_detected"]:
        st.metric("Lesion diameter", f"{lesion['lesion_diameter_mm']:.2f} mm")
    else:
        st.metric("Lesion diameter", "No lesion detected")


st.set_page_config(page_title="RCT Support System", page_icon="🦷", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1500px;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
    div.stButton > button {border-radius: 10px; min-height: 2.8rem; font-weight: 600;}
    [data-testid="stFileUploader"] {border: 1px dashed #6c8cff; border-radius: 12px; padding: .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title(" RCT Support System")
st.write("Decision-support tools for root canal treatment planning from dental radiographs.")
st.info("Research-use decision support only. Results require review by a qualified dental professional.")
wl_column, diagnostic_column = st.columns(2, gap="large")
with wl_column:
    with st.container(border=True):
        render_working_length_flow()
with diagnostic_column:
    with st.container(border=True):
        render_diagnostic_flow()
