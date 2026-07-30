"""Artifact-Aware Adaptive Core-Guided Tooth Segmentation pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.working_length.features.midline_detection import (
    extract_rowwise_midline,
    overlay_midline_on_mask,
)
from src.working_length.segmentation.hybrid_core import (
    competitive_core_reconstruction,
)
from src.working_length.segmentation.metal_detection import (
    create_metal_detection_visualization,
    remove_metal_artifacts,
    remove_metal_fixtures,
)
from src.working_length.segmentation.tooth_mask import (
    adaptive_tooth_threshold,
    clear_side_margins,
    crop_tooth_region,
    enhance_roi,
    extract_tooth_core,
    geometry_constrained_region_expansion,
    make_tooth_white,
    mask_from_isolated_tooth,
    morphology_cleanup,
    normalize_distance_transform,
    refine_tooth_boundary,
    remove_background,
    select_central_component,
    suppress_top_broad_artifacts,
)


def segment_tooth(image: np.ndarray, debug: bool = False) -> dict[str, object]:
    """Run artifact-aware, core-guided segmentation on one tooth ROI."""
    metal_removed, metal_mask = remove_metal_artifacts(image)
    enhanced = enhance_roi(metal_removed)
    binary = adaptive_tooth_threshold(enhanced)
    metal_stripped, fixture_mask = remove_metal_fixtures(enhanced, binary)
    margin_cleared = clear_side_margins(metal_stripped, margin_ratio=0.08)
    cleaned = morphology_cleanup(margin_cleared)
    clamp_suppressed = suppress_top_broad_artifacts(cleaned)
    core_mask, distance_transform = extract_tooth_core(clamp_suppressed)
    selected_core, component_scoring = select_central_component(core_mask)
    selected_core_territory, competitive_reconstruction = (
        competitive_core_reconstruction(
            selected_core,
            core_mask,
            clamp_suppressed,
        )
    )
    geometry_expansion = geometry_constrained_region_expansion(
        selected_core,
        selected_core_territory,
    )
    tooth_mask = refine_tooth_boundary(geometry_expansion)
    cropped_image, cropped_mask, offset = crop_tooth_region(
        enhanced,
        tooth_mask,
        padding_ratio=0.08,
    )
    tooth_only = remove_background(cropped_image, cropped_mask)
    tooth_white = make_tooth_white(cropped_mask)
    path, length = extract_rowwise_midline(
        cropped_mask,
        smooth_factor=60,
        min_row_pixels=5,
    )
    final_midline = overlay_midline_on_mask(cropped_mask, path)
    normalized_distance = normalize_distance_transform(distance_transform)
    metal_detection = create_metal_detection_visualization(
        image,
        metal_mask,
        fixture_mask,
    )

    if debug:
        print(f"  Metal pixels: {cv2.countNonZero(metal_mask)}")
        print(f"  Metal fixture pixels: {cv2.countNonZero(fixture_mask)}")
        print(f"  Binary pixels: {cv2.countNonZero(binary)}")
        print(f"  Core pixels: {cv2.countNonZero(core_mask)}")
        print(f"  Selected core pixels: {cv2.countNonZero(selected_core)}")
        print(f"  Tooth mask pixels: {cv2.countNonZero(tooth_mask)}")
        print(f"  Midline points: {len(path)}")
        print(f"  Midline length: {length:.2f} pixels")

    return {
        "original": image,
        "metal_detection": metal_detection,
        "metal_removed": metal_removed,
        "metal_mask": metal_mask,
        "enhanced": enhanced,
        "binary": binary,
        "metal_fixture_mask": fixture_mask,
        "metal_stripped_binary": metal_stripped,
        "margin_cleared": margin_cleared,
        "cleaned": cleaned,
        "clamp_suppressed": clamp_suppressed,
        "distance_transform": normalized_distance,
        "core_mask": core_mask,
        "component_scoring": component_scoring,
        "selected_core": selected_core,
        "selected_core_territory": selected_core_territory,
        "competitive_reconstruction": competitive_reconstruction,
        "geometry_expansion": geometry_expansion,
        "boundary_refined": tooth_mask,
        "tooth_mask": tooth_mask,
        "cropped_img": cropped_image,
        "cropped_mask": cropped_mask,
        "tooth_only": tooth_only,
        "tooth_white": tooth_white,
        "final_midline": final_midline,
        "midline_path": path,
        "midline_length": length,
        "crop_offset": offset,
    }


def segment_clean_roi(
    image: np.ndarray,
    mask_path: str | Path | None = None,
    debug: bool = False,
) -> dict[str, object]:
    """Measure an already isolated ROI or one paired with an existing mask."""
    tooth_mask = None
    if mask_path is not None and Path(mask_path).exists():
        tooth_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if tooth_mask is None:
        tooth_mask = mask_from_isolated_tooth(image)
    else:
        _, tooth_mask = cv2.threshold(tooth_mask, 127, 255, cv2.THRESH_BINARY)

    cropped_image, cropped_mask, offset = crop_tooth_region(
        image,
        tooth_mask,
        padding_ratio=0.03,
    )
    tooth_only = remove_background(cropped_image, cropped_mask)
    tooth_white = make_tooth_white(cropped_mask)
    path, length = extract_rowwise_midline(cropped_mask)
    final_midline = overlay_midline_on_mask(cropped_mask, path)

    if debug:
        print(f"  Clean ROI mask pixels: {cv2.countNonZero(tooth_mask)}")
        print(f"  Midline length: {length:.2f} pixels")

    return {
        "original": image,
        "tooth_mask": tooth_mask,
        "cropped_img": cropped_image,
        "cropped_mask": cropped_mask,
        "tooth_only": tooth_only,
        "tooth_white": tooth_white,
        "final_midline": final_midline,
        "midline_path": path,
        "midline_length": length,
        "crop_offset": offset,
    }
