"""Automatic detection and PCA-based measurement of a 10 mm image marker."""

import os
import cv2
import numpy as np
import pandas as pd


# Configuration
INPUT_DIR = "../../../data/abscess/raw/images"
OUTPUT_VIS_DIR = "../../../data/abscess/processed/calibration_visualizations"
OUTPUT_CSV = "../../../data/abscess/processed/calibration_results.csv"

MARKER_LENGTH_MM = 10.0
ROI_FRACTION = 0.20
MIN_HEIGHT_PIXELS = 50
MIN_AREA_PIXELS = 100.0
MIN_CONFIDENCE = 0.58

# Scores sum to one. Shape gets three independent descriptors before fusion.
SCORE_WEIGHTS = {
    "vertical_orientation_score": 0.20,
    "aspect_ratio_score": 0.18,
    "border_proximity_score": 0.15,
    "straightness_score": 0.20,
    "rectangularity_score": 0.15,
    "length_consistency_score": 0.12,
}


def _clip01(value):
    """Clamp a scalar to the closed interval [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def compute_pca_geometry(contour):
    """Measure a contour using the same projection geometry as the lesion module."""
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 2:
        return None

    center = points.mean(axis=0)
    centered_points = points - center
    covariance = np.cov(centered_points, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered_points @ principal_axis
    projection_min = float(projections.min())
    projection_max = float(projections.max())
    endpoint_1 = center + principal_axis * projection_min
    endpoint_2 = center + principal_axis * projection_max
    length_pixels = float(np.linalg.norm(endpoint_2 - endpoint_1))

    # Distance from p to the infinite PCA line is its PC2 component.
    perpendicular = centered_points - np.outer(projections, principal_axis)
    mean_perpendicular_deviation = float(
        np.mean(np.linalg.norm(perpendicular, axis=1))
    )

    return {
        "center": center,
        "principal_axis": principal_axis,
        "projection_range": (projection_min, projection_max),
        "endpoint_1": endpoint_1,
        "endpoint_2": endpoint_2,
        "length_pixels": length_pixels,
        "mean_perpendicular_deviation": mean_perpendicular_deviation,
    }


def _candidate_features(contour, roi_width, image_height):
    """Return geometric descriptors and normalized candidate scores."""
    x, y, width, height = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))
    if width <= 0 or height <= 0 or area <= 0:
        return None

    pca = compute_pca_geometry(contour)
    if pca is None or pca["length_pixels"] <= 0:
        return None

    aspect_ratio = float(height) / float(width)
    extent = area / float(width * height)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    rotated_rect = cv2.minAreaRect(contour)
    rotated_width, rotated_height = rotated_rect[1]
    rotated_area = float(rotated_width * rotated_height)
    rectangularity = area / rotated_area if rotated_area > 0 else 0.0
    shape_consistency = float(np.mean([
        _clip01(extent), _clip01(solidity), _clip01(rectangularity)
    ]))

    axis = pca["principal_axis"]
    vertical_score = _clip01(abs(float(axis[1])))
    aspect_score = _clip01((aspect_ratio - 2.0) / 6.0)
    distance_from_right = max(0.0, float(roi_width - (x + width)))
    border_score = _clip01(1.0 - distance_from_right / max(1.0, 0.15 * roi_width))

    # Normalize mean line deviation by PCA length; <=1% is excellent, >=8% poor.
    relative_deviation = pca["mean_perpendicular_deviation"] / pca["length_pixels"]
    straightness_score = _clip01((0.08 - relative_deviation) / 0.07)

    # A valid ruler should span at least 8% but not almost the whole image height.
    length_fraction = pca["length_pixels"] / max(1.0, float(image_height))
    minimum_length_score = _clip01(length_fraction / 0.08)
    excessive_length_penalty = _clip01((0.95 - length_fraction) / 0.15)
    length_score = minimum_length_score * excessive_length_penalty

    scores = {
        "vertical_orientation_score": vertical_score,
        "aspect_ratio_score": aspect_score,
        "border_proximity_score": border_score,
        "straightness_score": straightness_score,
        "rectangularity_score": shape_consistency,
        "length_consistency_score": length_score,
    }
    confidence = sum(SCORE_WEIGHTS[name] * value for name, value in scores.items())

    return {
        "contour": contour,
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "candidate_area": area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "rectangularity": rectangularity,
        "shape_consistency": shape_consistency,
        "distance_from_right": distance_from_right,
        "straightness": straightness_score,
        "pca": pca,
        "confidence_score": _clip01(confidence),
        **scores,
    }


def _validation_failures(candidate, roi_width, image_height):
    """List explicit marker quality checks that a candidate fails."""
    return [name for name, passed in {
        "right-border proximity": candidate["distance_from_right"] <= 0.15 * roi_width,
        "vertical orientation": candidate["vertical_orientation_score"] >= 0.90,
        "aspect ratio": candidate["aspect_ratio"] >= 3.0,
        "sufficient length": candidate["pca"]["length_pixels"] >= max(
            MIN_HEIGHT_PIXELS, 0.06 * image_height
        ),
        "straightness": candidate["straightness"] >= 0.50,
        "shape consistency": candidate["shape_consistency"] >= 0.60,
        "confidence": candidate["confidence_score"] >= MIN_CONFIDENCE,
    }.items() if not passed]


def _failure_payload(image, status, candidate=None):
    """Create a stable result schema for unsuccessful calibration."""
    metrics = {
        "aspect_ratio": None,
        "extent": None,
        "solidity": None,
        "rectangularity": None,
        "straightness": None,
        "candidate_area": None,
        "bounding_box_width": None,
        "bounding_box_height": None,
    }
    if candidate is not None:
        metrics.update({
            "aspect_ratio": round(candidate["aspect_ratio"], 4),
            "extent": round(candidate["extent"], 4),
            "solidity": round(candidate["solidity"], 4),
            "rectangularity": round(candidate["rectangularity"], 4),
            "straightness": round(candidate["straightness"], 4),
            "candidate_area": round(candidate["candidate_area"], 2),
            "bounding_box_width": candidate["w"],
            "bounding_box_height": candidate["h"],
        })
    return {
        "marker_start_x": None,
        "marker_start_y": None,
        "marker_end_x": None,
        "marker_end_y": None,
        "marker_length_pixels": None,
        "mm_per_pixel": None,
        **metrics,
        "confidence_score": round(candidate["confidence_score"], 4) if candidate else 0.0,
        "marker_detected": False,
        "calibration_status": status,
        "img_vis": image,
    }


def calibrate_image(image_path):
    """Detect the 10 mm marker and compute the millimetres-per-pixel factor."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    visualization = image.copy()
    image_height, image_width = image.shape[:2]
    roi_width = max(1, int(image_width * ROI_FRACTION))
    roi_start_x = image_width - roi_width
    roi = image[:, roi_start_x:image_width]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_roi = clahe.apply(gray_roi)
    adaptive_mask = cv2.adaptiveThreshold(
        enhanced_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -5
    )
    _, otsu_mask = cv2.threshold(
        enhanced_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    combined_mask = cv2.bitwise_or(adaptive_mask, otsu_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    morph = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if height < MIN_HEIGHT_PIXELS or area < MIN_AREA_PIXELS:
            continue
        candidate = _candidate_features(contour, roi_width, image_height)
        if candidate is not None:
            candidates.append(candidate)

    cv2.rectangle(
        visualization, (roi_start_x, 0), (image_width - 1, image_height - 1),
        (255, 255, 0), 2
    )
    cv2.putText(visualization, "Calibration ROI", (max(5, roi_start_x + 5), 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

    if not candidates:
        return _failure_payload(
            visualization, "Failed: No calibration marker candidates found"
        )

    valid_candidates = [
        candidate for candidate in candidates
        if not _validation_failures(candidate, roi_width, image_height)
    ]
    best = max(valid_candidates or candidates,
               key=lambda item: item["confidence_score"])
    global_contour = best["contour"].copy()
    global_contour[:, :, 0] += roi_start_x
    cv2.drawContours(visualization, [global_contour], -1, (0, 255, 0), 2)

    if not valid_candidates:
        failures = _validation_failures(best, roi_width, image_height)
        status = "Failed validation: " + ", ".join(failures)
        cv2.putText(visualization, status, (10, image_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        return _failure_payload(visualization, status, best)

    endpoint_1 = best["pca"]["endpoint_1"] + np.array([roi_start_x, 0.0])
    endpoint_2 = best["pca"]["endpoint_2"] + np.array([roi_start_x, 0.0])
    # Stable output ordering: start is the visually upper PCA endpoint.
    if endpoint_1[1] > endpoint_2[1]:
        endpoint_1, endpoint_2 = endpoint_2, endpoint_1
    start = tuple(np.rint(endpoint_1).astype(int))
    end = tuple(np.rint(endpoint_2).astype(int))
    marker_length_pixels = float(np.linalg.norm(endpoint_2 - endpoint_1))
    mm_per_pixel = MARKER_LENGTH_MM / marker_length_pixels

    cv2.line(visualization, start, end, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(visualization, start, 5, (0, 255, 255), -1)
    cv2.circle(visualization, end, 5, (255, 0, 255), -1)
    labels = [
        f"PCA length: {marker_length_pixels:.1f} px",
        f"Scale: {mm_per_pixel:.4f} mm/px",
        f"Confidence: {best['confidence_score']:.3f}",
    ]
    text_x = max(10, min(start[0], end[0]) - 330)
    text_y = max(25, min(start[1], end[1]))
    for label in labels:
        cv2.putText(visualization, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        text_y += 24

    return {
        "marker_start_x": start[0],
        "marker_start_y": start[1],
        "marker_end_x": end[0],
        "marker_end_y": end[1],
        "marker_length_pixels": round(marker_length_pixels, 2),
        "mm_per_pixel": round(mm_per_pixel, 4),
        "aspect_ratio": round(best["aspect_ratio"], 4),
        "extent": round(best["extent"], 4),
        "solidity": round(best["solidity"], 4),
        "rectangularity": round(best["rectangularity"], 4),
        "straightness": round(best["straightness"], 4),
        "candidate_area": round(best["candidate_area"], 2),
        "bounding_box_width": best["w"],
        "bounding_box_height": best["h"],
        "confidence_score": round(best["confidence_score"], 4),
        "marker_detected": True,
        "calibration_status": "Success",
        "img_vis": visualization,
    }


def main():
    print("=" * 60)
    print("  Automatic Image Calibration")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_abs_dir = os.path.normpath(os.path.join(script_dir, INPUT_DIR))
    if not os.path.exists(input_abs_dir):
        print(f"[ERROR] Input directory not found: {input_abs_dir}")
        return

    valid_extensions = (".jpg", ".jpeg", ".png")
    files = [name for name in os.listdir(input_abs_dir)
             if name.lower().endswith(valid_extensions)]
    if not files:
        print(f"[ERROR] No valid images found in {input_abs_dir}")
        return

    print(f"Found {len(files)} images to process.\n")
    visualization_dir = os.path.normpath(os.path.join(script_dir, OUTPUT_VIS_DIR))
    csv_path = os.path.normpath(os.path.join(script_dir, OUTPUT_CSV))
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    results = []
    for filename in sorted(files):
        print(f"Processing: {filename}")
        result = calibrate_image(os.path.join(input_abs_dir, filename))
        if result is None:
            print(f"  -> Could not load {filename}")
            continue
        print(f"  -> Status: {result['calibration_status']}")
        if result["marker_detected"]:
            print(f"  -> mm/px : {result['mm_per_pixel']}")

        cv2.imwrite(os.path.join(visualization_dir, f"calib_vis_{filename}"),
                    result["img_vis"])
        csv_row = {key: value for key, value in result.items() if key != "img_vis"}
        csv_row["filename"] = filename
        results.append(csv_row)

    if results:
        columns = [
            "filename", "marker_start_x", "marker_start_y", "marker_end_x",
            "marker_end_y", "marker_length_pixels", "mm_per_pixel",
            "aspect_ratio", "extent", "solidity", "rectangularity",
            "straightness", "candidate_area", "bounding_box_width",
            "bounding_box_height", "confidence_score", "marker_detected",
            "calibration_status",
        ]
        pd.DataFrame(results, columns=columns).to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Processed {len(results)} images.")
        print(f"Results saved to: {csv_path}")
        print(f"Visualizations saved to: {visualization_dir}")


if __name__ == "__main__":
    main()
