"""
lesion_measurement_pca_mm.py
─────────────────────────────
Measures lesion orientation and diameters using PCA (Principal Component
Analysis) on the largest lesion contour extracted from binary masks, then
converts every pixel-based measurement to millimeters using the per-image
mm_per_pixel factor produced by scale_calibration_manual.py.

Pipeline:
  1. Read each binary lesion mask from the input folder.
  2. Find the largest external contour (= the lesion boundary).
  3. Apply PCA on the contour points to determine orientation.
  4. Compute major/minor diameters (along PC1/PC2), horizontal/vertical
     diameters, area, perimeter, centroid, and orientation angle (pixels).
  5. Match the mask to its original image and look up mm_per_pixel from
     the calibration CSV, then convert diameters/perimeter/area to mm.
  6. Draw a visualization with axes and labels (pixels + millimeters).
  7. Save all measurements to a CSV file.

Input  : lesion_mask_*.png     from  data/abscess/raw/lesion_postprocessed_restored/
         scale_calibration.csv from  data/abscess/processed/scale_calibration.csv
Output : visualization images  →  data/abscess/raw/lesion_measurements/
         measurements CSV      →  data/abscess/raw/lesion_measurements/lesion_measurements.csv
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import math
import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR        = '../../../data/abscess/raw/lesion_postprocessed_restored'
OUTPUT_DIR       = '../../../data/abscess/raw/lesion_measurements'
CSV_PATH         = os.path.join(OUTPUT_DIR, 'lesion_measurements.csv')
CALIBRATION_CSV  = '../../../data/abscess/processed/scale_calibration.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only process mask files that match this prefix
INPUT_PREFIX = 'lesion_mask_'

# Suffixes appended to the original image id when the mask was produced
# (e.g. "lesion_mask_L22_clahe_sigmoid.png" -> original image id "L22").
ID_SUFFIXES_TO_STRIP = ['_clahe_sigmoid']


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Load calibration data (mm_per_pixel per original image)
# ─────────────────────────────────────────────────────────────────────────────
def load_calibration_lookup(csv_path: str) -> dict:
    """
    Load scale_calibration.csv and build a lookup keyed by the original
    image's base filename (no extension, lowercased) -> mm_per_pixel.

    Rows with a missing/empty mm_per_pixel are ignored.

    Returns
    -------
    lookup : dict[str, float]
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] Calibration CSV not found: {csv_path}")
        return {}

    calibration_df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in calibration_df.iterrows():
        mm_per_pixel = row.get('mm_per_pixel')
        image_name = row.get('image_name')
        if pd.isna(mm_per_pixel) or pd.isna(image_name):
            continue
        base_id = os.path.splitext(str(image_name))[0].lower()
        lookup[base_id] = float(mm_per_pixel)
    return lookup


def resolve_image_id(mask_filename: str) -> str:
    """
    Derive the original image id from a mask filename, e.g.
    "lesion_mask_L22_clahe_sigmoid.png" -> "l22".
    """
    name = mask_filename
    if name.lower().startswith(INPUT_PREFIX.lower()):
        name = name[len(INPUT_PREFIX):]
    name = os.path.splitext(name)[0]
    for suffix in ID_SUFFIXES_TO_STRIP:
        if name.lower().endswith(suffix.lower()):
            name = name[:-len(suffix)]
            break
    return name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Find the largest contour in a binary mask
# ─────────────────────────────────────────────────────────────────────────────
def find_largest_contour(binary_mask: np.ndarray):
    """
    Find external contours in *binary_mask* and return the one with
    the largest area.

    Parameters
    ----------
    binary_mask : np.ndarray (H, W), dtype uint8, values {0, 255}

    Returns
    -------
    largest_contour : np.ndarray or None
        The contour with the maximum area, or None if no contours found.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    # Pick the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Apply PCA and compute all measurements
# ─────────────────────────────────────────────────────────────────────────────
def compute_pca_measurements(contour: np.ndarray) -> dict:
    """
    Apply PCA to the contour points and compute lesion measurements.

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2)
        OpenCV contour array.

    Returns
    -------
    measurements : dict
        Dictionary containing all computed measurements.
    """
    # ── Extract (x, y) points from the contour ───────────────────────────────
    # OpenCV contours have shape (N, 1, 2) → squeeze to (N, 2)
    points = contour.squeeze()          # shape: (N, 2) where columns = [x, y]
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # ── PCA Center (Boundary Mean) ───────────────────────────────────────────
    # Used strictly for centering the contour points for PCA to ensure
    # the major and minor axes are calculated exactly as before.
    pca_center_x = np.mean(x_coords)
    pca_center_y = np.mean(y_coords)

    # ── Lesion Area Centroid (Moments) ───────────────────────────────────────
    # Used for all position-based measurements representing the true
    # center of mass of the lesion area, rather than the contour boundary.
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid_x = M["m10"] / M["m00"]
        centroid_y = M["m01"] / M["m00"]
    else:
        # Fall back to boundary mean if area is zero
        centroid_x = pca_center_x
        centroid_y = pca_center_y

    # ── PCA on contour points ────────────────────────────────────────────────
    # Center the data (subtract mean) using the boundary mean for PCA
    mean_point = np.array([pca_center_x, pca_center_y])
    centered_points = points.astype(np.float64) - mean_point

    # Covariance matrix (2×2)
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # np.linalg.eigh returns eigenvalues in ascending order.
    # PC1 = largest eigenvalue (last), PC2 = smallest (first).
    pc1 = eigenvectors[:, 1]    # Major axis direction
    pc2 = eigenvectors[:, 0]    # Minor axis direction

    # ── Project contour points onto PC1 and PC2 ─────────────────────────────
    projections_pc1 = centered_points @ pc1     # dot product with PC1
    projections_pc2 = centered_points @ pc2     # dot product with PC2

    # ── Diameters ────────────────────────────────────────────────────────────
    # Major diameter = extent along PC1
    major_diameter = float(projections_pc1.max() - projections_pc1.min())

    # Minor diameter = extent along PC2
    minor_diameter = float(projections_pc2.max() - projections_pc2.min())

    # Horizontal diameter = extent along x-axis
    horizontal_diameter = float(x_coords.max() - x_coords.min())

    # Vertical diameter = extent along y-axis
    vertical_diameter = float(y_coords.max() - y_coords.min())

    # ── Orientation angle ────────────────────────────────────────────────────
    # Angle of PC1 (major axis) relative to the positive x-axis
    orientation_angle = math.degrees(math.atan2(pc1[1], pc1[0]))

    # ── Area and Perimeter ───────────────────────────────────────────────────
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    # ── Collect results ──────────────────────────────────────────────────────
    measurements = {
        'area_pixels':                round(area, 2),
        'perimeter_pixels':           round(perimeter, 2),
        'centroid_x':                 round(centroid_x, 2),
        'centroid_y':                 round(centroid_y, 2),
        'orientation_angle_degrees':  round(orientation_angle, 2),
        'major_diameter_pixels':      round(major_diameter, 2),
        'minor_diameter_pixels':      round(minor_diameter, 2),
        'horizontal_diameter_pixels': round(horizontal_diameter, 2),
        'vertical_diameter_pixels':   round(vertical_diameter, 2),
    }

    # Also return PCA vectors and PCA center for visualization
    measurements['_pc1'] = pc1
    measurements['_pc2'] = pc2
    measurements['_proj_pc1_range'] = (projections_pc1.min(), projections_pc1.max())
    measurements['_proj_pc2_range'] = (projections_pc2.min(), projections_pc2.max())
    measurements['_pca_center_x'] = pca_center_x
    measurements['_pca_center_y'] = pca_center_y

    return measurements


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b: Convert pixel measurements to millimeters
# ─────────────────────────────────────────────────────────────────────────────
def add_mm_measurements(measurements: dict, mm_per_pixel: float) -> None:
    """
    Add millimeter-based fields to *measurements* in place, using the
    supplied mm_per_pixel calibration factor.
    """
    measurements['mm_per_pixel'] = round(mm_per_pixel, 6)
    measurements['major_diameter_mm'] = round(
        measurements['major_diameter_pixels'] * mm_per_pixel, 2)
    measurements['minor_diameter_mm'] = round(
        measurements['minor_diameter_pixels'] * mm_per_pixel, 2)
    measurements['horizontal_diameter_mm'] = round(
        measurements['horizontal_diameter_pixels'] * mm_per_pixel, 2)
    measurements['vertical_diameter_mm'] = round(
        measurements['vertical_diameter_pixels'] * mm_per_pixel, 2)
    measurements['perimeter_mm'] = round(
        measurements['perimeter_pixels'] * mm_per_pixel, 2)
    measurements['area_mm2'] = round(
        measurements['area_pixels'] * (mm_per_pixel ** 2), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Draw visualization
# ─────────────────────────────────────────────────────────────────────────────
def draw_visualization(mask: np.ndarray, contour: np.ndarray,
                       measurements: dict) -> np.ndarray:
    """
    Draw the lesion contour, PCA axes, centroid, and diameter labels
    onto a BGR copy of the mask.

    Colors:
        Green  – contour outline
        Red    – major axis (PC1)
        Blue   – minor axis (PC2)
        Yellow – centroid dot

    Returns
    -------
    vis : np.ndarray (H, W, 3), dtype uint8
    """
    # Create a BGR canvas from the grayscale mask
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # ── Draw the contour (green) ─────────────────────────────────────────────
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

    # ── Lesion Area Centroid (yellow dot) ────────────────────────────────────
    # Drawn using the moments-based centroid
    cx = int(round(measurements['centroid_x']))
    cy = int(round(measurements['centroid_y']))
    cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)  # filled yellow circle

    # ── PCA axes ─────────────────────────────────────────────────────────────
    # Drawn using the boundary-point mean (PCA center)
    pca_cx = measurements['_pca_center_x']
    pca_cy = measurements['_pca_center_y']

    pc1 = measurements['_pc1']
    pc2 = measurements['_pc2']
    proj_pc1_min, proj_pc1_max = measurements['_proj_pc1_range']
    proj_pc2_min, proj_pc2_max = measurements['_proj_pc2_range']

    # Major axis endpoints (red)
    major_start = (int(round(pca_cx + pc1[0] * proj_pc1_min)),
                   int(round(pca_cy + pc1[1] * proj_pc1_min)))
    major_end   = (int(round(pca_cx + pc1[0] * proj_pc1_max)),
                   int(round(pca_cy + pc1[1] * proj_pc1_max)))
    cv2.line(vis, major_start, major_end, (0, 0, 255), 2)   # red

    # Minor axis endpoints (blue)
    minor_start = (int(round(pca_cx + pc2[0] * proj_pc2_min)),
                   int(round(pca_cy + pc2[1] * proj_pc2_min)))
    minor_end   = (int(round(pca_cx + pc2[0] * proj_pc2_max)),
                   int(round(pca_cy + pc2[1] * proj_pc2_max)))
    cv2.line(vis, minor_start, minor_end, (255, 0, 0), 2)   # blue

    # ── Labels ───────────────────────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    color      = (255, 255, 255)  # white text

    y_text = 20  # starting y for label block
    line_gap = 18

    has_mm = 'major_diameter_mm' in measurements

    labels = [
        f"Major diam: {measurements['major_diameter_pixels']:.1f} px"
        + (f"  ({measurements['major_diameter_mm']:.2f} mm)" if has_mm else ""),
        f"Minor diam: {measurements['minor_diameter_pixels']:.1f} px"
        + (f"  ({measurements['minor_diameter_mm']:.2f} mm)" if has_mm else ""),
        f"H diam:     {measurements['horizontal_diameter_pixels']:.1f} px",
        f"V diam:     {measurements['vertical_diameter_pixels']:.1f} px",
        f"Angle:      {measurements['orientation_angle_degrees']:.1f} deg",
        f"Area:       {measurements['area_pixels']:.0f} px",
        f"Perimeter:  {measurements['perimeter_pixels']:.1f} px",
    ]
    if not has_mm:
        labels.append("mm: calibration unavailable")

    for label_text in labels:
        cv2.putText(vis, label_text, (10, y_text), font, font_scale,
                    color, thickness, cv2.LINE_AA)
        y_text += line_gap

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Process a single mask image
# ─────────────────────────────────────────────────────────────────────────────
def process_mask(mask_path: str, filename: str, calibration_lookup: dict) -> dict | None:
    """
    Full pipeline for one mask image:
      read → binarize → find contour → PCA → measure → calibrate (mm)
      → visualize → save.

    Returns a dict of measurements (or None if the mask has no lesion).
    """
    print(f"\nProcessing: {filename}")

    # ── Read mask ────────────────────────────────────────────────────────────
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [WARN] Could not read {mask_path} — skipping.")
        return None

    # ── Binarize ─────────────────────────────────────────────────────────────
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ── Find largest contour ─────────────────────────────────────────────────
    contour = find_largest_contour(binary_mask)
    if contour is None or len(contour) < 5:
        print(f"  [WARN] No valid contour found — skipping.")
        return None

    # ── PCA measurements (pixels) ────────────────────────────────────────────
    measurements = compute_pca_measurements(contour)
    measurements['filename'] = filename

    # ── Match to original image and convert to millimeters ──────────────────
    image_id = resolve_image_id(filename)
    mm_per_pixel = calibration_lookup.get(image_id)
    if mm_per_pixel is None:
        print(f"  [WARN] No calibration data found for '{filename}' "
              f"(resolved image id: '{image_id}') — skipping mm conversion.")
    else:
        add_mm_measurements(measurements, mm_per_pixel)

    print(f"  - Centroid         : ({measurements['centroid_x']}, {measurements['centroid_y']})")
    print(f"  - Orientation      : {measurements['orientation_angle_degrees']}°")
    print(f"  - Major diameter   : {measurements['major_diameter_pixels']} px")
    print(f"  - Minor diameter   : {measurements['minor_diameter_pixels']} px")
    print(f"  - H diameter       : {measurements['horizontal_diameter_pixels']} px")
    print(f"  - V diameter       : {measurements['vertical_diameter_pixels']} px")
    print(f"  - Area             : {measurements['area_pixels']} px²")
    print(f"  - Perimeter        : {measurements['perimeter_pixels']} px")
    if mm_per_pixel is not None:
        print(f"  - mm/pixel         : {measurements['mm_per_pixel']}")
        print(f"  - Major diameter   : {measurements['major_diameter_mm']} mm")
        print(f"  - Minor diameter   : {measurements['minor_diameter_mm']} mm")
        print(f"  - Area             : {measurements['area_mm2']} mm²")

    # ── Visualization ────────────────────────────────────────────────────────
    vis = draw_visualization(mask, contour, measurements)
    vis_filename = f"pca_vis_{filename}"
    vis_path = os.path.join(OUTPUT_DIR, vis_filename)
    cv2.imwrite(vis_path, vis)
    print(f"  -> Visualization saved: {vis_path}")

    return measurements


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  Lesion Measurement via PCA (with mm calibration)")
    print("=" * 65)
    print(f"  Input  directory   : {INPUT_DIR}")
    print(f"  Calibration CSV    : {CALIBRATION_CSV}")
    print(f"  Output directory   : {OUTPUT_DIR}")
    print(f"  CSV output         : {CSV_PATH}")
    print("=" * 65)

    supported_exts = ('.png', '.jpg', '.jpeg')

    # Load calibration lookup (image id -> mm_per_pixel)
    calibration_lookup = load_calibration_lookup(CALIBRATION_CSV)
    print(f"\nLoaded calibration for {len(calibration_lookup)} image(s).")

    # Collect mask files
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().startswith(INPUT_PREFIX) and f.lower().endswith(supported_exts)
    ]

    if not files:
        print(f"\n[ERROR] No '{INPUT_PREFIX}*' images found in:\n  {INPUT_DIR}")
        print("        Run lesion_postprocess.py first.")
    else:
        print(f"\nFound {len(files)} mask(s) to process.\n")

        all_measurements = []

        for filename in sorted(files):
            mask_path = os.path.join(INPUT_DIR, filename)
            result = process_mask(mask_path, filename, calibration_lookup)
            if result is not None:
                # Remove internal keys (PCA vectors) before saving to CSV
                csv_row = {k: v for k, v in result.items()
                           if not k.startswith('_')}
                all_measurements.append(csv_row)

        # ── Save CSV ─────────────────────────────────────────────────────────
        if all_measurements:
            # Define column order to match the spec
            columns = [
                'filename',
                'area_pixels',
                'perimeter_pixels',
                'centroid_x',
                'centroid_y',
                'orientation_angle_degrees',
                'major_diameter_pixels',
                'minor_diameter_pixels',
                'horizontal_diameter_pixels',
                'vertical_diameter_pixels',
                'mm_per_pixel',
                'major_diameter_mm',
                'minor_diameter_mm',
                'horizontal_diameter_mm',
                'vertical_diameter_mm',
                'perimeter_mm',
                'area_mm2',
            ]
            df = pd.DataFrame(all_measurements, columns=columns)
            df.to_csv(CSV_PATH, index=False)
            print(f"\n  -> CSV saved: {CSV_PATH}")
            print(f"     ({len(df)} rows)")
        else:
            print("\n  [WARN] No measurements to save.")

    print("\n" + "=" * 65)
    print("  Measurement complete!")
    print("=" * 65)
