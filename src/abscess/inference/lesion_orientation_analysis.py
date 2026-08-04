"""
lesion_orientation_analysis.py
──────────────────────────────
Calculates lesion orientation relative to the target tooth long axis.
Integrates outputs from tooth_orientation.py and lesion_measurement_pca.py.

Inputs:
  - BlackRemove images (for tooth axis estimation)
  - lesion_postprocessed_restored masks (for lesion PCA)

Outputs:
  - Visualizations showing the axes, projection, and perpendicular distance
  - CSV file with measurements
"""

import os
import math
import cv2
import numpy as np
import pandas as pd
import sys

# Add project root to path so we can import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

# Import existing logic
from src.abscess.preprocessing.tooth_orientation import get_tooth_midpoints_and_axis
from src.abscess.inference.lesion_measurement_pca import find_largest_contour, compute_pca_measurements

# Configuration
TOOTH_IMG_DIR = os.path.join(project_root, 'data', 'abscess', 'raw', 'BlackRemove')
LESION_MASK_DIR = os.path.join(project_root, 'data', 'abscess', 'raw', 'lesion_postprocessed_restored')
OUTPUT_DIR = os.path.join(project_root, 'data', 'abscess', 'raw', 'lesion_orientation')
CSV_PATH = os.path.join(OUTPUT_DIR, 'lesion_orientation.csv')

SWAP_LEFT_RIGHT = False  # Configurable flag to swap left/right labels if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Math & Geometry Functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_vector(v):
    """Normalize a 2D vector."""
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([0.0, 0.0])
    return v / norm

def calculate_relative_angle(tooth_vector, lesion_vector):
    """Calculate the acute angle between tooth long-axis and lesion major-axis (0-90 deg)."""
    v1 = normalize_vector(tooth_vector)
    v2 = normalize_vector(lesion_vector)
    
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
        
    dot_prod = abs(np.dot(v1, v2))
    # Clip to avoid floating point issues
    dot_prod = min(1.0, max(0.0, dot_prod))
    
    theta_rad = math.acos(dot_prod)
    return math.degrees(theta_rad)

def calculate_centroid_side(tooth_vector, tooth_point, centroid, swap_labels=False):
    """
    Calculate whether the lesion centroid is on the left, right, or center 
    using the signed 2D cross product.
    """
    vx, vy = tooth_vector
    x0, y0 = tooth_point
    cx, cy = centroid
    
    side_value = vx * (cy - y0) - vy * (cx - x0)
    
    if abs(side_value) < 1e-6:
        return "Center"
        
    # Standard math: if side_value > 0, it's on one side, < 0 on the other.
    # Because OpenCV y increases downward, the visual left/right might be flipped.
    if side_value > 0:
        return "Right" if not swap_labels else "Left"
    else:
        return "Left" if not swap_labels else "Right"

def calculate_perpendicular_distance(tooth_vector, tooth_point, centroid):
    """Calculate the perpendicular distance from the lesion centroid to the tooth long axis."""
    vx, vy = tooth_vector
    x0, y0 = tooth_point
    cx, cy = centroid
    
    norm = np.linalg.norm(tooth_vector)
    if norm == 0:
        return 0.0
        
    distance = abs(vx * (cy - y0) - vy * (cx - x0)) / norm
    return distance

def calculate_projection_point(tooth_vector, tooth_point, centroid):
    """Calculate the projection point of the lesion centroid onto the tooth long axis."""
    v = normalize_vector(tooth_vector)
    if np.linalg.norm(v) == 0:
        return centroid
        
    p0 = np.array(tooth_point, dtype=float)
    p = np.array(centroid, dtype=float)
    
    w = p - p0
    proj_len = np.dot(w, v)
    proj_pt = p0 + proj_len * v
    return proj_pt[0], proj_pt[1]

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_orientation_visualization(image, tooth_vector, tooth_point, lesion_vector, 
                                  centroid, proj_point, rel_angle, side, distance,
                                  pca_center, proj_pc1_range):
    """Draw all the required components on the original image."""
    # Ensure canvas is BGR
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        vis = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        vis = image.copy()
        
    h, w = vis.shape[:2]
    
    # Int conversions for drawing
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    px, py = int(round(proj_point[0])), int(round(proj_point[1]))
    
    # 1. Draw tooth axis (Green line)
    vx, vy = tooth_vector
    if abs(vy) > 1e-6:
        x0, y0 = tooth_point
        y1, y2 = 0, h - 1
        x1 = int(x0 + vx*(y1 - y0)/vy)
        x2 = int(x0 + vx*(y2 - y0)/vy)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 2. Draw lesion major axis (Red line using true PCA center and projection limits)
    lx, ly = normalize_vector(lesion_vector)
    pca_cx, pca_cy = pca_center
    proj_min, proj_max = proj_pc1_range
    
    lx1 = int(round(pca_cx + lx * proj_min))
    ly1 = int(round(pca_cy + ly * proj_min))
    lx2 = int(round(pca_cx + lx * proj_max))
    ly2 = int(round(pca_cy + ly * proj_max))
    cv2.line(vis, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
    
    # 3. Draw perpendicular line (Blue dashed/thin line)
    cv2.line(vis, (cx, cy), (px, py), (255, 0, 0), 2)
    
    # 4. Draw points
    cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1) # Centroid: Yellow
    cv2.circle(vis, (px, py), 5, (255, 0, 255), -1) # Projection: Magenta
    
    # 5. Draw text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, f"Rel Angle: {rel_angle:.1f} deg", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Side: {side}", (10, 60), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Dist: {distance:.1f} px", (10, 90), font, 0.7, (255, 255, 255), 2)
    
    return vis

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_image_pair(tooth_img_path, lesion_mask_path, filename):
    """Process a single image pair (tooth and lesion) to extract orientation."""
    
    # 1. Read images
    tooth_img = cv2.imread(tooth_img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(lesion_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if tooth_img is None or mask is None:
        print(f"[WARN] Missing or unreadable files for {filename}")
        return None
        
    # 2. Get tooth axis
    # get_tooth_midpoints_and_axis returns: midpoints, (vx, vy, x0, y0), angle_deg
    tooth_res = get_tooth_midpoints_and_axis(tooth_img, middle_region_ratio=0.6)
    if tooth_res[1] is None:
        print(f"[WARN] Could not estimate tooth axis for {filename}")
        return None
        
    _, tooth_line_params, tooth_angle = tooth_res
    tooth_vx, tooth_vy, tooth_x0, tooth_y0 = tooth_line_params
    tooth_vector = (tooth_vx, tooth_vy)
    tooth_point = (tooth_x0, tooth_y0)
    
    # 3. Get lesion PCA measurements
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contour = find_largest_contour(binary_mask)
    if contour is None or len(contour) < 5:
        print(f"[WARN] No valid lesion contour found for {filename}")
        return None
        
    lesion_meas = compute_pca_measurements(contour)
    lesion_vector = lesion_meas['_pc1']
    centroid = (lesion_meas['centroid_x'], lesion_meas['centroid_y'])
    lesion_angle = lesion_meas['orientation_angle_degrees']
    
    # 4. Calculate integrated orientation metrics
    rel_angle = calculate_relative_angle(tooth_vector, lesion_vector)
    side = calculate_centroid_side(tooth_vector, tooth_point, centroid, SWAP_LEFT_RIGHT)
    distance = calculate_perpendicular_distance(tooth_vector, tooth_point, centroid)
    proj_x, proj_y = calculate_projection_point(tooth_vector, tooth_point, centroid)
    
    # 5. Visualization
    pca_center = (lesion_meas['_pca_center_x'], lesion_meas['_pca_center_y'])
    proj_pc1_range = lesion_meas['_proj_pc1_range']
    vis = draw_orientation_visualization(
        tooth_img, tooth_vector, tooth_point, lesion_vector, 
        centroid, (proj_x, proj_y), rel_angle, side, distance,
        pca_center, proj_pc1_range
    )
    
    vis_filename = f"orient_vis_{filename}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, vis_filename), vis)
    
    # 6. Return results dict for CSV
    return {
        'filename': filename,
        'tooth_axis_angle_degrees': round(tooth_angle, 2),
        'lesion_axis_angle_degrees': round(lesion_angle, 2),
        'relative_orientation_degrees': round(rel_angle, 2),
        'lesion_side': side,
        'distance_to_tooth_axis_pixels': round(distance, 2),
        'lesion_centroid_x': round(centroid[0], 2),
        'lesion_centroid_y': round(centroid[1], 2),
        'projection_x': round(proj_x, 2),
        'projection_y': round(proj_y, 2)
    }

def main():
    print("=" * 65)
    print("  Lesion Orientation Analysis")
    print("=" * 65)
    
    supported_exts = ('.png', '.jpg', '.jpeg')
    results = []
    
    # Process all images in the tooth directory
    for filename in sorted(os.listdir(TOOTH_IMG_DIR)):
        if not filename.lower().endswith(supported_exts):
            continue
            
        tooth_path = os.path.join(TOOTH_IMG_DIR, filename)
        
        # Determine corresponding mask path
        # Assuming mask name format: "lesion_mask_" + filename
        mask_filename = f"lesion_mask_{filename}"
        mask_path = os.path.join(LESION_MASK_DIR, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"[SKIP] Mask not found: {mask_filename}")
            continue
            
        print(f"Processing {filename}...")
        res = process_image_pair(tooth_path, mask_path, filename)
        if res:
            results.append(res)
            
    # Save CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n[SUCCESS] Processed {len(results)} images. Saved to {CSV_PATH}")
    else:
        print("\n[WARN] No results generated.")

if __name__ == "__main__":
    main()
