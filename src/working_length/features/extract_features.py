"""
Feature Extraction for Working Length Prediction

Pipeline:
  Tooth Mask → Medial Axis Transform → Skeleton Pruning →
  Polynomial Smoothing → Midline Length Measurement
"""

import cv2
import numpy as np
import os
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from skimage import morphology, measure
from src.common.preprocessing.base_preprocess import get_dataset_path


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1: TOOTH MASK EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_tooth_mask(tooth_img_normalized, otsu_threshold=True, debug=False):
    """
    Creates binary mask of tooth from normalized grayscale image (0-1).
    
    Args:
        tooth_img_normalized: float array [0, 1]
        otsu_threshold: if True, use Otsu's method; else use fixed threshold
        debug: show intermediate steps
    
    Returns:
        binary_mask: bool array (True = tooth, False = background)
    """
    # Convert to 8-bit for OpenCV operations
    img_8bit = (tooth_img_normalized * 255).astype(np.uint8)
    
    if otsu_threshold:
        # Otsu's thresholding — adaptive to image characteristics
        _, mask = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Fixed threshold (mid-gray)
        _, mask = cv2.threshold(img_8bit, 128, 255, cv2.THRESH_BINARY)
    
    mask = mask > 0  # Convert to boolean
    
    if debug:
        cv2.imshow("Tooth Mask", (mask * 255).astype(np.uint8))
        cv2.waitKey(0)
    
    return mask


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2: MEDIAL AXIS TRANSFORM (SKELETON)
# ═════════════════════════════════════════════════════════════════════════════

def compute_skeleton(mask, debug=False):
    """
    Computes medial axis (skeleton) of tooth using scikit-image.
    
    Args:
        mask: binary array (True = tooth)
        debug: show skeleton
    
    Returns:
        skeleton: bool array (True = skeleton pixels)
        dist_transform: distance transform (used for pruning)
    """
    # Medial axis transform
    skeleton = morphology.medial_axis(mask)
    
    # Distance transform for each pixel (used in pruning)
    dist_transform = ndimage.distance_transform_edt(mask)
    
    if debug:
        cv2.imshow("Skeleton", (skeleton * 255).astype(np.uint8))
        cv2.waitKey(0)
    
    return skeleton, dist_transform


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3: SKELETON PRUNING
# ═════════════════════════════════════════════════════════════════════════════

def prune_skeleton(skeleton, dist_transform, min_radius=2, debug=False):
    """
    Removes weak/spurious branches from skeleton by keeping only pixels
    with distance > min_radius. Keeps main root structure.
    
    Args:
        skeleton: bool array
        dist_transform: distance values from edge
        min_radius: minimum distance from boundary (in pixels)
        debug: show pruned result
    
    Returns:
        pruned_skeleton: bool array (main root structure)
    """
    pruned = skeleton.copy()
    pruned[dist_transform < min_radius] = False
    
    if debug:
        cv2.imshow("Pruned Skeleton", (pruned * 255).astype(np.uint8))
        cv2.waitKey(0)
    
    return pruned


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4: EXTRACT CENTERLINE & APPLY POLYNOMIAL SMOOTHING
# ═════════════════════════════════════════════════════════════════════════════

def extract_centerline_points(skeleton, debug=False):
    """
    Extracts coordinates of skeleton pixels.
    
    Args:
        skeleton: bool array
        debug: show points
    
    Returns:
        points: (N, 2) array of (y, x) coordinates
    """
    points = np.column_stack(np.where(skeleton))  # (y, x) pairs
    
    return points


def order_centerline_points(points):
    """
    Orders skeleton points from apical (tip) to coronal (crown).
    Uses distance from bottom-right (apical region).
    
    Args:
        points: (N, 2) array of (y, x) coordinates
    
    Returns:
        ordered_points: (N, 2) sorted from tip to crown
    """
    if len(points) == 0:
        return points
    
    # Distance from bottom-right corner (typical apical position)
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    order = np.argsort(distances)[::-1]  # Descending (bottom to top)
    
    return points[order]


def smooth_centerline_polynomial(points, k=3, s=100.0, debug=False):
    """
    Fits polynomial spline through centerline points for smooth curve.
    
    Args:
        points: (N, 2) array of (y, x) coordinates [ordered]
        k: spline degree (1-5, default 3=cubic)
        s: smoothing factor (higher = smoother)
        debug: show comparison
    
    Returns:
        smooth_curve: (M, 2) resampled smooth centerline
    """
    if len(points) < k + 1:
        return points
    
    # Normalize for spline fitting
    y_orig = points[:, 0]
    x_orig = points[:, 1]
    
    # Fit spline: y = f(x)
    try:
        spline = UnivariateSpline(x_orig, y_orig, k=k, s=s)
    except:
        # If fitting fails, return original
        return points
    
    # Resample along x-axis
    x_smooth = np.linspace(x_orig.min(), x_orig.max(), len(points) * 2)
    y_smooth = spline(x_smooth)
    
    smooth_curve = np.column_stack([y_smooth, x_smooth])
    
    if debug:
        print(f"  Original points: {len(points)}")
        print(f"  Smoothed points: {len(smooth_curve)}")
    
    return smooth_curve


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5: MIDLINE LENGTH MEASUREMENT
# ═════════════════════════════════════════════════════════════════════════════

def measure_centerline_length(smooth_curve):
    """
    Calculates arc length of smoothed centerline in pixels.
    
    Args:
        smooth_curve: (N, 2) array of (y, x) coordinates
    
    Returns:
        length_px: total length in pixels
    """
    if len(smooth_curve) < 2:
        return 0.0
    
    # Compute distances between consecutive points
    diffs = np.diff(smooth_curve, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    
    length_px = np.sum(distances)
    
    return length_px


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN FEATURE EXTRACTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def extract_midline_length(tooth_img_normalized, debug=False):
    """
    Full pipeline: Mask → Skeleton → Prune → Smooth → Measure
    
    Args:
        tooth_img_normalized: float array [0, 1], single tooth ROI
        debug: show intermediate steps
    
    Returns:
        midline_length_px: length of tooth centerline in pixels
        diagnostics: dict with intermediate results (for debugging)
    """
    diagnostics = {}
    
    # Step 1: Extract tooth mask
    mask = extract_tooth_mask(tooth_img_normalized, otsu_threshold=True, debug=debug)
    diagnostics['mask'] = mask
    
    if not mask.any():
        print("  WARNING: Empty mask! No tooth detected.")
        return 0.0, diagnostics
    
    # Step 2: Compute skeleton
    skeleton, dist_transform = compute_skeleton(mask, debug=debug)
    diagnostics['skeleton'] = skeleton
    diagnostics['dist_transform'] = dist_transform
    
    if not skeleton.any():
        print("  WARNING: Empty skeleton! Mask might be too thin.")
        return 0.0, diagnostics
    
    # Step 3: Prune skeleton (remove spurious branches)
    pruned = prune_skeleton(skeleton, dist_transform, min_radius=2, debug=False)
    diagnostics['pruned_skeleton'] = pruned
    
    if not pruned.any():
        # If pruning removed everything, use unpruned skeleton
        print("  WARNING: Pruning removed all points. Using unpruned skeleton.")
        pruned = skeleton
    
    # Step 4: Extract and order centerline points
    points = extract_centerline_points(pruned, debug=False)
    ordered_points = order_centerline_points(points)
    diagnostics['centerline_points'] = ordered_points
    
    # Step 5: Smooth centerline
    smooth_curve = smooth_centerline_polynomial(ordered_points, k=3, s=50.0, debug=False)
    diagnostics['smooth_curve'] = smooth_curve
    
    # Step 6: Measure length
    midline_length_px = measure_centerline_length(smooth_curve)
    
    if debug:
        show_pipeline_debug_panel(tooth_img_normalized, diagnostics)
    
    return midline_length_px, diagnostics


def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return image


def resize_image_for_display(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def annotate_image(image, text):
    annotated = image.copy()
    cv2.putText(annotated, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return annotated


def overlay_centerline(image, smooth_curve, points=None):
    vis = to_bgr(image)
    if smooth_curve is not None and len(smooth_curve) > 0:
        pts = np.round(np.column_stack((smooth_curve[:, 1], smooth_curve[:, 0]))).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, (0, 0, 255), 2)
        for x, y in pts[::max(1, len(pts)//100)]:
            cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)
    if points is not None and len(points) > 0:
        pts2 = np.round(np.column_stack((points[:, 1], points[:, 0]))).astype(np.int32)
        if len(pts2) > 1:
            cv2.polylines(vis, [pts2], False, (0, 255, 0), 1)
    return vis


def show_pipeline_debug_panel(tooth_img_normalized, diagnostics):
    mask = diagnostics['mask'].astype(np.uint8)
    skeleton = diagnostics['skeleton'].astype(np.uint8)
    pruned = diagnostics['pruned_skeleton'].astype(np.uint8)
    smooth_curve = diagnostics.get('smooth_curve', np.zeros((0, 2), dtype=np.float32))
    centerline_points = diagnostics.get('centerline_points', np.zeros((0, 2), dtype=np.int32))

    original = to_bgr(tooth_img_normalized)
    mask_img = to_bgr(mask)
    skeleton_img = to_bgr(skeleton)
    pruned_img = to_bgr(pruned)
    overlay_img = overlay_centerline(tooth_img_normalized, smooth_curve, centerline_points)

    target_size = (320, 320)
    original = resize_image_for_display(original, target_size)
    mask_img = resize_image_for_display(mask_img, target_size)
    skeleton_img = resize_image_for_display(skeleton_img, target_size)
    pruned_img = resize_image_for_display(pruned_img, target_size)
    overlay_img = resize_image_for_display(overlay_img, target_size)

    original = annotate_image(original, "1. Tooth ROI")
    mask_img = annotate_image(mask_img, "2. Tooth Mask")
    skeleton_img = annotate_image(skeleton_img, "3. Skeleton")
    pruned_img = annotate_image(pruned_img, "4. Pruned Skeleton")
    overlay_img = annotate_image(overlay_img, "5. Smoothed Centerline")

    row1 = np.hstack([original, mask_img, skeleton_img])
    row2 = np.hstack([pruned_img, overlay_img, np.zeros_like(original)])
    panel = np.vstack([row1, row2])

    cv2.imshow("Pipeline Debug Panel", panel)
    cv2.waitKey(0)
    cv2.destroyWindow("Pipeline Debug Panel")


# ═════════════════════════════════════════════════════════════════════════════
#  BATCH PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def extract_features_batch(tooth_img_dir=None, output_csv=None, debug=False):
    """
    Processes all extracted tooth ROI images and extracts midline_length_px.
    
    Args:
        tooth_img_dir: directory containing tooth ROI images
                      default: data/working_length/processed/teeth
        output_csv: output CSV path
                   default: data/working_length/features/features.csv
        debug: show intermediate steps for each image
    
    Returns:
        results: list of dicts with features
    """
    if tooth_img_dir is None:
        tooth_img_dir = get_dataset_path("working_length", "processed", "teeth")
    if output_csv is None:
        output_csv = "data/working_length/features/features.csv"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Find all tooth images
    tooth_images = sorted([
        f for f in os.listdir(tooth_img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    if not tooth_images:
        print(f"No images found in {tooth_img_dir}")
        return []
    
    results = []
    
    for idx, filename in enumerate(tooth_images):
        img_path = os.path.join(tooth_img_dir, filename)
        
        print(f"[{idx+1}/{len(tooth_images)}] {filename}")
        
        # Load image (normalized 0-1)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ERROR: Could not load {img_path}")
            continue
        
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        
        # Extract features
        midline_length_px, diagnostics = extract_midline_length(
            img_normalized, debug=debug
        )
        
        results.append({
            'tooth_image_filename': filename,
            'midline_length_px': midline_length_px,
        })
        
        print(f"  ✓ midline_length_px: {midline_length_px:.2f}")
    
    # Save results to CSV
    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tooth_image_filename', 'midline_length_px'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nFeatures saved to: {output_csv}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract tooth features")
    parser.add_argument("--tooth_dir", type=str, default=None,
                        help="Directory with tooth ROI images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--debug", action="store_true",
                        help="Show intermediate steps")
    
    args = parser.parse_args()
    
    extract_features_batch(
        tooth_img_dir=args.tooth_dir,
        output_csv=args.output,
        debug=args.debug
    )
