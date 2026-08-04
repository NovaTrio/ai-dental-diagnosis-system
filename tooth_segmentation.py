# pip install opencv-python numpy scikit-image
"""
tooth_segmentation.py
---------------------
Segments the central target tooth/root from crown-cropped periapical
dental X-ray images using Morphological Chan-Vese active contour
segmentation.

This script uses ONLY classical image-processing techniques:
  - CLAHE contrast enhancement
  - Bilateral filtering
  - Morphological Chan-Vese (region-based active contour)
  - Morphological opening / closing / hole filling
  - Connected-component analysis with multi-criteria scoring

No deep learning, pretrained models, or brightness-only thresholding.
Chan-Vese detects regions by intensity consistency (inner region vs.
outer region average) rather than by a single brightness threshold,
so it works even when the background is brighter than the tooth.

Pipeline:
  1. Load image in grayscale.
  2. Remove black borders (rows/columns near zero intensity).
  3. Apply CLAHE and bilateral filtering.
  4. Normalise to float [0, 1].
  5. Create an automatic elliptical initial mask centred on the image.
  6. Run Morphological Chan-Vese active contour segmentation.
  7. Post-process: closing, opening, hole filling, remove small objects.
  8. Select the best connected component via composite score
     (centrality, area, vertical length, elongation, solidity).
  9. Save binary mask and overlay.

Input  : data/abscess/raw/croun_crops/
Output : data/abscess/raw/tooth_segmented/
"""

# =========================================================================
# Imports
# =========================================================================
import os
import sys
import cv2
import numpy as np

from skimage.segmentation import morphological_chan_vese
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
)

# =========================================================================
# Configurable Parameters  (edit these before running)
# =========================================================================

# --- Paths ----------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

INPUT_DIR  = os.path.join(_PROJECT_ROOT, "data", "abscess", "raw", "BlackRemove")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "abscess", "raw", "tooth_segmented")

# --- CLAHE ----------------------------------------------------------------
CLAHE_CLIP_LIMIT    = 3.0       # Contrast-limited clip value
CLAHE_TILE_GRID     = (8, 8)    # Tile grid size for local histogram eq.

# --- Bilateral filter -----------------------------------------------------
BILATERAL_D         = 9         # Diameter of pixel neighbourhood
BILATERAL_SIGMA_CLR = 75        # Filter sigma in the colour space
BILATERAL_SIGMA_SPC = 75        # Filter sigma in the coordinate space

# --- Black border removal -------------------------------------------------
BORDER_THRESH       = 10        # Pixel values <= this are "black"
BORDER_MIN_FRAC     = 0.85      # A row/col is "black" if >= this fraction
                                # of its pixels are <= BORDER_THRESH

# --- Chan-Vese active contour ---------------------------------------------
CV_NUM_ITER         = 200       # Number of active contour iterations
CV_SMOOTHING        = 3         # Smoothing passes per iteration (1-4)
CV_LAMBDA1          = 1.0       # Weight for the outer region
CV_LAMBDA2          = 1.0       # Weight for the inner region

# --- Elliptical initial mask ----------------------------------------------
INIT_ELLIPSE_W_FRAC = 0.50     # Ellipse semi-axis width  as fraction of W
INIT_ELLIPSE_H_FRAC = 0.45     # Ellipse semi-axis height as fraction of H

# --- Morphological post-processing ----------------------------------------
CLOSING_KERNEL_SIZE = 11        # Diameter of the closing kernel
OPENING_KERNEL_SIZE = 5         # Diameter of the opening kernel
HOLE_AREA_THRESHOLD = 500       # Fill holes smaller than this (px)
MIN_COMPONENT_AREA  = 200       # Remove components smaller than this (px)

# --- Component scoring weights --------------------------------------------
W_CENTRALITY   = 0.25           # Closeness to horizontal centre
W_AREA         = 0.20           # Large connected area
W_HEIGHT       = 0.20           # Vertical length
W_ELONGATION   = 0.20           # Aspect ratio (tall & narrow = tooth)
W_SOLIDITY     = 0.15           # Convex-hull solidity (compact shape)

# --- Supported image extensions -------------------------------------------
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")


# =========================================================================
#  Step 1: Load image in grayscale
# =========================================================================

def load_grayscale(image_path):
    """
    Read an image with cv2.IMREAD_UNCHANGED and convert to single-
    channel uint8 grayscale.  Handles grayscale, BGR, and BGRA inputs.
    """
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if raw.ndim == 2:
        return raw.copy()
    elif raw.ndim == 3 and raw.shape[2] == 4:
        return cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {raw.shape}")


# =========================================================================
#  Step 2: Remove black borders
# =========================================================================

def remove_black_borders(gray, thresh=BORDER_THRESH,
                         min_frac=BORDER_MIN_FRAC):
    """
    Crop rows and columns that are predominantly black.  A row (or
    column) is considered "black" if at least `min_frac` of its pixels
    have intensity <= `thresh`.
    """
    h, w = gray.shape

    # --- Detect black rows ------------------------------------------------
    row_black = np.mean(gray <= thresh, axis=1) >= min_frac
    top = 0
    while top < h and row_black[top]:
        top += 1
    bottom = h - 1
    while bottom > top and row_black[bottom]:
        bottom -= 1

    # --- Detect black columns ---------------------------------------------
    col_black = np.mean(gray <= thresh, axis=0) >= min_frac
    left = 0
    while left < w and col_black[left]:
        left += 1
    right = w - 1
    while right > left and col_black[right]:
        right -= 1

    # Ensure we keep at least some image
    if top >= bottom or left >= right:
        return gray.copy()

    return gray[top:bottom + 1, left:right + 1].copy()


# =========================================================================
#  Step 3: Apply CLAHE and bilateral filtering
# =========================================================================

def enhance_contrast(gray):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    followed by bilateral filtering to enhance local contrast while
    reducing noise and preserving edges.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    enhanced = clahe.apply(gray)

    filtered = cv2.bilateralFilter(
        enhanced,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_CLR,
        sigmaSpace=BILATERAL_SIGMA_SPC,
    )
    return filtered


# =========================================================================
#  Step 4: Normalise to float [0, 1]
# =========================================================================

def normalise_to_float(gray_uint8):
    """Convert uint8 image to float64 in [0, 1]."""
    return gray_uint8.astype(np.float64) / 255.0


# =========================================================================
#  Step 5: Create elliptical initial mask
# =========================================================================

def create_initial_mask(h, w,
                        w_frac=INIT_ELLIPSE_W_FRAC,
                        h_frac=INIT_ELLIPSE_H_FRAC):
    """
    Create a binary elliptical mask centred on the image.  The ellipse
    is sized as a fraction of the image dimensions.  This gives Chan-
    Vese a starting region near the expected tooth location.

    Returns
    -------
    mask : np.ndarray (H, W) uint8   {0, 1}
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    centre = (w // 2, h // 2)
    axes   = (int(w * w_frac / 2), int(h * h_frac / 2))
    cv2.ellipse(mask, centre, axes, angle=0,
                startAngle=0, endAngle=360, color=1, thickness=-1)
    return mask


# =========================================================================
#  Step 6: Run Morphological Chan-Vese segmentation
# =========================================================================

def run_chan_vese(gray_float, init_mask,
                 num_iter=CV_NUM_ITER,
                 smoothing=CV_SMOOTHING,
                 lambda1=CV_LAMBDA1,
                 lambda2=CV_LAMBDA2):
    """
    Run the Morphological Chan-Vese (MorphACWE) active contour on the
    normalised grayscale image, starting from `init_mask`.

    Chan-Vese segments by partitioning the image into two regions whose
    internal intensity variance is minimised.  This works even when the
    tooth and background have overlapping brightness ranges.

    Returns
    -------
    segmentation : np.ndarray (H, W) bool
    """
    segmentation = morphological_chan_vese(
        gray_float,
        num_iter=num_iter,
        init_level_set=init_mask,
        smoothing=smoothing,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    return segmentation.astype(bool)


# =========================================================================
#  Step 7: Post-process the raw segmentation mask
# =========================================================================

def postprocess_mask(raw_mask, h, w):
    """
    Clean the raw Chan-Vese output with morphological operations:
      1. Morphological closing to bridge small gaps.
      2. Morphological opening to remove thin protrusions.
      3. Fill small holes.
      4. Remove small connected components.

    Returns
    -------
    cleaned : np.ndarray (H, W) uint8   {0, 255}
    """
    mask_uint8 = raw_mask.astype(np.uint8) * 255

    # --- Morphological closing (bridge gaps) ------------------------------
    kern_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE)
    )
    mask_uint8 = cv2.morphologyEx(
        mask_uint8, cv2.MORPH_CLOSE, kern_close, iterations=2
    )

    # --- Morphological opening (smooth edges, remove thin protrusions) ----
    kern_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE)
    )
    mask_uint8 = cv2.morphologyEx(
        mask_uint8, cv2.MORPH_OPEN, kern_open, iterations=1
    )

    # --- Fill holes -------------------------------------------------------
    bool_mask = mask_uint8 > 0
    try:
        bool_mask = remove_small_holes(
            bool_mask, area_threshold=HOLE_AREA_THRESHOLD
        )
    except TypeError:
        # scikit-image >= 0.26 renamed parameter
        bool_mask = remove_small_holes(
            bool_mask, max_size=HOLE_AREA_THRESHOLD
        )

    # --- Remove small objects ---------------------------------------------
    try:
        bool_mask = remove_small_objects(
            bool_mask, min_size=MIN_COMPONENT_AREA
        )
    except TypeError:
        bool_mask = remove_small_objects(
            bool_mask, max_size=MIN_COMPONENT_AREA
        )

    # If cleaning removed everything, fall back to raw mask
    if not bool_mask.any():
        bool_mask = raw_mask.copy()

    return (bool_mask.astype(np.uint8) * 255)


# =========================================================================
#  Step 8: Select the best connected component
# =========================================================================

def select_best_component(mask_uint8, h, w):
    """
    Among all connected components in the cleaned mask, select the one
    most likely to be the target tooth/root using a composite score:

      score = W_CENTRALITY  * centrality
            + W_AREA        * normalised_area
            + W_HEIGHT      * normalised_height
            + W_ELONGATION  * elongation_score
            + W_SOLIDITY    * solidity

    Criteria explained:
      - centrality   : horizontal closeness to image centre
      - area         : fraction of image area (favours large regions)
      - height       : fraction of image height (favours tall regions)
      - elongation   : height / width ratio clipped to [0, 1]
                       (favours elongated tooth/root shape)
      - solidity     : contour area / convex hull area
                       (favours compact, non-fragmented shapes)

    Returns
    -------
    final_mask : np.ndarray (H, W) uint8   {0, 255}
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    if num_labels <= 1:
        raise ValueError("No connected components remain after post-processing.")

    img_cx = w / 2.0
    img_area = float(h * w)

    best_label = None
    best_score = -float("inf")

    for lbl in range(1, num_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_cx = centroids[lbl][0]

        if area < MIN_COMPONENT_AREA:
            continue

        # --- Centrality (1.0 = perfectly centred) -------------------------
        centrality = 1.0 - abs(comp_cx - img_cx) / max(img_cx, 1.0)
        centrality = max(centrality, 0.0)

        # --- Normalised area ----------------------------------------------
        norm_area = area / img_area

        # --- Normalised vertical height -----------------------------------
        norm_height = comp_h / float(h)

        # --- Elongation (tall & narrow → tooth-like) ----------------------
        # Ratio of height to width, capped at 1.0 via sigmoid-like mapping
        aspect = comp_h / max(comp_w, 1)
        elongation = min(aspect / 3.0, 1.0)   # peaks at aspect >= 3

        # --- Solidity (contour area / convex hull area) -------------------
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        solidity = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = cv2.contourArea(cnt) / hull_area

        # --- Composite score ----------------------------------------------
        score = (
            W_CENTRALITY  * centrality
            + W_AREA      * norm_area
            + W_HEIGHT    * norm_height
            + W_ELONGATION * elongation
            + W_SOLIDITY  * solidity
        )

        if score > best_score:
            best_score = score
            best_label = lbl

    if best_label is None:
        raise ValueError("No component passed the minimum-area filter.")

    final = np.zeros_like(mask_uint8)
    final[labels == best_label] = 255

    # One final closing to smooth the boundary
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE)
    )
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kern, iterations=1)

    return final


# =========================================================================
#  Step 9: Ensure correct polarity (tooth = white)
# =========================================================================

def ensure_tooth_is_white(mask_uint8, gray_float):
    """
    Chan-Vese may assign the tooth to label 0 or label 1 depending on
    the initial conditions.  We fix the polarity by checking:

    1. Border contact: the tooth (foreground) should touch fewer image-
       border pixels than the background.  If the segmented region
       touches more borders, it is likely the background → invert.
    2. Area ratio: the tooth should cover < 65% of the image area.
       If the mask covers more, it is likely the background → invert.

    This avoids using brightness assumptions, since the user specified
    that the background may be brighter than the tooth.
    """
    h, w = gray_float.shape

    seg_bool = mask_uint8 > 0
    inv_bool = ~seg_bool

    # Build a thin border mask (3 px wide)
    border = np.zeros((h, w), dtype=bool)
    border[:3, :] = True
    border[-3:, :] = True
    border[:, :3] = True
    border[:, -3:] = True

    seg_border = int(np.sum(seg_bool & border))
    inv_border = int(np.sum(inv_bool & border))

    # Area fraction
    area_ratio = seg_bool.sum() / max(h * w, 1)

    # Invert if the segmented region is the background
    if seg_border > inv_border or area_ratio > 0.65:
        return 255 - mask_uint8

    return mask_uint8


# =========================================================================
#  Step 10: Save outputs (mask and overlay)
# =========================================================================

def save_outputs(gray_uint8, tooth_mask, filename, output_dir):
    """
    Save two output images:

    1. tooth_mask_<filename>   — Binary mask (white = tooth, black = bg)
    2. tooth_overlay_<filename> — Green highlight + red boundary contour
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Binary tooth mask
    # ------------------------------------------------------------------
    mask_path = os.path.join(output_dir, f"tooth_mask_{filename}")
    cv2.imwrite(mask_path, tooth_mask)

    # ------------------------------------------------------------------
    # 2. Overlay: green highlight + red boundary
    # ------------------------------------------------------------------
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    highlight = base.copy()
    highlight[tooth_mask == 255] = [0, 200, 0]     # green fill

    # Blend the green highlight with the original
    overlay = cv2.addWeighted(base, 0.6, highlight, 0.4, 0)

    # Draw the tooth boundary contour in red
    contours, _ = cv2.findContours(
        tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    overlay_path = os.path.join(output_dir, f"tooth_overlay_{filename}")
    cv2.imwrite(overlay_path, overlay)

    return mask_path, overlay_path


# =========================================================================
#  Full pipeline for a single image
# =========================================================================

def segment_tooth(image_path, filename, output_dir=OUTPUT_DIR):
    """
    Run the complete tooth segmentation pipeline on one image.
    """
    print(f"\n{'=' * 60}")
    print(f"  Processing: {filename}")
    print(f"{'=' * 60}")

    # --- Step 1: Load grayscale -------------------------------------------
    gray_raw = load_grayscale(image_path)
    print(f"  Raw size             : {gray_raw.shape[1]} x {gray_raw.shape[0]} px")

    # --- Step 2: Remove black borders -------------------------------------
    gray_cropped = remove_black_borders(gray_raw)
    h, w = gray_cropped.shape
    print(f"  After border removal : {w} x {h} px")

    # --- Step 3: CLAHE + bilateral filtering ------------------------------
    gray_enhanced = enhance_contrast(gray_cropped)

    # --- Step 4: Normalise to float [0, 1] --------------------------------
    gray_float = normalise_to_float(gray_enhanced)

    # --- Step 5: Elliptical initial mask ----------------------------------
    init_mask = create_initial_mask(h, w)
    print(f"  Init ellipse area    : {int(init_mask.sum())} px")

    # --- Step 6: Chan-Vese segmentation -----------------------------------
    raw_seg = run_chan_vese(gray_float, init_mask)
    raw_area = int(raw_seg.sum())
    print(f"  Raw CV area          : {raw_area} px")

    # --- Ensure tooth is white (correct polarity) -------------------------
    raw_uint8 = raw_seg.astype(np.uint8) * 255
    raw_uint8 = ensure_tooth_is_white(raw_uint8, gray_float)
    raw_seg_corrected = raw_uint8 > 0

    # --- Step 7: Post-process ---------------------------------------------
    cleaned = postprocess_mask(raw_seg_corrected, h, w)

    # --- Step 8: Select the best component --------------------------------
    tooth_mask = select_best_component(cleaned, h, w)
    tooth_area = int((tooth_mask > 0).sum())
    print(f"  Final tooth area     : {tooth_area} px")

    # --- Step 9: Save outputs ---------------------------------------------
    mask_path, overlay_path = save_outputs(
        gray_cropped, tooth_mask, filename, output_dir
    )
    print(f"  Saved tooth mask     : {mask_path}")
    print(f"  Saved overlay        : {overlay_path}")

    return tooth_mask


# =========================================================================
#  Batch runner
# =========================================================================

def run_batch(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """Process all supported images in the input directory."""

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        raise SystemExit(1)

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_EXTS)
    ])

    if not files:
        print(f"[ERROR] No images found in: {input_dir}")
        raise SystemExit(1)

    print("=" * 60)
    print("  Tooth Segmentation via Morphological Chan-Vese")
    print("=" * 60)
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  CV iterations    : {CV_NUM_ITER}")
    print(f"  CV smoothing     : {CV_SMOOTHING}")
    print(f"  Ellipse W frac   : {INIT_ELLIPSE_W_FRAC}")
    print(f"  Ellipse H frac   : {INIT_ELLIPSE_H_FRAC}")
    print(f"  Min comp. area   : {MIN_COMPONENT_AREA} px")
    print(f"  Images found     : {len(files)}")
    print("=" * 60)

    success = 0
    failed  = 0

    for filename in files:
        image_path = os.path.join(input_dir, filename)
        try:
            segment_tooth(image_path, filename, output_dir)
            success += 1
        except Exception as exc:
            print(f"\n  [FAIL] {filename}: {exc}\n")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Batch complete: {success} succeeded, {failed} failed")
    print(f"{'=' * 60}")


# =========================================================================
#  Entry point
# =========================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single-image mode
        single_path = sys.argv[1]
        fname = os.path.basename(single_path)
        segment_tooth(single_path, fname)
    else:
        # Batch mode
        run_batch()
