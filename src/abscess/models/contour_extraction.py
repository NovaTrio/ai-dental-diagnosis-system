"""
contour_extraction.py
─────────────────────
Extracts contours from post-processed periapical lesion segmentation masks.
Finds the largest contour (assumed to be the lesion), computes area and
perimeter, draws the boundary on the mask, and exports a CSV summary.

Input  : Binary masks from data/abscess/raw/lesion_postprocessed/
Output : Contour images  → data/abscess/raw/contour_output/
         CSV summary     → data/abscess/raw/contour_output/contour_metrics.csv
"""

import os
import csv
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR  = '../../../data/abscess/raw/lesion_postprocessed'
OUTPUT_DIR = '../../../data/abscess/raw/contour_output'
CSV_PATH   = os.path.join(OUTPUT_DIR, 'contour_metrics.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Contour drawing colour (BGR: Green)
CONTOUR_COLOR     = (0, 255, 0)
CONTOUR_THICKNESS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Core Function
# ─────────────────────────────────────────────────────────────────────────────
def extract_contour(mask_path: str, filename: str) -> dict | None:
    """
    Load a binary lesion mask, extract the largest contour, compute
    area & perimeter, draw the contour, and save the result.

    Returns a dict with image_id, area, and perimeter — or None on failure.
    """
    print(f"\nProcessing: {filename}")

    # ── 1. Load mask in grayscale ─────────────────────────────────────────
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [WARN] Could not read {mask_path} — skipping.")
        return None

    # Ensure strictly binary (0 or 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ── 2. Find external contours ─────────────────────────────────────────
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"  [WARN] No contours found in {filename} — skipping.")
        return None

    # ── 3. Select the largest contour ─────────────────────────────────────
    largest_contour = max(contours, key=cv2.contourArea)

    # ── 4. Calculate metrics ──────────────────────────────────────────────
    contour_area      = cv2.contourArea(largest_contour)
    contour_perimeter = cv2.arcLength(largest_contour, closed=True)

    print(f"  Contour Area      : {contour_area:.2f} px²")
    print(f"  Contour Perimeter : {contour_perimeter:.2f} px")

    # ── 5. Draw contour on the mask ───────────────────────────────────────
    # Convert grayscale mask to BGR so the green contour is visible
    contour_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(
        contour_vis, [largest_contour], -1,
        CONTOUR_COLOR, CONTOUR_THICKNESS
    )

    # ── 6. Save the contour visualisation ─────────────────────────────────
    out_name = f"contour_{filename}"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, contour_vis)
    print(f"  -> Contour image saved: {out_path}")

    # ── 7. Return metrics for CSV ─────────────────────────────────────────
    image_id = os.path.splitext(filename)[0]
    return {
        "Image ID":          image_id,
        "Contour Area":      round(contour_area, 2),
        "Contour Perimeter": round(contour_perimeter, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  Contour Extraction from Post-Processed Lesion Masks")
    print("=" * 65)
    print(f"  Input  directory : {INPUT_DIR}")
    print(f"  Output directory : {OUTPUT_DIR}")
    print("=" * 65)

    supported_exts = ('.png', '.jpg', '.jpeg')

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().startswith('lesion_mask_') and f.lower().endswith(supported_exts)
    ]

    if not files:
        print(f"\n[ERROR] No mask images found in:\n  {INPUT_DIR}")
        print("        Run lesion_postprocess.py first.")
    else:
        print(f"\nFound {len(files)} mask(s) to process.\n")

        results = []
        for filename in sorted(files):
            mask_path = os.path.join(INPUT_DIR, filename)
            metrics = extract_contour(mask_path, filename)
            if metrics is not None:
                results.append(metrics)

        # ── Write CSV ─────────────────────────────────────────────────────
        if results:
            fieldnames = ["Image ID", "Contour Area", "Contour Perimeter"]
            with open(CSV_PATH, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  -> CSV saved: {CSV_PATH}")
        else:
            print("\n  [WARN] No valid contours extracted — CSV not created.")

    print("\n" + "=" * 65)
    print("  Contour Extraction Complete!")
    print("=" * 65)
