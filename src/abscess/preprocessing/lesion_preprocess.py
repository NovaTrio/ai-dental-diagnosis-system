"""
lesion_preprocess.py
────────────────────
Minimal preprocessing for periapical radiograph tooth-ROI images.

Pipeline:
  1. Background removal – threshold out the black surround and store the
     foreground region in an alpha channel (BGRA output).

Input  : ../../../data/common/processed/common_selected_tooth_roi/<name>.jpg
Output : ../../../data/abscess/raw/BlackRemove/<name>_clahe_sigmoid.png

NOTE:
The "_clahe_sigmoid" suffix is kept only for downstream filename compatibility.
No CLAHE, Sigmoid, or N4 correction is applied.

Usage:
    python lesion_preprocess.py
"""

import os
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR = '../../../data/common/processed/common_selected_tooth_roi'
OUTPUT_DIR = '../../../data/abscess/raw/BlackRemove'

# Kept only for downstream filename compatibility
OUTPUT_SUFFIX = '_clahe_sigmoid'

# Pixels whose B, G, and R values are all <= this value are treated as background
BLACK_THRESHOLD = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Background removal
# ─────────────────────────────────────────────────────────────────────────────
def foreground_mask(
    img_bgr: np.ndarray,
    threshold: int = BLACK_THRESHOLD
) -> np.ndarray:
    """
    Create a foreground mask by removing the near-black image surround.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image.
    threshold : int
        Pixels whose B, G, and R values are all less than or equal to this
        threshold are classified as background.

    Returns
    -------
    np.ndarray
        uint8 binary mask where:
        255 = foreground
        0   = background
    """
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array(
        [threshold, threshold, threshold],
        dtype=np.uint8
    )

    black_mask = cv2.inRange(
        img_bgr,
        lower_black,
        upper_black
    )

    foreground = cv2.bitwise_not(black_mask)
    return foreground


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("PREPROCESSING: Background Removal")
    print("=" * 60)
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Black threshold : {BLACK_THRESHOLD}")
    print("=" * 60)

    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input directory does not exist:\n  {INPUT_DIR}")
        return

    files = sorted(
        filename
        for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    if not files:
        print(f"[ERROR] No images found in:\n  {INPUT_DIR}")
        return

    successful = 0

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"  [FAIL] Could not read: {filename}")
            continue

        # Stage 1: Remove the near-black external background
        mask = foreground_mask(
            img,
            BLACK_THRESHOLD
        )

        # Convert the original image to grayscale.
        # No N4, CLAHE, Sigmoid, or other enhancement is applied.
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Ensure removed background pixels remain black
        gray[mask == 0] = 0

        # Save as BGRA:
        # B, G, R = original grayscale image
        # Alpha   = foreground mask
        bgra = cv2.merge([
            gray,
            gray,
            gray,
            mask
        ])

        output_filename = f"{base_name}{OUTPUT_SUFFIX}.png"
        output_path = os.path.join(
            OUTPUT_DIR,
            output_filename
        )

        saved = cv2.imwrite(
            output_path,
            bgra
        )

        if not saved:
            print(f"  [FAIL] Could not save: {output_filename}")
            continue

        successful += 1
        foreground_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.size)

        print(
            f"  [OK] {output_filename} "
            f"| foreground: {foreground_pixels}/{total_pixels} px"
        )

    print("\n" + "=" * 60)
    print(
        f"PROCESSING COMPLETE! "
        f"({successful}/{len(files)} images)"
    )
    print("  Pipeline: Background Removal Only")
    print("=" * 60)


if __name__ == '__main__':
    main()