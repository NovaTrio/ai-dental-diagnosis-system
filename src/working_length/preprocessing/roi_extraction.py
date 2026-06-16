import argparse
import cv2
import os
import numpy as np
from src.common.preprocessing.base_preprocess import get_dataset_path

#  PATHS

OUTPUT_DIR = "data/working_length/processed/teeth"
INPUT_DIR = get_dataset_path("common", "processed", "common_selected_tooth_roi")

#  PHASE 2 — ROI EXTRACTION  (process selected-tooth ROI images)
#  python roi_extraction.py
#
#  For every selected-tooth ROI image in:
#    data/common/processed/common_selected_tooth_roi
#    1. Remove black borders
#    2. Apply enhancement pipeline
#    3. Save result to data/working_length/processed/teeth
#  ESC = skip image  |  q = quit entire run

def run_preprocessing(overwrite=False, show_comparison=True, debug=False):
    input_dir = INPUT_DIR
    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    if not images:
        print("No images found in", input_dir)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = len(images)

    for idx, filename in enumerate(images):
        base = os.path.splitext(filename)[0]

        existing = [f for f in os.listdir(OUTPUT_DIR)
                    if f.startswith(base + "_tooth_")]
        if existing and not overwrite:
            print(f"[{idx+1}/{total}] Skipping (already processed): {filename}")
            continue

        input_path = os.path.join(input_dir, filename)
        print(f"\n[{idx+1}/{total}] {filename}")

        result = wl_roi_preprocess(input_path, filename, debug=debug)

        if result is None:
            print(f"  Skipped.")
            continue

        tooth_crop, raw_crop = result
        if show_comparison:
            show_crop_comparison(raw_crop, tooth_crop, filename)
        out_name = f"{base}_tooth.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, (tooth_crop * 255).astype(np.uint8))
        print(f"  Saved: {out_path}")

    print("\nAll images processed.")

#  CORE PIPELINE  (single image)

def wl_roi_preprocess(input_path, filename, debug=False):
    """
    Full pipeline for one selected tooth ROI image.
    Returns (processed_crop_float, raw_crop) or None if skipped.
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Could not read ROI image: {input_path}")
        return None

    img = img.astype("float32") / 255.0

    if debug:
        cv2.imshow("0. Input ROI", img)
        cv2.waitKey(0)

    # ── Step 1: remove black borders
    img = remove_black_borders(img)
    if debug:
        cv2.imshow("1. After Border Removal", img)
        cv2.waitKey(0)

    # ── Step 2: enhancement on ROI only
    result = enhance(img, debug=debug)

    return result, img

#  Interactive tooth selector

#  Enhancement pipeline  (runs on ROI only)

def enhance(img, debug=False):
    img = sigmoid_contrast(img)
    if debug:
        cv2.imshow("3. After Sigmoid Contrast", img)
        cv2.waitKey(0)

    img = sharpen_image(img)
    if debug:
        cv2.imshow("4. After Sharpening", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def sigmoid_contrast(img, gain=8, cutoff=0.5):
    return 1 / (1 + np.exp(-gain * (img - cutoff)))


def sharpen_image(img):
    img_uint8 = (img * 255).astype(np.uint8)
    blurred   = cv2.GaussianBlur(img_uint8, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_uint8, 2.0, blurred, -1.0, 0)
    return sharpened.astype(np.float32) / 255.0


def show_crop_comparison(raw_crop, processed_crop, filename, tooth_idx):
    raw_vis = (raw_crop * 255).astype(np.uint8)
    proc_vis = (processed_crop * 255).astype(np.uint8)
    if raw_vis.ndim == 2:
        raw_vis = cv2.cvtColor(raw_vis, cv2.COLOR_GRAY2BGR)
    if proc_vis.ndim == 2:
        proc_vis = cv2.cvtColor(proc_vis, cv2.COLOR_GRAY2BGR)

    diff = cv2.absdiff(raw_vis, proc_vis)
    if diff.ndim == 3 and diff.shape[2] == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    input_title = f"Input tooth crop [{filename} #{tooth_idx}]"
    result_title = f"Result tooth crop [{filename} #{tooth_idx}]"
    diff_title = f"Difference [{filename} #{tooth_idx}]"

    cv2.imshow(input_title, raw_vis)
    cv2.imshow(result_title, proc_vis)
    cv2.imshow(diff_title, diff_color)
    print("  Viewing comparison. Press any key to continue, q to quit.")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(input_title)
    cv2.destroyWindow(result_title)
    cv2.destroyWindow(diff_title)
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)


def remove_black_borders(img, threshold=0.1, padding=5):
    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)
    rows = np.where(row_means > threshold)[0]
    cols = np.where(col_means > threshold)[0]
    if rows.size == 0 or cols.size == 0:
        return img
    h, w  = img.shape
    y_min = max(0,     rows[0]  - padding)
    y_max = min(h - 1, rows[-1] + padding)
    x_min = max(0,     cols[0]  - padding)
    x_max = min(w - 1, cols[-1] + padding)
    return img[y_min:y_max+1, x_min:x_max+1]

def isolate_tooth(img, debug=False):
    """
    Removes background and isolates the tooth structure.
    Input/Output: normalized float (0-1)

    Strategy:
      1. Threshold to get bright regions (tooth + instrument)
      2. Morphological ops to fill gaps and connect tooth parts
      3. Find the main tooth contour
      4. Apply mask — background becomes black
    """
    img_uint8 = (img * 255).astype(np.uint8)

    # ── Step 1: CLAHE to boost local contrast before thresholding
    #    This helps separate tooth from bone (similar intensities)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(img_uint8)

    # ── Step 2: Adaptive threshold — finds tooth edges locally
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=35,   # tune: larger = smoother regions
        C=-5            # tune: more negative = stricter threshold
    )

    if debug:
        cv2.imshow("Thresh Raw", thresh)
        cv2.waitKey(0)

    # ── Step 3: Morphological cleanup
    #    CLOSE fills holes inside the tooth
    #    OPEN  removes small noise outside
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)

    if debug:
        cv2.imshow("Thresh Cleaned", thresh)
        cv2.waitKey(0)

    # ── Step 4: Find contours and pick the best one (tooth)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("  Warning: no contours found, returning original")
        return img

    h, w = img.shape

    # filter: ignore contours smaller than 5% of image
    min_area = h * w * 0.05
    valid    = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid:
        print("  Warning: no valid contours, returning original")
        return img

    # pick the largest valid contour = main tooth body
    best = max(valid, key=cv2.contourArea)

    # ── Step 5: Build filled mask from contour
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)

    # optional: slightly dilate mask to avoid clipping tooth edges
    mask = cv2.dilate(mask, kernel, iterations=1)

    if debug:
        cv2.imshow("Tooth Mask", mask)
        cv2.waitKey(0)

    # ── Step 6: Apply mask — background → 0 (black)
    isolated = img.copy()
    isolated[mask == 0] = 0.0

    if debug:
        cv2.imshow("Isolated Tooth", isolated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return isolated

#  ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and enhance teeth ROIs from working length X-rays"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess existing files and overwrite previous outputs"
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Do not show input/result/difference comparison windows"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug windows for each pipeline stage"
    )
    args = parser.parse_args()

    run_preprocessing(
        overwrite=args.overwrite,
        show_comparison=not args.no_view,
        debug=args.debug,
    )
