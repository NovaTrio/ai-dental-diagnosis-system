import argparse
import cv2
import os
import csv
import numpy as np
from src.common.preprocessing.base_preprocess import get_dataset_path, base_preprocess

#  PATHS

BOXES_CSV  = "data/working_length/labels/tooth_boxes.csv"
OUTPUT_DIR = "data/working_length/processed/teeth"

#  PHASE 2 — ROI EXTRACTION  (uses CSV built by wl_preprocess.py)
#  python roi_extraction.py
#
#  For every image in the raw folder:
#    1. Base preprocess  (no boxes shown here)
#    2. Remove black borders
#    3. Load boxes from CSV → show on preprocessed image
#    4. User clicks a tooth → ENTER to confirm
#    5. Enhancement pipeline on selected crop only
#    6. Auto-saved as  <base>_tooth_<N>.png
#    7. Next image loads automatically
#  ESC = skip image  |  q = quit entire run

def run_preprocessing(overwrite=False, show_comparison=True, debug=False):
    input_dir = get_dataset_path("working_length", "raw", "images")
    images    = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not images:
        print("No images found.")
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

        result = wl_roi_preprocess(input_path, debug=debug)

        if result is None:
            print(f"  Skipped.")
            continue

        tooth_crop, tooth_idx, raw_crop = result
        if show_comparison:
            show_crop_comparison(raw_crop, tooth_crop, filename, tooth_idx)
        out_name = f"{base}_tooth_{tooth_idx}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, (tooth_crop * 255).astype(np.uint8))
        print(f"  Saved: {out_path}")

    print("\nAll images processed.")

#  CORE PIPELINE  (single image)

def wl_roi_preprocess(input_path, debug=False):
    """
    Full pipeline for one image.
    Returns (processed_crop_float, tooth_index_1based)  or  None if skipped.
    """
    filename = os.path.basename(input_path)

    # ── Step 0: base preprocessing — boxes NOT shown here
    img = base_preprocess(input_path)
    if debug:
        cv2.imshow("0. After Base Preprocess", img)
        cv2.waitKey(0)

    # ── Step 1: remove black borders
    img = remove_black_borders(img)
    if debug:
        cv2.imshow("1. After Border Removal", img)
        cv2.waitKey(0)

    # ── Step 2: load boxes from CSV built by wl_preprocess.py
    boxes = load_boxes_for_image(filename)
    if not boxes:
        print(f"  No boxes found for {filename}. Run wl_preprocess.py first.")
        return None

    # ── Step 3: show boxes on preprocessed image → user selects tooth
    selection = select_tooth_interactively(img, boxes, filename)
    if selection is None:
        return None

    tooth_crop, tooth_idx = selection

    if debug:
        cv2.imshow("2. Selected Tooth (raw crop)", tooth_crop)
        cv2.waitKey(0)

    # ── Step 4: enhancement on selected crop only
    result = enhance(tooth_crop, debug=debug)

    return result, tooth_idx, tooth_crop

#  Interactive tooth selector

def select_tooth_interactively(img, boxes, filename):
    """
    Draws tooth boxes on the preprocessed image.
    Hover highlights; click to select; ENTER to confirm.

    Returns (cropped_tooth_float, tooth_index_1based)  or  None.
    q = quit entire run | ESC = skip this image.
    """
    h, w    = img.shape
    COLOURS = [(0,255,0),(0,128,255),(255,0,128),(255,255,0),(0,255,255)]
    WIN     = f"Select tooth  |  click + ENTER=confirm  ESC=skip  q=quit  [{filename}]"

    selected = [None]

    def make_vis(highlight=None):
        vis = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, (x1p,y1p,x2p,y2p) in enumerate(boxes):
            x1,y1 = int(x1p*w), int(y1p*h)
            x2,y2 = int(x2p*w), int(y2p*h)
            c     = COLOURS[i % len(COLOURS)]
            thick = 3 if highlight == i else 2
            cv2.rectangle(vis, (x1,y1), (x2,y2), c, thick)
            cv2.putText(vis, f"T{i+1}", (x1+6, y1+26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        cv2.putText(vis, "click tooth  ENTER=confirm  ESC=skip  q=quit",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return vis

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            for i, (x1p,y1p,x2p,y2p) in enumerate(boxes):
                if int(x1p*w)<=x<=int(x2p*w) and int(y1p*h)<=y<=int(y2p*h):
                    cv2.imshow(WIN, make_vis(highlight=i))
                    return
            cv2.imshow(WIN, make_vis())

        elif event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1p,y1p,x2p,y2p) in enumerate(boxes):
                if int(x1p*w)<=x<=int(x2p*w) and int(y1p*h)<=y<=int(y2p*h):
                    selected[0] = i
                    print(f"  Selected: Tooth {i+1}  (press ENTER to confirm)")
                    cv2.imshow(WIN, make_vis(highlight=i))
                    return

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_cb)
    cv2.imshow(WIN, make_vis())

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13 and selected[0] is not None:   # ENTER
            break
        elif key == 27:                              # ESC — skip image
            cv2.destroyWindow(WIN)
            return None
        elif key == ord('q'):                        # q   — quit run
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyWindow(WIN)

    i = selected[0]
    x1p,y1p,x2p,y2p = boxes[i]
    x1,y1 = int(x1p*w), int(y1p*h)
    x2,y2 = int(x2p*w), int(y2p*h)
    return img[y1:y2, x1:x2], i + 1

#  Enhancement pipeline  (runs on selected tooth crop only)

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

#  CSV helper

def load_boxes_for_image(filename):
    """Loads tooth boxes for a given filename from BOXES_CSV."""
    if not os.path.exists(BOXES_CSV):
        return []
    boxes = []
    with open(BOXES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["filename"] == filename:
                boxes.append((
                    int(row["tooth_index"]),
                    float(row["x1_pct"]), float(row["y1_pct"]),
                    float(row["x2_pct"]), float(row["y2_pct"])
                ))
    boxes.sort(key=lambda r: r[0])
    return [(r[1], r[2], r[3], r[4]) for r in boxes]

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
