import cv2
import os
import csv
import numpy as np
from src.common.preprocessing.base_preprocess import get_dataset_path, base_preprocess

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BOXES_CSV = "data/working_length/labels/tooth_boxes.csv"


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — AUTOMATIC PREPROCESSING
#  Run once:  python wl_preprocess.py
#
#  Loop every raw image, run `base_preprocess()` and save the
#  resulting preprocessed image to:
#    data/working_length/processed/preprocessed
#
PREPROCESSED_DIR = get_dataset_path("working_length", "processed", "preprocessed")


def build_dataset():
    input_dir = get_dataset_path("working_length", "raw", "images")
    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not images:
        print("No images found in", input_dir)
        return

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    total = len(images)
    for idx, filename in enumerate(images):
        print(f"[{idx+1}/{total}] Processing: {filename}")
        img_path = os.path.join(input_dir, filename)
        try:
            proc = base_preprocess(img_path)
        except Exception as e:
            print(f"  Failed preprocessing {filename}: {e}")
            continue

        out_path = os.path.join(PREPROCESSED_DIR, filename)
        # convert float image (0-1) to uint8 for saving
        out_uint8 = (np.clip(proc, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.imwrite(out_path, out_uint8)
        print(f"  Saved preprocessed image: {out_path}")

    print(f"\nPreprocessing complete. Images saved to: {PREPROCESSED_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive box drawing on raw image
# ─────────────────────────────────────────────────────────────────────────────
def _interactive_box_draw(img_path):
    """
    Shows the raw image. User clicks top-left then bottom-right
    of each tooth to draw a bounding box.

    Returns list of (x1_pct, y1_pct, x2_pct, y2_pct)  or  None if skipped.
    """
    raw = cv2.imread(img_path)
    if raw is None:
        print(f"  Could not read {img_path}")
        return []

    h, w       = raw.shape[:2]
    
    # Resize image to practical size for annotation (max 800x1200)
    MAX_DISPLAY_WIDTH = 400
    MAX_DISPLAY_HEIGHT = 600
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
    display_w = int(w * scale)
    display_h = int(h * scale)
    display_img = cv2.resize(raw, (display_w, display_h), interpolation=cv2.INTER_AREA)
    
    boxes, pts = [], []
    COLOURS    = [(0,255,0),(0,128,255),(255,0,128),(255,255,0),(0,255,255)]
    WIN        = "Phase 1 — Draw tooth boxes  |  r=undo  q=save & next  ESC=skip"

    def redraw():
        tmp = display_img.copy()
        for i, (bx1p,by1p,bx2p,by2p) in enumerate(boxes):
            bx1,by1 = int(bx1p*w*scale), int(by1p*h*scale)
            bx2,by2 = int(bx2p*w*scale), int(by2p*h*scale)
            c = COLOURS[i % len(COLOURS)]
            cv2.rectangle(tmp, (bx1,by1), (bx2,by2), c, 2)
            cv2.putText(tmp, f"T{i+1}", (bx1+6, by1+26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        cv2.putText(tmp, "r=undo  q=save & next  ESC=skip",
                    (10, display_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow(WIN, tmp)

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Convert display coordinates back to original image coordinates
        orig_x = x / scale
        orig_y = y / scale
        pts.append((orig_x, orig_y))
        redraw()
        if len(pts) % 2 == 0:
            x1, y1_ = pts[-2]
            x2, y2_ = pts[-1]
            boxes.append((x1/w, y1_/h, x2/w, y2_/h))
            print(f"    Tooth {len(boxes)}: "
                  f"({x1/w:.3f}, {y1_/h:.3f}, {x2/w:.3f}, {y2_/h:.3f})")
            redraw()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_cb)
    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r') and boxes:
            boxes.pop()
            if len(pts) >= 2:
                pts.pop(); pts.pop()
            redraw()
            print("    Undid last box.")
        elif key == ord('q'):
            cv2.destroyWindow(WIN)
            return boxes
        elif key == 27:             # ESC
            cv2.destroyWindow(WIN)
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  CSV helper
# ─────────────────────────────────────────────────────────────────────────────
def _load_done_filenames():
    """Returns set of filenames already recorded in the CSV."""
    done = set()
    if not os.path.exists(BOXES_CSV):
        return done
    with open(BOXES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["filename"])
    return done


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_dataset()
