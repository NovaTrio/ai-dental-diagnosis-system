"""
run_pipeline.py
───────────────
End-to-end periapical-lesion pipeline for a SINGLE X-ray, wiring together the
existing stage scripts in order:

    X-ray
      │  (manual)  tooth selection  ── click 2 points  → tooth_clicks/<name>.json
      │  (manual)  scale marker      ── click 2 points  → scale_calibration.csv
      ▼
    1. selected_tooth_roi      → common_selected_tooth_roi (+ debug)
    2. lesion_preprocess       → BlackRemove   (background removal)
    3. crop_croun              → croun_crops   (root/periapical crop)
    4. FCM lesion segmentation → lesion_segmented/lesion_mask_<name>.png
    5. restore_lesion_masks    → lesion_postprocessed_restored
    6. lesion_measurement_pca  → lesion_measurements   (viz + CSV)   ← OUTPUT
    7. lesion_orientation      → lesion_orientation     (viz + CSV)  ← OUTPUT

Each stage reuses the original script's own functions (no re-implementation
except the small FCM step, whose source module is currently disabled). Every
stage is driven on a single image via per-image working dirs, so nothing else in
the dataset is touched.

Live interactive run:
    1. Pick an X-ray (a file dialog opens if --image is omitted).
    2. Click the target tooth (centre + vertical direction), press 's'.
    3. Click the two ends of the 10 mm scale marker, press 'c'.
    4. The rest runs automatically; the two final visualizations pop up and stay
       open until you press a key. Images + CSVs are still saved to their folders.

Pass --reuse to skip the two click windows when the annotation/scale already
exists (e.g. the L* dataset images).

Usage:
    python run_pipeline.py                       # file dialog to choose the image
    python run_pipeline.py --image path/to/xray.jpg
    python run_pipeline.py --image L1.jpg --reuse
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import sys

import cv2
import numpy as np
import skfuzzy as fuzz
import joblib

try:                                    # some stage modules print non-ASCII at import
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/abscess
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
_D = os.path.join(PROJECT_ROOT, "data")

# ── Canonical absolute directories ───────────────────────────────────────────
IMAGES_DIR   = os.path.join(_D, "abscess", "raw", "images")
TOOTH_CLICKS = os.path.join(_D, "common", "annotations", "tooth_clicks")
SCALE_CSV    = os.path.join(_D, "abscess", "processed", "scale_calibration.csv")
ROI_DIR      = os.path.join(_D, "common", "processed", "common_selected_tooth_roi")
ROI_DEBUG    = os.path.join(_D, "common", "processed", "debug_common_selected_tooth_roi")
BLACKREMOVE  = os.path.join(_D, "abscess", "raw", "BlackRemove")
CROUN_DIR    = os.path.join(_D, "abscess", "raw", "croun_crops")
SEG_DIR      = os.path.join(_D, "abscess", "raw", "lesion_segmented")
RESTORED_DIR = os.path.join(_D, "abscess", "raw", "lesion_postprocessed_restored")
MEAS_DIR     = os.path.join(_D, "abscess", "raw", "lesion_measurements")
ORIENT_DIR   = os.path.join(_D, "abscess", "raw", "lesion_orientation")
WORK_ROOT    = os.path.join(_D, "abscess", "raw", "_pipeline_work")

SUFFIX = "_clahe_sigmoid"     # downstream filename convention (kept for compatibility)

# FCM segmentation params (reproduce the disabled lesion_segmentation.py)
FCM_CLUSTERS, FCM_M, FCM_ERR, FCM_ITER = 3, 2.0, 0.005, 1000
FCM_SPATIAL_WEIGHT = 0.30


# ═════════════════════════════════════════════════════════════════════════════
# Load stage modules, each with cwd = its own folder so its relative paths work
# ═════════════════════════════════════════════════════════════════════════════
def _load(rel_path: str):
    path = os.path.join(PROJECT_ROOT, rel_path)
    name = "pl_" + os.path.splitext(os.path.basename(path))[0]
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(path))
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


roi_mod  = _load("src/common/preprocessing/selected_tooth_roi.py")
crop_mod = _load("src/abscess/preprocessing/crop_croun.py")
toothseg_mod = _load("src/abscess/Segmentation/tooth_segmentation.py")  # RF tooth segmenter
rest_mod = _load("src/abscess/preprocessing/restore_lesion_masks.py")
meas_mod = _load("src/abscess/inference/lesion_measurement_pca.py")
orient_mod = _load("src/abscess/inference/lesion_orientation_analysis.py")
scale_gui = _load("src/abscess/training/scale_calibration_manual.py")
tooth_gui = _load("src/common/preprocessing/manual_tooth_selector.py")

TOOTHSEG_DIR = os.path.join(_D, "abscess", "raw", "rf_tooth_output", "tooth_mask")


def _ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


BLACK_THRESHOLD = 50


def _foreground_mask(img_bgr: np.ndarray, threshold: int = BLACK_THRESHOLD) -> np.ndarray:
    """Tooth-region mask: 255 where the image is not near-black (background)."""
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([threshold, threshold, threshold], np.uint8)
    return cv2.bitwise_not(cv2.inRange(img_bgr, lower, upper))


# ═════════════════════════════════════════════════════════════════════════════
# Manual step 1 — tooth selection (2 clicks) → tooth_clicks/<stem>.json
# ═════════════════════════════════════════════════════════════════════════════
def step_tooth_click(image_name: str, reuse: bool) -> str:
    """Interactive tooth selection: click the tooth centre, then a point along its
    vertical (root) direction, press 's' to save. Feeds selected_tooth_roi.
    With --reuse, an existing click JSON is used instead of opening the window."""
    stem = os.path.splitext(image_name)[0]
    ann_path = os.path.join(TOOTH_CLICKS, stem + ".json")
    if reuse and os.path.exists(ann_path):
        print(f"  [tooth]  reusing existing clicks: {ann_path}")
        return ann_path
    _ensure(TOOTH_CLICKS)
    tooth_gui.ANNOTATION_DIR = TOOTH_CLICKS
    print("  [tooth]  CLICK 1 = tooth centre, CLICK 2 = vertical direction, "
          "then press 's' to save ('r' reset, ESC cancel)")
    tooth_gui.annotate_image(os.path.join(IMAGES_DIR, image_name))
    if not os.path.exists(ann_path):
        raise RuntimeError("tooth not selected — click 2 points and press 's'.")
    print(f"  [tooth]  saved clicks: {ann_path}")
    return ann_path


# ═════════════════════════════════════════════════════════════════════════════
# Manual step 2 — scale calibration (2 clicks on the 10 mm marker) → CSV
# ═════════════════════════════════════════════════════════════════════════════
def step_scale(image_name: str, reuse: bool) -> None:
    existing = scale_gui.load_existing_results(SCALE_CSV) if os.path.exists(SCALE_CSV) else {}
    if reuse:
        if image_name in existing:
            print(f"  [scale]  reusing existing calibration for {image_name}")
        else:
            print(f"  [scale]  --reuse set and no calibration for {image_name} — skipping (mm unavailable)")
        return
    _ensure(os.path.dirname(SCALE_CSV),
            os.path.join(_D, "abscess", "processed", "manual_calibration_visualizations"))
    image = cv2.imread(os.path.join(IMAGES_DIR, image_name))
    if image is None:
        print("  [scale]  image unreadable — skipping scale (mm unavailable)")
        return
    cv2.namedWindow(scale_gui.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print("  [scale]  click the two ends of the 10 mm marker; press 'c' to confirm, 's' to skip")
    outcome = scale_gui.calibrate_image_manual(image, scale_gui.WINDOW_NAME)
    cv2.destroyAllWindows()
    if not isinstance(outcome, dict):
        print("  [scale]  skipped — mm measurements will be unavailable")
        return
    existing[image_name] = {"image_name": image_name, **outcome}
    scale_gui.save_results(SCALE_CSV, existing)
    print(f"  [scale]  mm/px = {outcome['mm_per_pixel']}")


# ═════════════════════════════════════════════════════════════════════════════
# Auto stages
# ═════════════════════════════════════════════════════════════════════════════
def step_roi(image_name: str, ann_path: str) -> str:
    _ensure(ROI_DIR, ROI_DEBUG)
    out = os.path.join(ROI_DIR, image_name)
    roi_mod.extract_selected_tooth_roi(
        image_path=os.path.join(IMAGES_DIR, image_name),
        annotation_path=ann_path,
        save_path=out,
        debug_path=os.path.join(ROI_DEBUG, image_name))
    print(f"  [roi]    -> {out}")
    return out


def step_background_removal(roi_path: str, stem: str) -> str:
    _ensure(BLACKREMOVE)
    img = cv2.imread(roi_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"ROI unreadable: {roi_path}")
    mask = _foreground_mask(img, BLACK_THRESHOLD)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[mask == 0] = 0
    bgra = cv2.merge([gray, gray, gray, mask])
    out = os.path.join(BLACKREMOVE, f"{stem}{SUFFIX}.png")
    cv2.imwrite(out, bgra)
    print(f"  [bgrm]   -> {out}")
    return out


def step_crop(blackremove_path: str, stem: str) -> str:
    _ensure(CROUN_DIR)
    out = os.path.join(CROUN_DIR, f"{stem}{SUFFIX}.png")
    crop_mod.extract_anatomical_region(image_path=blackremove_path, save_path=out, debug_path=None)
    print(f"  [crop]   -> {out}")
    return out


def step_tooth_segmentation(crop_path: str, stem: str) -> str:
    """RF tooth segmentation (tooth_segmentation.py) on the crop → tooth mask."""
    _ensure(TOOTHSEG_DIR)
    out = os.path.join(TOOTHSEG_DIR, f"tooth_mask_{stem}{SUFFIX}.png")
    if not os.path.exists(toothseg_mod.MODEL_PATH):
        print(f"  [tooth-seg] model missing ({toothseg_mod.MODEL_PATH}) — "
              f"run tooth_segmentation.py to train it. Skipping tooth mask.")
        return ""
    model = joblib.load(toothseg_mod.MODEL_PATH)["model"]
    loaded = toothseg_mod.load_gray_valid(crop_path)
    if loaded is None:
        print("  [tooth-seg] crop unreadable — skipping tooth mask.")
        return ""
    gray, valid = loaded
    mask = toothseg_mod.predict_mask(model, gray, valid)
    cv2.imwrite(out, mask)
    print(f"  [tooth-seg]-> {out}  (tooth px: {int((mask > 0).sum())})")
    return out


def step_fcm_segment(crop_path: str, stem: str) -> str:
    """Spatial FCM; darkest cluster = lesion candidate mask (reproduces the
    original lesion_segmentation.py, whose module is currently disabled)."""
    _ensure(SEG_DIR)
    img = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"crop unreadable: {crop_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = img[:, :, 3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        alpha = None
    h, w = gray.shape
    g = gray.astype(np.float64) / 255.0
    valid = (alpha > 0) if alpha is not None else np.ones((h, w), bool)
    fg = np.flatnonzero(valid.reshape(-1))
    mask = np.zeros((h, w), np.uint8)
    if fg.size >= FCM_CLUSTERS * 10:
        yy, xx = np.mgrid[0:h, 0:w]
        feat = np.vstack([
            g.reshape(-1)[fg],
            (xx.reshape(-1)[fg] / max(w - 1, 1)) * FCM_SPATIAL_WEIGHT,
            (yy.reshape(-1)[fg] / max(h - 1, 1)) * FCM_SPATIAL_WEIGHT])
        cntr, u, *_ = fuzz.cluster.cmeans(feat, FCM_CLUSTERS, FCM_M,
                                          error=FCM_ERR, maxiter=FCM_ITER, init=None, seed=42)
        lesion_idx = int(np.argmin(cntr[:, 0]))
        labels = np.argmax(u, axis=0)
        sel = fg[labels == lesion_idx]
        flat = mask.reshape(-1); flat[sel] = 255; mask = flat.reshape(h, w)
        if alpha is not None:
            mask[alpha == 0] = 0
    out = os.path.join(SEG_DIR, f"lesion_mask_{stem}{SUFFIX}.png")
    cv2.imwrite(out, mask)
    print(f"  [seg]    -> {out}  (lesion px: {int((mask > 0).sum())})")
    return out


def step_restore(seg_mask_path: str, stem: str, work: str) -> str:
    _ensure(RESTORED_DIR)
    masks_tmp = os.path.join(work, "seg")
    _ensure(masks_tmp)
    shutil.copy(seg_mask_path, os.path.join(masks_tmp, os.path.basename(seg_mask_path)))
    rest_mod.restore_masks(masks_dir=masks_tmp, crops_dir=CROUN_DIR,
                           black_remove_dir=BLACKREMOVE, output_dir=RESTORED_DIR)
    out = os.path.join(RESTORED_DIR, f"lesion_mask_{stem}{SUFFIX}.png")
    print(f"  [restore]-> {out}")
    return out


def _pick_image() -> str:
    """Open a file dialog so the user can select an X-ray. Returns '' if none."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select an X-ray image",
            initialdir=IMAGES_DIR if os.path.isdir(IMAGES_DIR) else os.getcwd(),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        root.destroy()
        return path or ""
    except Exception as exc:
        print(f"[WARN] file dialog unavailable ({exc}); pass --image instead.")
        return ""


def _show_results(stem: str) -> None:
    """Display the two final visualizations (exactly the files written by
    lesion_measurement_pca.py and lesion_orientation_analysis.py) and hold the
    windows open until the user presses a key."""
    panels = [
        ("Lesion Measurement (PCA)", os.path.join(MEAS_DIR, f"pca_vis_{stem}{SUFFIX}.png")),
        ("Lesion Orientation",       os.path.join(ORIENT_DIR, f"orient_vis_{stem}{SUFFIX}.png")),
    ]
    shown = False
    for x_off, (title, path) in enumerate(panels):
        img = cv2.imread(path)
        if img is None:
            print(f"  [display] not found: {path}")
            continue
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        h = img.shape[0]
        cv2.resizeWindow(title, max(240, int(img.shape[1] * 700 / max(h, 1))), 700)
        cv2.moveWindow(title, 60 + x_off * 380, 60)
        cv2.imshow(title, img)
        shown = True
    if shown:
        print("\n  >>> Showing final outputs. Press any key on a window to close. <<<")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("  [display] no final visualizations to show.")


def _append_csv(path: str, row: dict) -> None:
    row = {k: v for k, v in row.items() if not str(k).startswith("_")}
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            wr.writeheader()
        wr.writerow(row)


def step_measurement(restored_path: str, stem: str) -> None:
    _ensure(MEAS_DIR)
    meas_mod.OUTPUT_DIR = MEAS_DIR
    res = meas_mod.process_mask(restored_path, f"{stem}{SUFFIX}.png")
    if res:
        _append_csv(os.path.join(MEAS_DIR, "lesion_measurements.csv"), res)
    print(f"  [measure]-> {MEAS_DIR}")


def step_orientation(restored_path: str, blackremove_path: str, stem: str) -> None:
    _ensure(ORIENT_DIR)
    orient_mod.OUTPUT_DIR = ORIENT_DIR
    res = orient_mod.process_image_pair(blackremove_path, restored_path, f"{stem}{SUFFIX}.png")
    if res:
        _append_csv(os.path.join(ORIENT_DIR, "lesion_orientation.csv"), res)
    print(f"  [orient] -> {ORIENT_DIR}")


# ═════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════════
def run(image_arg: str, reuse: bool) -> None:
    _ensure(IMAGES_DIR)
    # Accept a full path or a bare filename already in images/.
    if os.path.isabs(image_arg) or os.path.dirname(image_arg):
        src = image_arg
        image_name = os.path.basename(image_arg)
        dst = os.path.join(IMAGES_DIR, image_name)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
    else:
        image_name = image_arg
    if not os.path.exists(os.path.join(IMAGES_DIR, image_name)):
        print(f"[ERROR] image not found: {os.path.join(IMAGES_DIR, image_name)}")
        return
    stem = os.path.splitext(image_name)[0]
    work = os.path.join(WORK_ROOT, stem)
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    _ensure(work)

    print("=" * 64)
    print(f"  PIPELINE  |  {image_name}")
    print("=" * 64)

    ann = step_tooth_click(image_name, reuse)      # manual: select tooth (click)
    step_scale(image_name, reuse)                  # manual: select scale (click)
    roi = step_roi(image_name, ann)                # 1  selected_tooth_roi
    br = step_background_removal(roi, stem)        # 2  BlackRemove
    crop = step_crop(br, stem)                     # 3  crop_croun
    step_tooth_segmentation(crop, stem)            # 4  tooth_segmentation.py (RF tooth mask)
    seg = step_fcm_segment(crop, stem)            # 5  FCM lesion candidates
    restored = step_restore(seg, stem, work)       # 6  restore_lesion_masks
    step_measurement(restored, stem)               # 7  lesion_measurement_pca  OUTPUT
    step_orientation(restored, br, stem)           # 8  lesion_orientation       OUTPUT

    shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 64)
    print("  DONE — final outputs")
    print("=" * 64)
    meas_viz = os.path.join(MEAS_DIR, f"pca_vis_{stem}{SUFFIX}.png")
    orient_viz = os.path.join(ORIENT_DIR, f"orient_vis_{stem}{SUFFIX}.png")
    print(f"  measurement viz : {meas_viz}"
          f"{'' if os.path.exists(meas_viz) else '   [not produced]'}")
    print(f"  orientation viz : {orient_viz}"
          f"{'' if os.path.exists(orient_viz) else '   [not produced]'}")
    print(f"  measurement CSV : {os.path.join(MEAS_DIR, 'lesion_measurements.csv')}")
    print(f"  orientation CSV : {os.path.join(ORIENT_DIR, 'lesion_orientation.csv')}")
    print("=" * 64)

    _show_results(stem)      # pop up the two final visualizations, hold until keypress


def main() -> None:
    ap = argparse.ArgumentParser(description="Single X-ray → lesion measurement + orientation")
    ap.add_argument("--image", default=None,
                    help="X-ray path or a filename in images/. Omit to pick via a file dialog.")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing tooth-click / scale annotations (skip those GUIs)")
    args = ap.parse_args()
    image = args.image or _pick_image()
    if not image:
        print("[INFO] No image selected — nothing to do.")
        return
    run(image, args.reuse)


if __name__ == "__main__":
    main()
