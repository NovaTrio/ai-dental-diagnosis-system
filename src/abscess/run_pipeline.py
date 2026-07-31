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
    2. lesion_preprocess       → BlackRemove   (background removal + CLAHE/sigmoid)
    3. crop_croun              → croun_crops   (root/periapical crop)
    4. tooth_segmentation      → rf_tooth_output/tooth_mask/tooth_mask_<name>.png
    5. lesion_segmentation     → lesion_segmented/lesion_mask_<name>.png
    6. lesion_detection        → lesion_detection_output (viz)  — GATE: if no
                                  lesion is detected, the pipeline STOPS here.
    7. restore_lesion_masks       → lesion_postprocessed_restored
    8. lesion_measurement_pca_mm  → lesion_measurements  (viz + CSV, px + mm)  ← OUTPUT
    9. lesion_orientation         → lesion_orientation    (viz + CSV)          ← OUTPUT

This file only orchestrates: every stage calls the reusable function(s) the
corresponding module already exposes (no processing logic is re-implemented
here) and simply passes each stage's output path to the next stage. Every
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


roi_mod   = _load("src/common/preprocessing/selected_tooth_roi.py")
preproc_mod = _load("src/abscess/preprocessing/lesion_preprocess.py")   # background removal + CLAHE/sigmoid
crop_mod  = _load("src/abscess/preprocessing/crop_croun.py")
toothseg_mod = _load("src/abscess/Segmentation/tooth_segmentation.py")  # RF tooth segmenter
lesionseg_mod = _load("src/abscess/Segmentation/lesion_segmentation.py")  # RF lesion segmenter
detect_mod = _load("src/abscess/Segmentation/lesion_detection.py")      # RF+SVM lesion detector
rest_mod = _load("src/abscess/preprocessing/restore_lesion_masks.py")
meas_mod = _load("src/abscess/training/lesion_measurement_pca_mm.py")   # PCA measurement + mm conversion
orient_mod = _load("src/abscess/inference/lesion_orientation_analysis.py")
scale_gui = _load("src/abscess/training/scale_calibration_manual.py")
tooth_gui = _load("src/common/preprocessing/manual_tooth_selector.py")

TOOTHSEG_DIR = os.path.join(_D, "abscess", "raw", "rf_tooth_output", "tooth_mask")
DETECT_DIR   = os.path.join(_D, "abscess", "raw", "lesion_detection_output")


def _ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


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
    """Background removal + histogram matching + CLAHE/sigmoid
    (lesion_preprocess.py's own reusable single-image function)."""
    _ensure(BLACKREMOVE)
    out = os.path.join(BLACKREMOVE, f"{stem}{SUFFIX}.png")
    preproc_mod.preprocess_single_image(input_path=roi_path, output_path=out)
    print(f"  [bgrm]   -> {out}")
    return out


def step_crop(blackremove_path: str, stem: str) -> str:
    _ensure(CROUN_DIR)
    out = os.path.join(CROUN_DIR, f"{stem}{SUFFIX}.png")
    crop_mod.extract_anatomical_region(image_path=blackremove_path, save_path=out, debug_path=None)
    print(f"  [crop]   -> {out}")
    return out


def step_tooth_segmentation(crop_path: str, stem: str) -> str:
    """RF tooth segmentation (tooth_segmentation.py's own reusable single-image
    inference) on the crop → tooth mask. Returns '' if the model isn't trained
    yet or the crop is unreadable."""
    _ensure(TOOTHSEG_DIR)
    out = os.path.join(TOOTHSEG_DIR, f"tooth_mask_{stem}{SUFFIX}.png")
    try:
        mask = toothseg_mod.segment_tooth_image(crop_path)
    except FileNotFoundError as exc:
        print(f"  [tooth-seg] {exc}\n  [tooth-seg] skipping tooth mask.")
        return ""
    if mask is None:
        print("  [tooth-seg] crop unreadable — skipping tooth mask.")
        return ""
    cv2.imwrite(out, mask)
    print(f"  [tooth-seg]-> {out}  (tooth px: {int((mask > 0).sum())})")
    return out


def step_lesion_segment(crop_path: str, tooth_mask_path: str, stem: str) -> str:
    """RF lesion segmentation (lesion_segmentation.py's own reusable single-
    image inference) on the crop + tooth mask → lesion candidate mask. Falls
    back to an all-black mask (no candidates) if the model isn't trained yet,
    so the rest of the pipeline (detection gate) still runs and correctly
    reports "no lesion"."""
    _ensure(SEG_DIR)
    out = os.path.join(SEG_DIR, f"lesion_mask_{stem}{SUFFIX}.png")
    try:
        mask = lesionseg_mod.segment_lesion_image(
            crop_path, tooth_mask=tooth_mask_path or None)
    except FileNotFoundError as exc:
        print(f"  [seg] {exc}\n  [seg] using an empty candidate mask.")
        mask = None
    if mask is None:
        loaded = lesionseg_mod.load_gray_valid(crop_path)
        if loaded is None:
            raise RuntimeError(f"crop unreadable: {crop_path}")
        gray, _ = loaded
        mask = np.zeros(gray.shape, np.uint8)
    cv2.imwrite(out, mask)
    print(f"  [seg]    -> {out}  (lesion px: {int((mask > 0).sum())})")
    return out


def step_lesion_detection(crop_path: str, seg_mask_path: str, tooth_mask_path: str,
                          stem: str) -> bool:
    """RF+SVM lesion detector (lesion_detection.py) on the segmentation-stage
    candidate mask. This is the GATE: returns True if a lesion is detected,
    False otherwise — the caller stops the pipeline (skips restore/
    measurement/orientation) on False. If the detector models haven't been
    trained yet, the gate is skipped (treated as detected) so the rest of the
    pipeline still runs."""
    detect_mod._ensure_dirs()
    try:
        models = detect_mod.load_lesion_detectors()
    except FileNotFoundError as exc:
        print(f"  [detect] detector model missing ({exc}); skipping gate, continuing.")
        return True
    result = detect_mod.detect_lesion_single(
        crop_path, fcm_mask=seg_mask_path,
        tooth_mask=tooth_mask_path or None, models=models)
    if result is None:
        print("  [detect] crop unreadable — skipping gate, continuing.")
        return True

    full_stem = f"{stem}{SUFFIX}"
    best_id = None
    if result["candidates"]:
        best_id = max(result["candidates"], key=lambda c: c["rule_score"])["id"]
    detect_mod._save_detection(full_stem, result["gray"], result["candidates"],
                               best_id, bool(result["detected"]))
    detect_mod._save_explanation(full_stem, result["gray"],
                                 result["best_mask"] if result["detected"] else None,
                                 result)
    print(f"  [detect] ensemble_prob={result['ensemble_prob']:.3f}  "
          f"detected={result['detected']}  -> {DETECT_DIR}")
    return bool(result["detected"])


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
    """PCA lesion measurement (lesion_measurement_pca_mm.py's own reusable
    process_mask), converted px → mm using the calibration this pipeline
    already collected in step_scale (SCALE_CSV)."""
    _ensure(MEAS_DIR)
    meas_mod.OUTPUT_DIR = MEAS_DIR
    calibration_lookup = meas_mod.load_calibration_lookup(SCALE_CSV)
    res = meas_mod.process_mask(restored_path, f"{stem}{SUFFIX}.png", calibration_lookup)
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
    tooth_mask = step_tooth_segmentation(crop, stem)          # 4  tooth_segmentation.py (RF tooth mask)
    seg = step_lesion_segment(crop, tooth_mask, stem)          # 5  lesion_segmentation.py (RF lesion mask)
    detected = step_lesion_detection(crop, seg, tooth_mask, stem)  # 6  lesion_detection GATE

    if not detected:
        shutil.rmtree(work, ignore_errors=True)
        print("\n" + "=" * 64)
        print("  STOPPED — no lesion detected")
        print("=" * 64)
        print(f"  detection viz   : {os.path.join(DETECT_DIR, 'detection', f'detection_{stem}{SUFFIX}.png')}")
        print(f"  explanation viz : {os.path.join(DETECT_DIR, 'explain', f'explain_{stem}{SUFFIX}.png')}")
        print("  (restore / measurement / orientation steps skipped)")
        print("=" * 64)
        return

    restored = step_restore(seg, stem, work)       # 7  restore_lesion_masks
    step_measurement(restored, stem)               # 8  lesion_measurement_pca  OUTPUT
    step_orientation(restored, br, stem)            # 9  lesion_orientation       OUTPUT

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
