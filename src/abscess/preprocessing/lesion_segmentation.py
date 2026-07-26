"""
lesion_segmentation.py
──────────────────────
Machine-vision periapical lesion segmentation on N4-bias-corrected crown crops.

This replaces the previous "darkest-cluster" Fuzzy C-Means approach (which
declared the lowest-intensity cluster to be the lesion — unreliable, because
root canals, marrow spaces, the PDL space and image borders are also
radiolucent). It now runs an anatomically-constrained, multi-feature spatial
FCM lesion-candidate pipeline with feature-based validation.

Pipeline (all classical machine vision — no deep learning):
  1. Load N4-corrected crop (BGRA; alpha = tooth foreground).
  2. Light edge-preserving denoise (no CLAHE/Sigmoid — N4Output.py already
     normalised intensity; re-enhancing would undo that).
  3. Restrict to an anatomical ROI (centred mid-root band, or apical band when
     a real root apex can be detected).
  4. Describe each ROI pixel with a multi-feature vector (intensity, local
     mean/std, gradient, Laplacian, Gabor, LBP, entropy, x, y, apex-distance).
  5. Spatial Fuzzy C-Means clustering (m = 2, fixed seed).
  6. Rank clusters by darkness + texture + local contrast + apical proximity
     and take the lesion-candidate cluster(s).
  7. Morphological cleanup → connected components → per-candidate geometric /
     intensity / texture / anatomical features → transparent weighted score.
  8. Keep candidates above threshold (empty mask if none — never forces a lesion).

The heavy algorithm lives in ``lesion_segmentation_advanced.py`` and is reused
here so both files stay in sync; this file is the driver that wires the N4
input, the downstream-compatible output naming and the evaluation together.
The key tunables are surfaced in the CONFIG block below.

Input  : ../../../data/abscess/raw/N4Output/<id>_clahe_sigmoid.png
Output : ../../../data/abscess/raw/lesion_segmented/
            lesion_mask_<id>_clahe_sigmoid.png     (white = lesion, black = bg)
            segmap_<id>_clahe_sigmoid.png          (FCM cluster map, debug)
            lesion_overlay_<id>_clahe_sigmoid.png  (lesion drawn on the crop)
GT     : ../../../data/abscess/raw/mask/  (optional evaluation)

Usage:
    python lesion_segmentation.py
"""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

import lesion_segmentation_advanced as adv  # shared classical-CV algorithm library


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw", "N4Output")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw", "lesion_segmented")
MASK_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw", "mask")

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")
RUN_EVALUATION = True
# Many GT masks were drawn on an OLDER crop size than the current crops. When
# True, evaluation additionally resizes each mismatched mask to the crop and
# reports an APPROXIMATE metric over all images (boundaries won't align exactly,
# so treat it as indicative). The exact-match metric over dimension-matched
# images is always reported and is the trustworthy number.
EVAL_RESIZE_MASKS = True

# Light denoise applied to the N4 output before feature extraction.
# (Deliberately NOT CLAHE/Sigmoid — N4Output.py already normalised intensity.)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 40.0
BILATERAL_SIGMA_SPACE = 40.0

# ── Segmentation tunables (override the shared algorithm module) ──────────────
# Anatomical ROI
adv.TOOTH_ORIENTATION = "auto"       # "mandibular" | "maxillary" | "auto"
adv.APICAL_ROI_FRACTION = 0.65
adv.FALLBACK_ROI_MODE = "centered"   # "centered" | "apical_end"
# Spatial FCM
adv.N_CLUSTERS = 3
adv.FUZZINESS = 2.0
# Candidate cleanup / validation
adv.MIN_COMPONENT_AREA_FRAC = 0.0015
adv.MIN_AREA_FRAC = 0.0015
adv.GATE_ON_SOLIDITY = False
adv.SCORE_THRESHOLD = 0.50
adv.SELECT_MODE = "all"              # "all" (retain every valid lesion) | "best"
# Optional boundary refinement
adv.ENABLE_REFINEMENT = False


# ═════════════════════════════════════════════════════════════════════════════
# Preprocessing (N4 already applied upstream)
# ═════════════════════════════════════════════════════════════════════════════
# def light_preprocess(gray: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Edge-preserving denoise of the already-N4-corrected grayscale crop.

#     Returns (enhanced_uint8, enhanced_float01). Invalid (transparent) pixels are
#     zeroed so they cannot influence clustering.
#     """
#     enhanced = cv2.bilateralFilter(
#         gray, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
#     )
#     enhanced = enhanced.copy()
#     enhanced[~valid] = 0
#     return enhanced, enhanced.astype(np.float64) / 255.0


# ═════════════════════════════════════════════════════════════════════════════
# Segmentation of a single image
# ═════════════════════════════════════════════════════════════════════════════
def segment_image(path: str, filename: str) -> Dict[str, object]:
    """
    Run the full machine-vision segmentation on one N4-corrected crop and write
    the lesion mask, cluster map and overlay. Returns a small status dict.
    """
    stem = os.path.splitext(filename)[0]
    result = {"filename": filename, "status": "ok", "final_area": 0,
              "num_candidates": 0, "num_accepted": 0, "fpc": float("nan")}

    loaded = adv.load_image(path)
    if loaded is None:
        result["status"] = "unreadable"
        print(f"  [WARN] Could not read {filename}")
        return result

    gray, valid = loaded
    h, w = gray.shape

    # Guard tiny / near-empty crops.
    if h < 16 or w < 16 or int(valid.sum()) < adv.N_CLUSTERS * 20:
        result["status"] = "too_small"
        _write_mask(stem, np.zeros((h, w), np.uint8))
        print(f"  [WARN] {filename}: too small / little foreground")
        return result

    # 1. Light denoise (N4 already applied).
    enhanced_u8, enhanced_f = light_preprocess(gray, valid)

    # 2. Orientation + apex + ROI.
    orientation = adv.detect_orientation(enhanced_f, valid)
    apex, apex_trusted = adv.estimate_apex(valid, orientation)
    roi = adv.build_apical_roi(valid, orientation, apex_trusted)
    if not apex_trusted and adv.FALLBACK_ROI_MODE == "centered":
        ys_roi, xs_roi = np.nonzero(roi)
        if ys_roi.size:
            apex = (int(xs_roi.mean()), int(ys_roi.mean()))
    axis_x = w / 2.0

    if int(roi.sum()) < adv.N_CLUSTERS * 20:
        result["status"] = "empty_roi"
        _write_mask(stem, np.zeros((h, w), np.uint8))
        print(f"  [WARN] {filename}: ROI too small")
        return result

    # 3-4. Feature maps → ROI feature matrix.
    maps = adv.compute_feature_maps(enhanced_f, enhanced_u8, apex)
    features, roi_index, raw_by_name = adv.stack_roi_features(maps, roi)

    # 5. Spatial FCM.
    labels, _cntr, fpc = adv.run_spatial_fcm(features)
    result["fpc"] = fpc

    # 6. Lesion-candidate cluster(s) → raw candidate mask, then cleanup.
    cand_clusters, _ = adv.rank_clusters(labels, raw_by_name)
    cand_flat = np.zeros(h * w, dtype=np.uint8)
    cand_flat[roi_index[np.isin(labels, cand_clusters)]] = 255
    raw_mask = cand_flat.reshape(h, w)

    cleaned = adv.clean_mask(raw_mask)
    cleaned &= (roi.astype(np.uint8) * 255)

    # 7. Per-candidate features + scoring.
    n_lab, lab_img = cv2.connectedComponents((cleaned > 0).astype(np.uint8), 8)
    img_area = h * w
    candidates: List[dict] = []
    for cid in range(1, n_lab):
        comp = lab_img == cid
        extracted = adv.extract_candidate_features(
            comp, enhanced_f, enhanced_u8, roi, apex, axis_x)
        if extracted is None:
            continue
        feats, _ring = extracted
        score, accepted, _reasons, _sub = adv.score_candidate(feats, img_area)
        candidates.append({"candidate_id": cid, "mask": comp,
                           "features": feats, "lesion_score": score,
                           "accepted": accepted})

    # 8. Final selection (empty mask if nothing qualifies).
    final_mask, _ids = adv.select_final_lesions(candidates, (h, w))

    result["num_candidates"] = len(candidates)
    result["num_accepted"] = sum(1 for c in candidates if c["accepted"])
    result["final_area"] = int((final_mask > 0).sum())

    # ── Save outputs (downstream-compatible naming) ──────────────────────────
    _write_mask(stem, final_mask)
    _write_segmap(stem, labels, roi_index, (h, w))
    _write_overlay(stem, enhanced_u8, final_mask)

    print(f"  {filename}: cand={result['num_candidates']} "
          f"accepted={result['num_accepted']} area={result['final_area']} "
          f"orient={orientation}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Output writers
# ═════════════════════════════════════════════════════════════════════════════
def _write_mask(stem: str, mask: np.ndarray) -> None:
    """Binary lesion mask: white = lesion, black = background."""
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"lesion_mask_{stem}.png"), mask)


def _write_segmap(stem: str, labels: np.ndarray, roi_index: np.ndarray,
                  shape: Tuple[int, int]) -> None:
    """Colour-mapped FCM cluster label map (debug artifact)."""
    lab = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    lab[roi_index] = (labels + 1) * (255 // (adv.N_CLUSTERS + 1))
    lab = lab.reshape(shape)
    colored = cv2.applyColorMap(lab, cv2.COLORMAP_JET)
    colored[lab == 0] = 0
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"segmap_{stem}.png"), colored)


def _write_overlay(stem: str, enhanced_u8: np.ndarray, final_mask: np.ndarray) -> None:
    """Lesion region highlighted on the enhanced crop."""
    vis = cv2.cvtColor(enhanced_u8, cv2.COLOR_GRAY2BGR)
    if final_mask.max() > 0:
        tint = vis.copy()
        tint[final_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"lesion_overlay_{stem}.png"), vis)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation (only on images whose GT mask matches the crop dimensions)
# ═════════════════════════════════════════════════════════════════════════════
def _find_mask(stem: str) -> str:
    for f in os.listdir(MASK_DIR):
        if f.lower().endswith(SUPPORTED_EXTS) and f.startswith(f"{stem}_mask"):
            return os.path.join(MASK_DIR, f)
    return ""


def evaluate() -> None:
    """
    Compare predicted lesion masks with ground truth. Images whose GT mask does
    not match the current crop dimensions are SKIPPED (the crops were
    regenerated after the masks were drawn) and counted separately, so the
    reported metrics stay meaningful.
    """
    if not os.path.isdir(MASK_DIR):
        print("  [WARN] No mask directory — skipping evaluation.")
        return

    def metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
        p = pred > 127
        g = gt > 127
        tp = int(np.sum(p & g))
        fp = int(np.sum(p & ~g))
        fn = int(np.sum(~p & g))
        return (adv._safe_div(2 * tp, 2 * tp + fp + fn),
                adv._safe_div(tp, tp + fp + fn),
                adv._safe_div(tp, tp + fp),
                adv._safe_div(tp, tp + fn))

    exact: List[Tuple] = []      # dimension-matched (trustworthy)
    approx: List[Tuple] = []     # all comparable, mask resized (indicative)
    skipped = 0
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if not (f.startswith("lesion_mask_") and f.endswith(".png")):
            continue
        stem = f[len("lesion_mask_"):-len(".png")]
        gt_path = _find_mask(stem)
        if not gt_path:
            continue
        pred = cv2.imread(os.path.join(OUTPUT_DIR, f), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        if pred.shape == gt.shape:
            m = metrics(pred, gt)
            exact.append(m)
            approx.append(m)
        else:
            skipped += 1
            if EVAL_RESIZE_MASKS:
                gt_r = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
                approx.append(metrics(pred, gt_r))

    def report(title: str, data: List[Tuple]) -> None:
        print(f"\n  {title}  (n={len(data)})")
        if not data:
            return
        arr = np.array(data)
        for i, name in enumerate(["Dice", "IoU", "Precision", "Recall"]):
            print(f"    {name:10s}: {arr[:, i].mean():.4f} ± {arr[:, i].std():.4f}")

    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    report("Exact (dimension-matched masks) — trustworthy", exact)
    if EVAL_RESIZE_MASKS:
        report("Approximate (mismatched masks resized to crop) — indicative", approx)
    if skipped:
        print(f"\n  [NOTE] {skipped} image(s) have GT masks drawn on an older crop"
              f"\n         size. The exact metric excludes them; the approximate"
              f"\n         metric resizes their masks (boundaries won't align)."
              f"\n         Regenerate masks on the current crops for a clean score.")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 60)
    print("  Machine-vision lesion segmentation (input: N4Output)")
    print("=" * 60)
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        print("        Run N4Output.py first to generate the N4-corrected crops.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(SUPPORTED_EXTS))
    if not files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return
    print(f"\nProcessing {len(files)} image(s).\n")

    for filename in files:
        try:
            segment_image(os.path.join(INPUT_DIR, filename), filename)
        except Exception as exc:            # keep the batch alive
            print(f"  [ERROR] {filename}: {type(exc).__name__}: {exc}")

    if RUN_EVALUATION:
        evaluate()

    print("\n" + "=" * 60)
    print("  Segmentation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
