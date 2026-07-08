"""
evaluate_segmentation.py
------------------------
Comprehensive segmentation evaluation workflow for FCM-FRWS lesion segmentation.

Steps performed:
  1. Load ground truth masks  (from data/abscess/raw/mask/)
  2. Load predicted masks      (from data/abscess/raw/lesion_postprocessed/)
  3. Compute per-image metrics: Dice, IoU, Precision, Recall, F1
  4. Analyse over- & under-segmentation (FP %, FN %)
  5. Generate error-visualisation overlays (Green / Red / Blue)
  6. Print a summary table and save CSV + per-image visualisations

Ground truth naming : <ID>_clahe_sigmoid_mask_1_<ID>.png   (e.g. L10_clahe_sigmoid_mask_1_L10.png)
Predicted naming    : lesion_mask_<ID>_clahe_sigmoid.png    (e.g. lesion_mask_L10_clahe_sigmoid.png)

Usage:
    python evaluate_segmentation.py
"""

import os
import re
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving charts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH_DIR = '../../../data/abscess/raw/mask'
PREDICTED_DIR    = '../../../data/abscess/raw/lesion_postprocessed'
OUTPUT_DIR       = '../../../data/abscess/raw/segmentation_evaluation_results'
CROUN_CROP_DIR   = '../../../data/abscess/raw/croun_crops'

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg')
DICE_THRESHOLD = 0.7   # Dice >= threshold => PASS, else FAIL


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract sample ID from a filename
# ─────────────────────────────────────────────────────────────────────────────
def extract_id(filename: str) -> str:
    """
    Extract the sample identifier (e.g. 'L10') from either naming convention.

    Ground truth : L10_clahe_sigmoid_mask_1_L10.png  →  L10
    Predicted    : lesion_mask_L10_clahe_sigmoid.png  →  L10
    """
    name = os.path.splitext(filename)[0]

    # Predicted mask pattern: lesion_mask_<ID>_clahe_sigmoid
    m = re.match(r'lesion_mask_(L\d+)_', name)
    if m:
        return m.group(1)

    # Ground truth pattern: <ID>_clahe_sigmoid_mask_1_<ID>
    m = re.match(r'(L\d+)_', name)
    if m:
        return m.group(1)

    return name  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute pixel-level segmentation metrics between binary masks.

    Returns dict with: TP, FP, FN, TN, Dice, IoU, Precision, Recall, F1,
                       OverSeg%, UnderSeg%
    """
    gt_bin   = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)

    tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))

    # ── Core metrics ─────────────────────────────────────────────────────────
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    dice      = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # ── Over / Under segmentation (Step 4) ───────────────────────────────────
    total_pred_pos = tp + fp
    total_gt_pos   = tp + fn

    over_seg_pct  = (fp / total_pred_pos * 100) if total_pred_pos > 0 else 0.0
    under_seg_pct = (fn / total_gt_pos * 100)   if total_gt_pos   > 0 else 0.0

    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Dice': dice,
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'OverSeg%': over_seg_pct,
        'UnderSeg%': under_seg_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Error visualisation
# ─────────────────────────────────────────────────────────────────────────────
def create_error_visualisation(gt: np.ndarray, pred: np.ndarray,
                                original_img: np.ndarray = None) -> np.ndarray:
    """
    Build an RGB error map:
        Green  (0, 255, 0)  = True Positive  – correctly segmented lesion
        Red    (0, 0, 255)  = False Positive  – over-segmented area
        Blue   (255, 0, 0)  = False Negative  – missed lesion area
    If an original image is provided, use it as a dimmed background.
    """
    gt_bin   = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)

    h, w = gt_bin.shape[:2]

    # Start with dimmed original or dark grey background
    if original_img is not None:
        if len(original_img.shape) == 2:
            bg = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        elif original_img.shape[2] == 4:
            bg = original_img[:, :, :3].copy()
        else:
            bg = original_img.copy()
        # Resize if necessary
        if bg.shape[:2] != (h, w):
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
        vis = (bg * 0.3).astype(np.uint8)  # dim the background
    else:
        vis = np.zeros((h, w, 3), dtype=np.uint8)

    # True Positive → Green
    tp_mask = (pred_bin == 1) & (gt_bin == 1)
    vis[tp_mask] = [0, 255, 0]

    # False Positive → Red  (over-segmentation)
    fp_mask = (pred_bin == 1) & (gt_bin == 0)
    vis[fp_mask] = [0, 0, 255]

    # False Negative → Blue (under-segmentation / missed)
    fn_mask = (pred_bin == 0) & (gt_bin == 1)
    vis[fn_mask] = [255, 0, 0]

    return vis


def create_side_by_side(gt: np.ndarray, pred: np.ndarray,
                         error_vis: np.ndarray, sample_id: str,
                         metrics: dict) -> np.ndarray:
    """
    Build a presentation-ready side-by-side comparison:
        [Ground Truth] | [Predicted Mask] | [Error Map]
    with metric annotations at the bottom.
    """
    h, w = gt.shape[:2]

    # Convert single-channel masks to BGR for display
    gt_bgr   = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    # Resize all panels to same height
    panel_h = max(h, 200)
    panel_w = max(w, 200)

    gt_bgr    = cv2.resize(gt_bgr,    (panel_w, panel_h))
    pred_bgr  = cv2.resize(pred_bgr,  (panel_w, panel_h))
    error_vis_r = cv2.resize(error_vis, (panel_w, panel_h))

    # Create labels
    label_h = 40
    gap = 10

    total_w = panel_w * 3 + gap * 2
    total_h = panel_h + label_h + 80  # extra space for metrics

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # Place panels
    labels = ['Ground Truth', 'Predicted (FCM-FRWS)', 'Error Map']
    for i, (panel, label) in enumerate(zip([gt_bgr, pred_bgr, error_vis_r], labels)):
        x_off = i * (panel_w + gap)
        # Label
        cv2.putText(canvas, label, (x_off + 5, label_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        # Image
        canvas[label_h:label_h + panel_h, x_off:x_off + panel_w] = panel

    # Metric text at the bottom
    metric_y = label_h + panel_h + 20
    metric_text = (
        f"{sample_id}  |  "
        f"Dice: {metrics['Dice']:.4f}  |  "
        f"IoU: {metrics['IoU']:.4f}  |  "
        f"Prec: {metrics['Precision']:.4f}  |  "
        f"Rec: {metrics['Recall']:.4f}  |  "
        f"F1: {metrics['F1']:.4f}"
    )
    cv2.putText(canvas, metric_text, (10, metric_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    seg_text = (
        f"Over-seg: {metrics['OverSeg%']:.2f}%  |  "
        f"Under-seg: {metrics['UnderSeg%']:.2f}%  |  "
        f"TP: {metrics['TP']}  FP: {metrics['FP']}  FN: {metrics['FN']}"
    )
    cv2.putText(canvas, seg_text, (10, metric_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Legend
    legend_x = total_w - 300
    legend_y = metric_y
    for colour, text in [((0, 255, 0), 'TP (Correct)'),
                          ((0, 0, 255), 'FP (Over-seg)'),
                          ((255, 0, 0), 'FN (Missed)')]:
        cv2.rectangle(canvas, (legend_x, legend_y - 10), (legend_x + 12, legend_y + 2), colour, -1)
        cv2.putText(canvas, text, (legend_x + 18, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        legend_y += 18

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Segmentation Evaluation - FCM-FRWS Lesion Segmentation")
    print("=" * 70)
    print(f"  Ground Truth dir : {GROUND_TRUTH_DIR}")
    print(f"  Predicted dir    : {PREDICTED_DIR}")
    print(f"  Output dir       : {OUTPUT_DIR}")
    print("=" * 70)

    # ── Build lookup dictionaries keyed by sample ID ─────────────────────────
    gt_files = {}
    for f in os.listdir(GROUND_TRUTH_DIR):
        if f.lower().endswith(SUPPORTED_EXTS):
            sid = extract_id(f)
            gt_files[sid] = os.path.join(GROUND_TRUTH_DIR, f)

    pred_files = {}
    for f in os.listdir(PREDICTED_DIR):
        if f.lower().startswith('lesion_mask_') and f.lower().endswith(SUPPORTED_EXTS):
            sid = extract_id(f)
            pred_files[sid] = os.path.join(PREDICTED_DIR, f)

    # ── Find common IDs ──────────────────────────────────────────────────────
    common_ids = sorted(set(gt_files.keys()) & set(pred_files.keys()),
                        key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    if not common_ids:
        print("\n[ERROR] No matching samples found between ground truth and predicted masks!")
        print(f"  GT IDs   : {sorted(gt_files.keys())}")
        print(f"  Pred IDs : {sorted(pred_files.keys())}")
        return

    print(f"\nFound {len(common_ids)} matched sample(s): {', '.join(common_ids)}\n")

    # ── Evaluate each pair ───────────────────────────────────────────────────
    all_metrics = []
    csv_rows = []

    for sid in common_ids:
        print(f"-- {sid} {'-' * 55}")
        gt_path   = gt_files[sid]
        pred_path = pred_files[sid]

        gt_mask   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt_mask is None or pred_mask is None:
            print(f"  [WARN] Could not read masks for {sid} - skipping.")
            continue

        # Resize predicted to match ground truth if sizes differ
        if gt_mask.shape != pred_mask.shape:
            print(f"  [INFO] Resizing predicted mask {pred_mask.shape} -> {gt_mask.shape}")
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # ── Step 3: Compute metrics ──────────────────────────────────────────
        metrics = compute_metrics(gt_mask, pred_mask)
        all_metrics.append(metrics)

        # Determine PASS / FAIL
        status = 'PASS' if metrics['Dice'] >= DICE_THRESHOLD else 'FAIL'

        print(f"  Dice      : {metrics['Dice']:.4f}  [{status}]")
        print(f"  IoU       : {metrics['IoU']:.4f}")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  Recall    : {metrics['Recall']:.4f}")
        print(f"  F1        : {metrics['F1']:.4f}")
        print(f"  Over-seg  : {metrics['OverSeg%']:.2f}%")
        print(f"  Under-seg : {metrics['UnderSeg%']:.2f}%")

        csv_rows.append({'SampleID': sid, 'Status': status, **metrics})

        # ── Step 5: Error visualisation ──────────────────────────────────────
        # Try to load original croun crop for background
        orig_img = None
        for ext in SUPPORTED_EXTS:
            orig_candidate = os.path.join(CROUN_CROP_DIR, f"{sid}_clahe_sigmoid{ext}")
            if os.path.exists(orig_candidate):
                orig_img = cv2.imread(orig_candidate, cv2.IMREAD_UNCHANGED)
                break

        error_vis = create_error_visualisation(gt_mask, pred_mask, orig_img)
        error_path = os.path.join(OUTPUT_DIR, f"error_map_{sid}.png")
        cv2.imwrite(error_path, error_vis)
        print(f"  -> Error map saved : {error_path}")

        # Side-by-side comparison
        comparison = create_side_by_side(gt_mask, pred_mask, error_vis, sid, metrics)
        comp_path = os.path.join(OUTPUT_DIR, f"comparison_{sid}.png")
        cv2.imwrite(comp_path, comparison)
        print(f"  -> Comparison saved: {comp_path}")

    # ── Aggregate summary ────────────────────────────────────────────────────
    if all_metrics:
        print("\n" + "=" * 70)
        print("  AGGREGATE RESULTS")
        print("=" * 70)

        metric_keys = ['Dice', 'IoU', 'Precision', 'Recall', 'F1', 'OverSeg%', 'UnderSeg%']
        header = f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
        print(header)
        print("-" * len(header))

        summary_row = {'SampleID': 'MEAN'}
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            mean_v = np.mean(values)
            std_v  = np.std(values)
            min_v  = np.min(values)
            max_v  = np.max(values)
            summary_row[key] = mean_v
            fmt = '.4f' if '%' not in key else '.2f'
            print(f"  {key:<13} {mean_v:>10{fmt}} {std_v:>10{fmt}} {min_v:>10{fmt}} {max_v:>10{fmt}}")

        # ── Success / Failure Rate ────────────────────────────────────────────
        total    = len(csv_rows)
        n_pass   = sum(1 for r in csv_rows if r['Status'] == 'PASS')
        n_fail   = total - n_pass
        success_rate = (n_pass / total * 100) if total > 0 else 0.0
        failure_rate = (n_fail / total * 100) if total > 0 else 0.0

        print(f"\n  Dice Threshold    : {DICE_THRESHOLD}")
        print(f"  PASS              : {n_pass}/{total}")
        print(f"  FAIL              : {n_fail}/{total}")
        print(f"  Success Rate      : {success_rate:.2f}%")
        print(f"  Failure Rate      : {failure_rate:.2f}%")

        # ── Save CSV report ──────────────────────────────────────────────────
        csv_path = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
        fieldnames = ['SampleID', 'Status'] + metric_keys + ['TP', 'FP', 'FN', 'TN']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
            # Write mean row
            writer.writerow(summary_row)
            # Write success / failure summary rows
            writer.writerow({})
            writer.writerow({'SampleID': 'SUMMARY', 'Status': f'Threshold={DICE_THRESHOLD}'})
            writer.writerow({'SampleID': 'Success Rate', 'Status': f'{success_rate:.2f}%',
                             'Dice': f'{n_pass}/{total}'})
            writer.writerow({'SampleID': 'Failure Rate', 'Status': f'{failure_rate:.2f}%',
                             'Dice': f'{n_fail}/{total}'})

        print(f"\n  CSV report saved : {csv_path}")

    print("\n" + "=" * 70)
    print("  Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
