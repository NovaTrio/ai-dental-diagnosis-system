"""
evaluate_diameter.py
─────────────────────
Evaluates the predicted lesion MAJOR DIAMETER (from lesion_measurement_pca_mm.py's
output, in millimeters) against the ground-truth diameters recorded in
lesion_detect.csv — for images L1 through L61 only.

Ground truth : src/abscess/training/lesion_detect.csv
                columns: Label, Lesion_Detection, lesion_Diameter (mm), lesion_orientation
Predicted    : data/abscess/raw/lesion_measurements/lesion_measurements.csv
                column : major_diameter_mm  (matched to a Label via its filename)

Only L1-L61 are considered. Within that range, a sample is SCORED only if it
has both a ground-truth diameter and a predicted diameter; samples missing
either (no lesion / diameter not recorded / not yet measured by the pipeline)
are still listed in the CSV with a status, not silently dropped.

Outputs (../../../data/abscess/raw/diameter_evaluation_results/):
    diameter_evaluation.csv   per-image GT vs predicted + summary metrics
    diameter_scatter.png      GT vs predicted scatter plot (y = x reference line)

Usage:
    python evaluate_diameter.py
"""

import os
import re
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving charts
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

GT_CSV      = os.path.join(PROJECT_ROOT, "src", "abscess", "training", "lesion_detect.csv")
PRED_CSV    = os.path.join(PROJECT_ROOT, "data", "abscess", "raw",
                           "lesion_measurements", "lesion_measurements.csv")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "abscess", "raw", "diameter_evaluation_results")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "diameter_evaluation.csv")
SCATTER_PNG = os.path.join(OUTPUT_DIR, "diameter_scatter.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_LABEL_ID, MAX_LABEL_ID = 1, 61   # evaluate L1..L61 only


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _label_num(label: str):
    m = re.match(r"L(\d+)$", str(label).strip())
    return int(m.group(1)) if m else None


def resolve_pred_label(filename: str):
    """'lesion_mask_L10_clahe_sigmoid.png' -> 'L10'."""
    m = re.match(r"lesion_mask_(L\d+)_", str(filename))
    return m.group(1) if m else None


def load_ground_truth(csv_path: str) -> dict:
    """{'L1': 13.25, ...} — labels L1..L61 with a lesion actually present
    (Lesion_Detection == 1) and a positive recorded diameter. Rows with no
    lesion use diameter 0 as a placeholder, not a real measurement, so
    they're excluded rather than scored as a diameter match."""
    df = pd.read_csv(csv_path)
    gt = {}
    for _, row in df.iterrows():
        n = _label_num(row.get("Label", ""))
        if n is None or not (MIN_LABEL_ID <= n <= MAX_LABEL_ID):
            continue
        if row.get("Lesion_Detection") != 1:
            continue
        diam = row.get("lesion_Diameter")
        if pd.isna(diam) or float(diam) <= 0:
            continue
        gt[f"L{n}"] = float(diam)
    return gt


def load_predictions(csv_path: str) -> dict:
    """{'L1': major_diameter_mm, ...}."""
    df = pd.read_csv(csv_path)
    pred = {}
    for _, row in df.iterrows():
        label = resolve_pred_label(row.get("filename", ""))
        if label is None:
            continue
        val = row.get("major_diameter_mm")
        if pd.isna(val):
            continue
        pred[label] = float(val)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  Lesion Major-Diameter Evaluation (L{MIN_LABEL_ID}-L{MAX_LABEL_ID}) vs Ground Truth")
    print("=" * 70)
    print(f"  Ground truth CSV : {GT_CSV}")
    print(f"  Predictions CSV  : {PRED_CSV}")
    print(f"  Output dir       : {OUTPUT_DIR}")
    print("=" * 70)

    if not os.path.exists(GT_CSV):
        print(f"\n[ERROR] ground truth CSV not found: {GT_CSV}")
        return
    if not os.path.exists(PRED_CSV):
        print(f"\n[ERROR] predictions CSV not found: {PRED_CSV}")
        print("        Run run_pipeline.py (or lesion_measurement_pca_mm.py) first.")
        return

    gt = load_ground_truth(GT_CSV)
    pred = load_predictions(PRED_CSV)

    all_labels = [f"L{n}" for n in range(MIN_LABEL_ID, MAX_LABEL_ID + 1)]

    rows = []
    for label in all_labels:
        gt_v, pred_v = gt.get(label), pred.get(label)
        if gt_v is None and pred_v is None:
            status = "no_gt_no_pred"
        elif gt_v is None:
            status = "no_gt"
        elif pred_v is None:
            status = "no_pred"
        else:
            status = "matched"

        row = {"Label": label, "gt_diameter_mm": gt_v, "pred_diameter_mm": pred_v,
               "status": status, "abs_error_mm": None, "signed_error_mm": None,
               "pct_error": None}
        if status == "matched":
            err = pred_v - gt_v
            row["abs_error_mm"] = round(abs(err), 3)
            row["signed_error_mm"] = round(err, 3)
            row["pct_error"] = round(abs(err) / gt_v * 100, 2) if gt_v != 0 else None
        rows.append(row)

    matched = [r for r in rows if r["status"] == "matched"]
    print(f"\n  Labels in range      : {len(all_labels)}  (L{MIN_LABEL_ID}-L{MAX_LABEL_ID})")
    print(f"  Matched (GT + pred)  : {len(matched)}")
    print(f"  Missing ground truth : {sum(1 for r in rows if r['status'] in ('no_gt', 'no_gt_no_pred'))}")
    print(f"  Missing prediction   : {sum(1 for r in rows if r['status'] in ('no_pred', 'no_gt_no_pred'))}")

    mae = rmse = mape = bias = corr = None
    if matched:
        gt_arr = np.array([r["gt_diameter_mm"] for r in matched])
        pred_arr = np.array([r["pred_diameter_mm"] for r in matched])
        errors = pred_arr - gt_arr
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape = float(np.mean(abs_errors / gt_arr) * 100)
        bias = float(np.mean(errors))
        corr = float(np.corrcoef(gt_arr, pred_arr)[0, 1]) if len(matched) > 1 else float("nan")

        print("\n" + "-" * 70)
        print("  METRICS (matched samples only)")
        print("-" * 70)
        print(f"  MAE                     : {mae:.3f} mm")
        print(f"  RMSE                    : {rmse:.3f} mm")
        print(f"  MAPE                    : {mape:.2f} %")
        print(f"  Bias (pred - gt, mean)  : {bias:+.3f} mm")
        print(f"  Pearson r               : {corr:.4f}")

        # ── Scatter plot: GT vs predicted ────────────────────────────────────
        plt.figure(figsize=(6, 6))
        lim = float(max(gt_arr.max(), pred_arr.max())) * 1.1
        plt.plot([0, lim], [0, lim], 'k--', linewidth=1, label='y = x (ideal)')
        plt.scatter(gt_arr, pred_arr, c='tab:blue', edgecolors='k', s=40, zorder=3)
        for r in matched:
            plt.annotate(r["Label"], (r["gt_diameter_mm"], r["pred_diameter_mm"]),
                        fontsize=7, xytext=(3, 3), textcoords='offset points')
        plt.xlim(0, lim)
        plt.ylim(0, lim)
        plt.xlabel("Ground truth major diameter (mm)")
        plt.ylabel("Predicted major diameter (mm)")
        plt.title(f"Predicted vs Ground Truth major diameter  (n={len(matched)})\n"
                  f"MAE={mae:.2f}mm  RMSE={rmse:.2f}mm  MAPE={mape:.1f}%  r={corr:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(SCATTER_PNG, dpi=150)
        plt.close()
        print(f"\n  Scatter plot saved : {SCATTER_PNG}")
    else:
        print("\n[WARN] No matched samples (GT + prediction both present) — nothing to score.")

    # ── Save CSV report ──────────────────────────────────────────────────────
    fieldnames = ["Label", "gt_diameter_mm", "pred_diameter_mm", "status",
                 "abs_error_mm", "signed_error_mm", "pct_error"]
    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow({k: "" for k in fieldnames})
        writer.writerow({"Label": "SUMMARY"})
        writer.writerow({"Label": "N_matched", "gt_diameter_mm": len(matched)})
        if matched:
            writer.writerow({"Label": "MAE_mm", "gt_diameter_mm": round(mae, 3)})
            writer.writerow({"Label": "RMSE_mm", "gt_diameter_mm": round(rmse, 3)})
            writer.writerow({"Label": "MAPE_pct", "gt_diameter_mm": round(mape, 2)})
            writer.writerow({"Label": "Bias_mm", "gt_diameter_mm": round(bias, 3)})
            writer.writerow({"Label": "Pearson_r", "gt_diameter_mm": round(corr, 4)})

    print(f"  CSV report saved   : {RESULTS_CSV}")
    print("\n" + "=" * 70)
    print("  Diameter evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
