"""
evaluate_lesion_detection.py
─────────────────────────────
Standalone evaluation of the SVM + Random Forest lesion DETECTION models
(``lesion_detection.py``) against the ground truth in ``lesion_detect.csv``.

This script does NOT re-run FCM candidate extraction or feature engineering —
it reads the already-produced ``image_summary.csv`` (stem, split, true_label,
rf_prob, svm_prob, ensemble_prob, detected, n_candidates, status) and scores
RandomForest / SVM / Ensemble against the true labels, split by train/test so
the reported test-set numbers stay honest.

For each of RF, SVM, and the Ensemble it reports:
    - accuracy, precision, recall, specificity, F1
    - confusion matrix (plot + numbers)
    - ROC curve + AUC
    - Precision-Recall curve + average precision
    - probability distribution (lesion vs no-lesion)
    - a threshold sweep (sensitivity/specificity/precision vs decision
      threshold), to help choose DETECT_PROB_THRESHOLD in lesion_detection.py

Inputs
    Predictions : ../../../data/abscess/raw/lesion_detection_output/image_summary.csv
    Ground truth: ../../../src/abscess/training/lesion_detect.csv  (only used to
                  double-check / re-merge if image_summary.csv's true_label is
                  missing for some rows)
Outputs (../../../data/abscess/raw/lesion_detection_output/evaluation/)
    metrics_report.txt                 all metrics, train + test, all 3 models
    metrics_per_model.csv              same numbers in a flat CSV
    confusion_matrix_<model>_<split>.png
    roc_curve_<split>.png              RF vs SVM vs Ensemble overlaid
    pr_curve_<split>.png
    probability_distribution_<split>.png
    threshold_sweep_<split>.csv/.png

Usage:
    python lesion_detection.py             # first, to produce image_summary.csv
    python evaluate_lesion_detection.py    # then this
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw")
DETECTION_OUTPUT_DIR = os.path.join(_RAW, "lesion_detection_output")
IMAGE_SUMMARY_CSV = os.path.join(DETECTION_OUTPUT_DIR, "image_summary.csv")
LABELS_CSV = os.path.join(SCRIPT_DIR, "..", "training", "lesion_detect.csv")
EVAL_DIR = os.path.join(DETECTION_OUTPUT_DIR, "evaluation")

MODELS = ("rf", "svm", "ensemble")
MODEL_LABELS = {"rf": "Random Forest", "svm": "SVM", "ensemble": "Ensemble"}
PROB_COL = {"rf": "rf_prob", "svm": "svm_prob", "ensemble": "ensemble_prob"}
MODEL_COLOR = {"rf": "#2b8a3e", "svm": "#1971c2", "ensemble": "#e8590c"}

DECISION_THRESHOLD = 0.5             # threshold used for the headline metrics
THRESHOLD_SWEEP = np.round(np.arange(0.30, 0.76, 0.05), 2)
SPLITS_TO_REPORT = ("test", "train", "all_labeled")   # "test" first: the honest one


def _ensure_dirs() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Load predictions + ground truth
# ═════════════════════════════════════════════════════════════════════════════
def _extract_label_id(stem: str) -> str:
    m = re.match(r"^(L\d+)", str(stem))
    return m.group(1) if m else str(stem)


def load_predictions() -> pd.DataFrame:
    """Load image_summary.csv and make sure every row has a usable true_label,
    re-merging from lesion_detect.csv if it was left blank."""
    if not os.path.exists(IMAGE_SUMMARY_CSV):
        raise FileNotFoundError(
            f"{IMAGE_SUMMARY_CSV} not found. Run lesion_detection.py first.")
    df = pd.read_csv(IMAGE_SUMMARY_CSV)
    df = df[df["status"] == "ok"].copy()

    if os.path.exists(LABELS_CSV):
        gt = pd.read_csv(LABELS_CSV).dropna(subset=["Label", "Lesion_Detection"])
        gt_map = {str(r["Label"]).strip(): int(r["Lesion_Detection"]) for _, r in gt.iterrows()}
        missing = df["true_label"].isna() | (df["true_label"] == "")
        df.loc[missing, "true_label"] = df.loc[missing, "stem"].map(
            lambda s: gt_map.get(_extract_label_id(s), np.nan))

    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
    for col in ("rf_prob", "svm_prob", "ensemble_prob"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _subset(df: pd.DataFrame, split: str) -> pd.DataFrame:
    labeled = df.dropna(subset=["true_label"])
    if split == "all_labeled":
        return labeled
    return labeled[labeled["split"] == split]


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════
def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def compute_metrics(y_true: np.ndarray, prob: np.ndarray,
                    threshold: float = DECISION_THRESHOLD) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": _specificity(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, prob)
        metrics["roc_auc"] = auc(fpr, tpr)
        metrics["avg_precision"] = average_precision_score(y_true, prob)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["avg_precision"] = float("nan")
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = (
        int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Plots
# ═════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true: np.ndarray, prob: np.ndarray, model: str,
                          split: str, threshold: float = DECISION_THRESHOLD) -> None:
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No lesion", "Lesion"])
    ax.set_yticklabels(["No lesion", "Lesion"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{MODEL_LABELS[model]} — {split}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, f"confusion_matrix_{model}_{split}.png"), dpi=150)
    plt.close(fig)


def plot_roc_curves(data: Dict[str, Tuple[np.ndarray, np.ndarray]], split: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for model, (y_true, prob) in data.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[model]} (AUC={auc(fpr, tpr):.3f})",
               color=MODEL_COLOR[model])
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve — {split}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, f"roc_curve_{split}.png"), dpi=150)
    plt.close(fig)


def plot_pr_curves(data: Dict[str, Tuple[np.ndarray, np.ndarray]], split: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for model, (y_true, prob) in data.items():
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ax.plot(rec, prec, label=f"{MODEL_LABELS[model]} (AP={average_precision_score(y_true, prob):.3f})",
               color=MODEL_COLOR[model])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve — {split}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, f"pr_curve_{split}.png"), dpi=150)
    plt.close(fig)


def plot_probability_distribution(y_true: np.ndarray, prob: np.ndarray,
                                  model: str, split: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(prob[y_true == 0], bins=bins, alpha=0.6, label="No lesion (true)", color="#e03131")
    ax.hist(prob[y_true == 1], bins=bins, alpha=0.6, label="Lesion (true)", color="#2b8a3e")
    ax.axvline(DECISION_THRESHOLD, color="black", linestyle="--", linewidth=1,
              label=f"threshold={DECISION_THRESHOLD}")
    ax.set_xlabel(f"{MODEL_LABELS[model]} predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{MODEL_LABELS[model]} probability distribution — {split}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, f"probability_distribution_{model}_{split}.png"), dpi=150)
    plt.close(fig)


def threshold_sweep(y_true: np.ndarray, prob: np.ndarray, model: str, split: str) -> pd.DataFrame:
    rows = []
    for thr in THRESHOLD_SWEEP:
        m = compute_metrics(y_true, prob, threshold=thr)
        rows.append({"threshold": thr, "sensitivity": m["recall_sensitivity"],
                    "specificity": m["specificity"], "precision": m["precision"],
                    "f1": m["f1"], "accuracy": m["accuracy"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EVAL_DIR, f"threshold_sweep_{model}_{split}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, color in [("sensitivity", "#2b8a3e"), ("specificity", "#1971c2"),
                       ("precision", "#e8590c"), ("f1", "#862e9c")]:
        ax.plot(df["threshold"], df[col], marker="o", label=col, color=color)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title(f"{MODEL_LABELS[model]} threshold sweep — {split}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, f"threshold_sweep_{model}_{split}.png"), dpi=150)
    plt.close(fig)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════════════
def _format_metrics_line(model: str, m: Dict[str, float]) -> str:
    return (f"  {MODEL_LABELS[model]:14s} n={m['n']:3d}  "
           f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
           f"sens={m['recall_sensitivity']:.3f}  spec={m['specificity']:.3f}  "
           f"f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}  ap={m['avg_precision']:.3f}  "
           f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")


def evaluate_split(df: pd.DataFrame, split: str, lines: List[str],
                   csv_rows: List[dict]) -> None:
    sub = _subset(df, split)
    if sub.empty:
        lines.append(f"\n  [{split}] no labeled rows — skipped.")
        return
    y_true = sub["true_label"].to_numpy().astype(int)
    lines.append(f"\n  [{split}]  (n={len(sub)}, "
                 f"{int((y_true == 1).sum())} lesion / {int((y_true == 0).sum())} no-lesion)")

    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        prob = sub[PROB_COL[model]].to_numpy()
        m = compute_metrics(y_true, prob)
        lines.append(_format_metrics_line(model, m))
        csv_rows.append({"split": split, "model": model, **m})
        roc_data[model] = (y_true, prob)

        plot_confusion_matrix(y_true, prob, model, split)
        plot_probability_distribution(y_true, prob, model, split)
        threshold_sweep(y_true, prob, model, split)

    plot_roc_curves(roc_data, split)
    plot_pr_curves(roc_data, split)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 60)
    print("  Lesion DETECTION evaluation (RF / SVM / Ensemble)")
    print("=" * 60)
    _ensure_dirs()

    df = load_predictions()
    print(f"  Loaded {len(df)} predictions from {IMAGE_SUMMARY_CSV}")
    n_labeled = df["true_label"].notna().sum()
    print(f"  Rows with a usable ground-truth label: {n_labeled}")
    if n_labeled == 0:
        print("[ERROR] no labeled rows to evaluate against.")
        return

    lines: List[str] = ["=" * 60, "  LESION DETECTION EVALUATION", "=" * 60]
    csv_rows: List[dict] = []
    for split in SPLITS_TO_REPORT:
        evaluate_split(df, split, lines, csv_rows)

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(EVAL_DIR, "metrics_report.txt"), "w") as fh:
        fh.write(report + "\n")
    pd.DataFrame(csv_rows).to_csv(os.path.join(EVAL_DIR, "metrics_per_model.csv"), index=False)

    print(f"\n  Evaluation outputs -> {EVAL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
