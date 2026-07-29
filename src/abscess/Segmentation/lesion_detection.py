"""
lesion_detection.py
───────────────────
A segmentation-driven, anatomy-aware and EXPLAINABLE periapical-lesion DETECTION
framework. It converts the unsupervised Fuzzy-C-Means (FCM) lesion candidates in
``lesion_segmented/`` into per-image features (5 handcrafted families), then
trains SVM and Random Forest classifiers on those features against the
image-level ground truth in ``lesion_detect.csv`` to decide lesion / no-lesion.

Feature families (per candidate, same as before):
    1. Tooth-relative spatial  (apex distance, tooth distance, root-axis position)
    2. Radiographic            (darkness, contrast vs surrounding bone)
    3. Texture                 (GLCM contrast/homogeneity/energy/correlation,
                                LBP, entropy)
    4. Shape                   (area, circularity, solidity, eccentricity, extent)
    5. Fuzzy membership        (per-candidate FCM lesion-cluster membership)

The single most lesion-like candidate per image (ranked by a transparent
handcrafted rule score, kept as one more ML feature: ``rule_score``) supplies the
per-image feature vector, augmented with simple aggregates across all candidates
(``n_candidates``, ``mean_rule_score``). This vector is what SVM and Random
Forest are trained/predict on.

Explainability: for the LINEAR SVM, each feature's contribution to the decision
is EXACT (coef_i × scaled_feature_i, summing to the decision function). For the
Random Forest, contribution is an approximation (feature_importances_i ×
z-scored feature_i) — the largest contributions are printed and drawn as bars in
the explain/ panel, so every detection states *why* it fired.

Inputs
    Crops        : ../../../data/abscess/raw/croun_crops/<stem>.png
    Tooth masks  : ../../../data/abscess/raw/rf_tooth_output/tooth_mask/tooth_mask_<stem>.png
    FCM candidates: ../../../data/abscess/raw/lesion_segmented/lesion_mask_<stem>.png
    Labels (GT)  : ../../../src/abscess/training/lesion_detect.csv
                   columns: Label (e.g. "L1"), Lesion_Detection (0/1),
                   lesion_Diameter, lesion_orientation
    Pixel GT     : ../../../data/abscess/raw/mask/<stem>_mask*.png  (optional, for
                   a secondary pixel-overlap sanity check only)
Outputs (../../../data/abscess/raw/lesion_detection_output/)
    models/rf_detector.joblib, svm_detector.joblib, scaler.joblib
    detection/detection_<stem>.png     candidates coloured by rule score, best boxed
    explain/explain_<stem>.png         RF + SVM top contributing features
    candidates.csv                     every FCM candidate: features + rule score
    image_summary.csv                  per image: rf/svm/ensemble prob, detected, split, true label
    detection_metrics.txt              accuracy / precision / recall / F1 (train & test)

Usage:
    python lesion_detection.py             # train + predict + evaluate
    python lesion_detection.py --predict   # load saved models, predict + explain only
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import skfuzzy as fuzz

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw")
INPUT_DIR = os.path.join(_RAW, "croun_crops")
TOOTH_MASK_DIR = os.path.join(_RAW, "rf_tooth_output", "tooth_mask")
FCM_DIR = os.path.join(_RAW, "lesion_output", "lesion_mask")   # FCM/RF candidate masks
GT_DIR = os.path.join(_RAW, "mask")                       # optional pixel GT
LABELS_CSV = os.path.join(SCRIPT_DIR, "..", "training", "lesion_detect.csv")
OUTPUT_DIR = os.path.join(_RAW, "lesion_detection_output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_detector.joblib")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_detector.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")
RANDOM_SEED = 42

# ── FCM (to reproduce the lesion-cluster membership map) ─────────────────────
FCM_CLUSTERS = 3
FCM_M = 2.0
FCM_ERROR = 0.005
FCM_MAXITER = 1000
FCM_INTENSITY_WEIGHT = 1.0
FCM_SPATIAL_WEIGHT = 0.30

# ── Candidate filtering ──────────────────────────────────────────────────────
MIN_CAND_AREA_FRAC = 0.0015        # ignore FCM specks smaller than this * frame

# ── Feature params ───────────────────────────────────────────────────────────
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
BILATERAL_D, BILATERAL_SIGMA = 5, 40.0
RING_PX = 12                       # surrounding-bone ring for contrast
GLCM_LEVELS = 32
LBP_P, LBP_R = 8, 1
APEX_END_FRAC = 0.15

# ── Normalisers (map a raw feature to a 0-1 sub-score for the rule score) ─────
CONTRAST_NORM = 0.32
GLCM_CONTRAST_NORM = 3.0
ENTROPY_NORM = 8.0
PREF_AREA_FRAC = (0.006, 0.20)
MIN_AREA_FRAC, MAX_AREA_FRAC = 0.0015, 0.5
CIRC_RANGE = (0.15, 1.2)
MIN_SOLIDITY = 0.5

# Rule score used to (a) RANK/select the representative candidate per image and
# (b) feed the ML models as one extra engineered feature — not the final
# lesion/no-lesion decision anymore (that comes from the trained classifiers).
RULE_SCORE_WEIGHTS: Dict[str, float] = {
    "spatial":  0.20, "contrast": 0.24, "darkness": 0.12, "fuzzy": 0.16,
    "texture":  0.16, "shape":    0.06, "size":     0.06,
}

# ── Machine-learning feature vector (per image) ──────────────────────────────
# Scale-invariant subset of the candidate features + simple aggregates.
ML_FEATURE_KEYS: List[str] = [
    "area_frac", "apex_dist_norm", "apical_proj", "dist_to_tooth_norm",
    "inside_tooth_frac", "mean_intensity", "darkness", "std_intensity",
    "contrast_vs_bone", "glcm_contrast", "glcm_homogeneity", "glcm_energy",
    "glcm_correlation", "lbp_mean", "lbp_std", "entropy", "circularity",
    "solidity", "extent", "aspect_ratio", "eccentricity", "fuzzy_mean",
    "fuzzy_max", "fuzzy_std", "rule_score", "n_candidates", "mean_rule_score",
]

# ── Random Forest ────────────────────────────────────────────────────────────
RF_ESTIMATORS = 300
RF_MAX_DEPTH = 6            # shallow — dataset is small (~80 images), avoid overfit
RF_MIN_LEAF = 2

# ── SVM ───────────────────────────────────────────────────────────────────────
SVM_KERNEL = "linear"       # linear → exact per-feature contribution for "reason"
SVM_C = 1.0

# ── Train/test split + decision ──────────────────────────────────────────────
TEST_FRAC = 0.25
DETECT_PROB_THRESHOLD = 0.5     # ensemble (mean of RF+SVM prob) >= this → detected

# ── Optional secondary pixel-overlap sanity check ────────────────────────────
RUN_PIXEL_CHECK = True
HIT_IOU = 0.10

SUBDIRS = {"detection": "detection", "explain": "explain"}
_EPS = 1e-8


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _safe_div(a: float, b: float) -> float:
    return float(a) / (float(b) + _EPS)


def _odd(n: int) -> int:
    n = int(round(n))
    return max(1, n + 1 if n % 2 == 0 else n)


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))


def _ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for sub in SUBDIRS.values():
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def _find(mask_dir: str, stem: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not os.path.isdir(mask_dir):
        return None
    cand = [m for m in os.listdir(mask_dir)
            if m.lower().endswith(SUPPORTED_EXTS)
            and m.startswith((f"{stem}_mask", f"tooth_mask_{stem}", f"lesion_mask_{stem}"))]
    if not cand:
        return None
    m = cv2.imread(os.path.join(mask_dir, cand[0]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def load_gray_valid(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        gray, valid = img, np.ones(img.shape, bool)
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        valid = img[:, :, 3] > 0
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = np.ones(img.shape[:2], bool)
    return gray.astype(np.uint8), valid


def _extract_label_id(stem: str) -> Optional[str]:
    """'L10_clahe_sigmoid' -> 'L10' (matches the lesion_detect.csv Label column)."""
    m = re.match(r"^(L\d+)", stem)
    return m.group(1) if m else None


def load_labels(csv_path: str = LABELS_CSV) -> Dict[str, int]:
    """Load {image_id ('L1', ...): 0/1} from lesion_detect.csv, dropping blanks."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Label", "Lesion_Detection"])
    return {str(row["Label"]).strip(): int(row["Lesion_Detection"])
            for _, row in df.iterrows()}


# ═════════════════════════════════════════════════════════════════════════════
# Fuzzy C-Means lesion-membership map (reproduces the FCM segmentation)
# ═════════════════════════════════════════════════════════════════════════════
def fcm_membership(enh_f: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Spatial FCM on [intensity, x, y]; per-pixel membership to the darkest
    (lesion) cluster. Background = 0. Seeded → reproducible."""
    h, w = enh_f.shape
    fg = np.flatnonzero(valid.reshape(-1))
    if fg.size < FCM_CLUSTERS * 10:
        return np.zeros((h, w), np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    feat = np.vstack([
        enh_f.reshape(-1)[fg] * FCM_INTENSITY_WEIGHT,
        (xx.reshape(-1)[fg] / max(w - 1, 1)) * FCM_SPATIAL_WEIGHT,
        (yy.reshape(-1)[fg] / max(h - 1, 1)) * FCM_SPATIAL_WEIGHT,
    ])
    cntr, u, *_ = fuzz.cluster.cmeans(feat, FCM_CLUSTERS, FCM_M, error=FCM_ERROR,
                                      maxiter=FCM_MAXITER, init=None, seed=RANDOM_SEED)
    lesion_idx = int(np.argmin(cntr[:, 0]))
    mem = np.zeros(h * w, np.float64)
    mem[fg] = u[lesion_idx]
    return mem.reshape(h, w)


# ═════════════════════════════════════════════════════════════════════════════
# Tooth anatomy (PCA axis + apex)
# ═════════════════════════════════════════════════════════════════════════════
def tooth_anatomy(tooth: Optional[np.ndarray]) -> Optional[dict]:
    if tooth is None:
        return None
    ys, xs = np.nonzero(tooth)
    if ys.size < 20:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centroid = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - centroid).T))
    major = evecs[:, int(np.argmax(evals))]
    minor = evecs[:, int(np.argmin(evals))]
    t = (pts - centroid) @ major
    s = (pts - centroid) @ minor
    t_min, t_max = float(t.min()), float(t.max())
    root_len = max(t_max - t_min, 1.0)
    root_width = max(float(np.ptp(s)), 1.0)
    end = APEX_END_FRAC * root_len
    w_max = float(np.ptp(s[t > (t_max - end)])) if (t > (t_max - end)).any() else np.inf
    w_min = float(np.ptp(s[t < (t_min + end)])) if (t < (t_min + end)).any() else np.inf
    t_apex = t_max if w_max <= w_min else t_min
    apex = centroid + t_apex * major
    direction = major * (1.0 if t_apex >= 0 else -1.0)
    return {"centroid": centroid, "direction": direction, "apex": apex,
            "root_len": root_len, "root_width": root_width}


# ═════════════════════════════════════════════════════════════════════════════
# Candidate feature extraction (5 families)
# ═════════════════════════════════════════════════════════════════════════════
def candidate_features(comp: np.ndarray, enh_f: np.ndarray, enh_u8: np.ndarray,
                       valid: np.ndarray, tooth: Optional[np.ndarray],
                       anat: Optional[dict], membership: np.ndarray,
                       shape: Tuple[int, int]) -> Dict[str, float]:
    h, w = shape
    diag = math.hypot(h, w)
    ys, xs = np.nonzero(comp)
    area = int(comp.sum())
    cx, cy = float(xs.mean()), float(ys.mean())
    f: Dict[str, float] = {"centroid_x": cx, "centroid_y": cy, "area": float(area)}

    # ── 1. Tooth-relative spatial ──
    if anat is not None:
        ax, ay = anat["apex"]
        f["apex_dist_norm"] = _clip01(math.hypot(cx - ax, cy - ay) / diag)
        d = anat["direction"]
        f["apical_proj"] = ((cx - ax) * d[0] + (cy - ay) * d[1]) / anat["root_len"]
    else:
        f["apex_dist_norm"] = 0.5
        f["apical_proj"] = 0.0
    if tooth is not None:
        dt = cv2.distanceTransform((tooth == 0).astype(np.uint8), cv2.DIST_L2, 5)
        f["dist_to_tooth_norm"] = _clip01(float(dt[comp].min()) / diag)
        f["inside_tooth_frac"] = _safe_div(int((comp & (tooth > 0)).sum()), area)
    else:
        f["dist_to_tooth_norm"] = 0.5
        f["inside_tooth_frac"] = 0.0

    # ── 2. Radiographic ──
    vals = enh_f[comp]
    f["mean_intensity"] = float(vals.mean())
    f["min_intensity"] = float(vals.min())
    f["max_intensity"] = float(vals.max())
    f["std_intensity"] = float(vals.std())
    f["darkness"] = 1.0 - f["mean_intensity"]
    ring = (cv2.dilate(comp.astype(np.uint8), _ellipse(RING_PX)) > 0) & (~comp) & valid
    ring_mean = float(enh_f[ring].mean()) if ring.sum() else f["mean_intensity"]
    f["contrast_vs_bone"] = ring_mean - f["mean_intensity"]

    # ── 3. Texture ──
    f.update(_glcm_lbp_entropy(comp, enh_u8, ys, xs))

    # ── 4. Shape ──
    f.update(_shape_features(comp, area))

    # ── 5. Fuzzy membership ──
    mem = membership[comp]
    f["fuzzy_mean"] = float(mem.mean()) if mem.size else 0.0
    f["fuzzy_max"] = float(mem.max()) if mem.size else 0.0
    f["fuzzy_std"] = float(mem.std()) if mem.size else 0.0
    return f


def _glcm_lbp_entropy(comp: np.ndarray, enh_u8: np.ndarray,
                      ys: np.ndarray, xs: np.ndarray) -> Dict[str, float]:
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = enh_u8[y0:y1, x0:x1]
    q = np.clip(patch.astype(np.float64) / 256.0 * GLCM_LEVELS, 0, GLCM_LEVELS - 1).astype(np.uint8)
    out = {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0,
           "glcm_correlation": 0.0}
    try:
        glcm = graycomatrix(q, [1], [0, np.pi / 2], levels=GLCM_LEVELS,
                            symmetric=True, normed=True)
        for p in ("contrast", "homogeneity", "energy", "correlation"):
            out[f"glcm_{p}"] = float(graycoprops(glcm, p).mean())
    except Exception:
        pass
    lbp = local_binary_pattern(patch, LBP_P, LBP_R, method="uniform")
    region_lbp = lbp[comp[y0:y1, x0:x1]]
    out["lbp_mean"] = float(region_lbp.mean()) if region_lbp.size else 0.0
    out["lbp_std"] = float(region_lbp.std()) if region_lbp.size else 0.0
    out["entropy"] = float(shannon_entropy(enh_u8[comp])) if comp.any() else 0.0
    return out


def _shape_features(comp: np.ndarray, area: int) -> Dict[str, float]:
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea) if cnts else None
    perim = cv2.arcLength(c, True) if c is not None else 1.0
    x, y, bw, bh = cv2.boundingRect(c) if c is not None else (0, 0, 1, 1)
    hull = cv2.contourArea(cv2.convexHull(c)) if c is not None else area
    if c is not None and len(c) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        major, minor = max(MA, ma), min(MA, ma)
        ecc = math.sqrt(1.0 - (minor / (major + _EPS)) ** 2)
    else:
        ecc = 0.0
    return {
        "perimeter": float(perim),
        "equiv_diameter": math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0,
        "circularity": _clip01(_safe_div(4 * math.pi * area, perim * perim)),
        "solidity": _clip01(_safe_div(area, hull)),
        "extent": _clip01(_safe_div(area, bw * bh)),
        "aspect_ratio": _safe_div(bw, bh),
        "eccentricity": float(ecc),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Transparent handcrafted rule score (kept as a ranking tool + one ML feature)
# ═════════════════════════════════════════════════════════════════════════════
def _area_score(area_frac: float) -> float:
    lo, hi = PREF_AREA_FRAC
    if lo <= area_frac <= hi:
        return 1.0
    if area_frac < lo:
        return _clip01(_safe_div(area_frac - MIN_AREA_FRAC, lo - MIN_AREA_FRAC))
    return _clip01(_safe_div(MAX_AREA_FRAC - area_frac, MAX_AREA_FRAC - hi))


def _shape_score(f: Dict[str, float]) -> float:
    lo, hi = CIRC_RANGE
    circ = 1.0 if lo <= f["circularity"] <= hi else _clip01(
        1.0 - min(abs(f["circularity"] - lo), abs(f["circularity"] - hi)))
    sol = _clip01(_safe_div(f["solidity"] - MIN_SOLIDITY, 1.0 - MIN_SOLIDITY))
    return 0.5 * circ + 0.5 * sol


def rule_score(f: Dict[str, float], img_area: int) -> float:
    """Transparent weighted heuristic used to RANK candidates (not the final
    lesion/no-lesion call — that's the trained classifiers' job)."""
    area_frac = _safe_div(f["area"], img_area)
    apical_bonus = _clip01(0.5 + 0.5 * math.tanh(3.0 * f["apical_proj"]))
    w = RULE_SCORE_WEIGHTS
    sub = {
        "spatial": _clip01(0.55 * (1.0 - f["apex_dist_norm"] * 4.0)
                           + 0.25 * (1.0 - f["dist_to_tooth_norm"] * 6.0)
                           + 0.20 * apical_bonus),
        "contrast": _clip01(_safe_div(f["contrast_vs_bone"], CONTRAST_NORM)),
        "darkness": _clip01(f["darkness"]),
        "fuzzy": _clip01(f["fuzzy_mean"]),
        "texture": _clip01(0.5 * (1.0 - _safe_div(f["glcm_contrast"], GLCM_CONTRAST_NORM))
                           + 0.5 * f["glcm_homogeneity"]),
        "shape": _shape_score(f),
        "size": _area_score(area_frac),
    }
    return _clip01(sum(w[k] * v for k, v in sub.items()))


# ═════════════════════════════════════════════════════════════════════════════
# Per-image ML feature vector (best candidate + aggregates)
# ═════════════════════════════════════════════════════════════════════════════
def build_image_features(candidates: List[dict], img_area: int) -> Dict[str, float]:
    """Build the fixed-length ML feature vector for one image from its FCM
    candidates. Uses the top-ranked (by rule_score) candidate's own features
    plus simple aggregates across all candidates. An image with zero candidates
    gets an all-neutral (zero) vector — a valid "nothing suspicious" example."""
    if not candidates:
        vec = dict.fromkeys(ML_FEATURE_KEYS, 0.0)
        return vec

    scores = [c["rule_score"] for c in candidates]
    best = candidates[int(np.argmax(scores))]
    f = best["features"]

    vec = {
        "area_frac": _safe_div(f["area"], img_area),
        "apex_dist_norm": f["apex_dist_norm"],
        "apical_proj": f["apical_proj"],
        "dist_to_tooth_norm": f["dist_to_tooth_norm"],
        "inside_tooth_frac": f["inside_tooth_frac"],
        "mean_intensity": f["mean_intensity"],
        "darkness": f["darkness"],
        "std_intensity": f["std_intensity"],
        "contrast_vs_bone": f["contrast_vs_bone"],
        "glcm_contrast": f["glcm_contrast"],
        "glcm_homogeneity": f["glcm_homogeneity"],
        "glcm_energy": f["glcm_energy"],
        "glcm_correlation": f["glcm_correlation"],
        "lbp_mean": f["lbp_mean"],
        "lbp_std": f["lbp_std"],
        "entropy": f["entropy"],
        "circularity": f["circularity"],
        "solidity": f["solidity"],
        "extent": f["extent"],
        "aspect_ratio": f["aspect_ratio"],
        "eccentricity": f["eccentricity"],
        "fuzzy_mean": f["fuzzy_mean"],
        "fuzzy_max": f["fuzzy_max"],
        "fuzzy_std": f["fuzzy_std"],
        "rule_score": best["rule_score"],
        "n_candidates": float(len(candidates)),
        "mean_rule_score": float(np.mean(scores)),
    }
    for k, v in vec.items():
        if not np.isfinite(v):
            vec[k] = 0.0
    return vec


# ═════════════════════════════════════════════════════════════════════════════
# Per-image candidate extraction (FCM components + features + rule score)
# ═════════════════════════════════════════════════════════════════════════════
def extract_candidates(stem: str) -> Tuple[Optional[np.ndarray], List[dict], Optional[Tuple[int, int]]]:
    """Load the crop, run FCM membership, and describe every dark candidate.
    Returns (gray, candidates, shape) — gray is None if the crop is unreadable."""
    loaded = load_gray_valid(os.path.join(INPUT_DIR, f"{stem}.png"))
    if loaded is None:
        return None, [], None
    gray, valid = loaded
    h, w = gray.shape

    fcm = _find(FCM_DIR, stem, (h, w))
    if fcm is None or fcm.sum() == 0:
        return gray, [], (h, w)

    tooth = _find(TOOTH_MASK_DIR, stem, (h, w))
    anat = tooth_anatomy(tooth)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enh_u8 = cv2.bilateralFilter(clahe.apply(gray), BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
    enh_u8[~valid] = 0
    enh_f = enh_u8.astype(np.float64) / 255.0
    membership = fcm_membership(enh_f, valid)

    n, lab = cv2.connectedComponents(fcm, 8)
    img_area = h * w
    candidates: List[dict] = []
    for cid in range(1, n):
        comp = lab == cid
        if comp.sum() < MIN_CAND_AREA_FRAC * img_area:
            continue
        f = candidate_features(comp, enh_f, enh_u8, valid, tooth, anat, membership, (h, w))
        candidates.append({"id": cid, "mask": comp, "features": f,
                           "rule_score": rule_score(f, img_area)})
    return gray, candidates, (h, w)


# ═════════════════════════════════════════════════════════════════════════════
# Model training + prediction + explanation
# ═════════════════════════════════════════════════════════════════════════════
def train_classifiers(X: np.ndarray, y: np.ndarray
                      ) -> Tuple[RandomForestClassifier, SVC, StandardScaler,
                                np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split, scale, train RF + linear SVM. Returns
    (rf, svm, scaler, X_train, X_test, y_train, y_test)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                min_samples_leaf=RF_MIN_LEAF, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    svm = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True,
             class_weight="balanced", random_state=RANDOM_SEED)
    svm.fit(X_train_s, y_train)

    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(svm, SVM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return rf, svm, scaler, X_train_s, X_test_s, y_train, y_test


def load_classifiers() -> Tuple[RandomForestClassifier, SVC, StandardScaler]:
    return (joblib.load(RF_MODEL_PATH), joblib.load(SVM_MODEL_PATH),
            joblib.load(SCALER_PATH))


def predict_and_explain(vec: Dict[str, float], rf: RandomForestClassifier,
                        svm: SVC, scaler: StandardScaler
                        ) -> Dict[str, object]:
    """Predict lesion probability with both models and build the explanation:
    exact per-feature contribution for the linear SVM, approximate contribution
    (importance × z-score) for the Random Forest."""
    x = np.array([[vec[k] for k in ML_FEATURE_KEYS]], dtype=np.float64)
    x_s = scaler.transform(x)[0]

    rf_prob = float(rf.predict_proba(x_s.reshape(1, -1))[0, 1])
    svm_prob = float(svm.predict_proba(x_s.reshape(1, -1))[0, 1])
    ensemble_prob = 0.5 * (rf_prob + svm_prob)
    detected = ensemble_prob >= DETECT_PROB_THRESHOLD

    rf_contrib = dict(zip(ML_FEATURE_KEYS, rf.feature_importances_ * x_s))
    if SVM_KERNEL == "linear":
        svm_contrib = dict(zip(ML_FEATURE_KEYS, svm.coef_[0] * x_s))
    else:
        svm_contrib = {}   # no exact decomposition for non-linear kernels

    return {"rf_prob": rf_prob, "svm_prob": svm_prob, "ensemble_prob": ensemble_prob,
            "detected": detected, "rf_contrib": rf_contrib, "svm_contrib": svm_contrib}


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════════════
def _save_detection(stem: str, gray: np.ndarray, candidates: List[dict],
                    best_id: Optional[int], detected: bool) -> None:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in candidates:
        s = c["rule_score"]
        color = (0, int(255 * s), int(255 * (1 - s)))
        cnts, _ = cv2.findContours(c["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 1)
    if detected and best_id is not None:
        best = next(c for c in candidates if c["id"] == best_id)
        x, y, bw, bh = cv2.boundingRect(best["mask"].astype(np.uint8))
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, SUBDIRS["detection"], f"detection_{stem}.png"), vis)


def _save_explanation(stem: str, gray: np.ndarray, best_mask: Optional[np.ndarray],
                      result: Dict[str, object]) -> None:
    """Panel: the detection + the top RF and SVM contributing features (why)."""
    h, w = gray.shape
    panel_w = 300
    canvas = np.zeros((max(h, 260), w + panel_w, 3), np.uint8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if best_mask is not None:
        tint = vis.copy(); tint[best_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
        cnts, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    canvas[:h, :w] = vis

    def _panel(title, contrib, y0):
        cv2.putText(canvas, title, (w + 8, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        if not contrib:
            cv2.putText(canvas, "(no exact contributions available)", (w + 8, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
            return y0 + 40
        items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        max_abs = max(abs(v) for _, v in items) + _EPS
        yy = y0 + 22
        for name, val in items:
            color = (0, 200, 0) if val >= 0 else (0, 0, 220)
            cv2.putText(canvas, f"{name:18s} {val:+.2f}", (w + 8, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 255), 1, cv2.LINE_AA)
            bar = int(90 * min(abs(val) / max_abs, 1.0))
            bx = w + 190
            cv2.rectangle(canvas, (bx, yy - 8), (bx + bar, yy - 2), color, -1)
            yy += 20
        return yy + 10

    y = 20
    cv2.putText(canvas, f"ensemble prob = {result['ensemble_prob']:.2f}  "
                        f"({'DETECTED' if result['detected'] else 'no lesion'})",
                (w + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
    y += 26
    y = _panel(f"Random Forest (p={result['rf_prob']:.2f}) - approx.", result["rf_contrib"], y)
    _panel(f"SVM (p={result['svm_prob']:.2f}) - exact (linear)", result["svm_contrib"], y)
    cv2.imwrite(os.path.join(OUTPUT_DIR, SUBDIRS["explain"], f"explain_{stem}.png"), canvas)


# ═════════════════════════════════════════════════════════════════════════════
# Optional secondary pixel-overlap sanity check (GT masks, when present)
# ═════════════════════════════════════════════════════════════════════════════
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.sum(a & b)); union = int(np.sum(a | b))
    return _safe_div(inter, union)


def pixel_overlap_check(records: List[dict]) -> None:
    """For images with a pixel GT mask, check whether the DETECTED region
    (if any) actually overlaps it. Secondary diagnostic only — the primary
    ground truth for this script is lesion_detect.csv."""
    n_gt = hits = 0
    for r in records:
        if r["gray_shape"] is None:
            continue
        gt = _find(GT_DIR, r["stem"], r["gray_shape"])
        if gt is None:
            continue
        n_gt += 1
        if r["detected"] and r["best_mask"] is not None and _iou(r["best_mask"], gt > 0) >= HIT_IOU:
            hits += 1
    if n_gt:
        print(f"\n  [secondary] pixel-overlap check on {n_gt} image(s) with a GT "
              f"mask: {hits}/{n_gt} detections overlap it (IoU ≥ {HIT_IOU}).")


# ═════════════════════════════════════════════════════════════════════════════
# Reports
# ═════════════════════════════════════════════════════════════════════════════
def _safe_write_csv(path: str, fields: List[str], rows: List[dict]) -> None:
    try:
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader(); wr.writerows(rows)
    except PermissionError:
        print(f"  [WARN] could not write {os.path.basename(path)} "
              f"(file is open elsewhere?) — skipped.")


def _write_candidates_csv(all_candidates: Dict[str, List[dict]]) -> None:
    rows, feat_keys = [], None
    for stem, cands in all_candidates.items():
        for c in cands:
            if feat_keys is None:
                feat_keys = list(c["features"].keys())
            row = {"stem": stem, "candidate_id": c["id"], "rule_score": c["rule_score"]}
            row.update(c["features"])
            rows.append(row)
    if rows:
        fields = ["stem", "candidate_id", "rule_score"] + (feat_keys or [])
        _safe_write_csv(os.path.join(OUTPUT_DIR, "candidates.csv"), fields, rows)


def _write_metrics(y_train, rf_train_pred, svm_train_pred, ens_train_pred,
                   y_test, rf_test_pred, svm_test_pred, ens_test_pred) -> None:
    def block(name, y_true, y_pred):
        return {
            "model": name,
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    lines = ["=" * 60, "  TRAIN split", "=" * 60]
    for name, pred in [("RandomForest", rf_train_pred), ("SVM", svm_train_pred),
                       ("Ensemble", ens_train_pred)]:
        b = block(name, y_train, pred)
        lines.append(f"  {b['model']:14s} acc={b['accuracy']:.3f} prec={b['precision']:.3f} "
                     f"rec={b['recall']:.3f} f1={b['f1']:.3f}")
    lines += ["", "=" * 60, "  TEST split (held out — trustworthy)", "=" * 60]
    for name, pred in [("RandomForest", rf_test_pred), ("SVM", svm_test_pred),
                       ("Ensemble", ens_test_pred)]:
        b = block(name, y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        lines.append(f"  {b['model']:14s} acc={b['accuracy']:.3f} prec={b['precision']:.3f} "
                     f"rec={b['recall']:.3f} f1={b['f1']:.3f}  confusion={cm.tolist()}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(OUTPUT_DIR, "detection_metrics.txt"), "w") as fh:
        fh.write(text + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    predict_only = "--predict" in sys.argv

    print("=" * 60)
    print("  Periapical lesion DETECTION — SVM + Random Forest")
    print("=" * 60)
    print(f"  Crops        : {INPUT_DIR}")
    print(f"  Tooth masks  : {TOOTH_MASK_DIR}")
    print(f"  FCM candidates: {FCM_DIR}")
    print(f"  Labels (CSV) : {LABELS_CSV}")
    print("=" * 60)

    for d, name in [(INPUT_DIR, "crops"), (FCM_DIR, "FCM candidates")]:
        if not os.path.isdir(d):
            print(f"[ERROR] {name} directory not found: {d}")
            return
    _ensure_dirs()

    labels = load_labels()
    if not labels and not predict_only:
        print(f"[ERROR] no usable labels in {LABELS_CSV}")
        return
    print(f"  Labeled images in CSV: {len(labels)}")

    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(INPUT_DIR))
             if f.lower().endswith(SUPPORTED_EXTS)]
    print(f"  Crops to process     : {len(stems)}\n")

    # ── Pass 1: extract candidates + per-image feature vector for every crop ──
    all_candidates: Dict[str, List[dict]] = {}
    image_vecs: Dict[str, Dict[str, float]] = {}
    gray_cache: Dict[str, Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]] = {}
    for stem in stems:
        try:
            gray, cands, shape = extract_candidates(stem)
        except Exception as exc:
            print(f"  [ERROR] {stem}: {type(exc).__name__}: {exc}")
            gray, cands, shape = None, [], None
        all_candidates[stem] = cands
        gray_cache[stem] = (gray, shape)
        img_area = (shape[0] * shape[1]) if shape else 1
        image_vecs[stem] = build_image_features(cands, img_area)
        print(f"  {stem}: candidates={len(cands)}")

    _write_candidates_csv(all_candidates)

    # ── Build the labeled training table ─────────────────────────────────────
    label_rows = []
    for stem in stems:
        lid = _extract_label_id(stem)
        if lid in labels:
            label_rows.append((stem, labels[lid]))
    print(f"\n  Crops matched to a label: {len(label_rows)}")

    # ── Train (or load) the classifiers ──────────────────────────────────────
    if predict_only and all(os.path.exists(p) for p in
                            (RF_MODEL_PATH, SVM_MODEL_PATH, SCALER_PATH)):
        rf, svm, scaler = load_classifiers()
        print(f"  Loaded models from {MODEL_DIR}")
        # Recompute (not retrain) the same deterministic stratified split so
        # image_summary.csv still tags each row train/test correctly.
        if label_rows:
            stems_only = [s for s, _ in label_rows]
            y_all = np.array([lab for _, lab in label_rows])
            idx_train, idx_test = train_test_split(
                np.arange(len(label_rows)), test_size=TEST_FRAC,
                random_state=RANDOM_SEED, stratify=y_all)
            train_stems = {stems_only[i] for i in idx_train}
            test_stems = {stems_only[i] for i in idx_test}
        else:
            train_stems, test_stems = set(), set()
    else:
        if len(label_rows) < 10:
            print("[ERROR] not enough labeled images to train (<10).")
            return
        X = np.array([[image_vecs[s][k] for k in ML_FEATURE_KEYS] for s, _ in label_rows])
        y = np.array([lab for _, lab in label_rows])
        rf, svm, scaler, X_train_s, X_test_s, y_train, y_test = train_classifiers(X, y)

        # Recover which stems ended up in train/test (same split, same seed).
        stems_only = [s for s, _ in label_rows]
        idx_train, idx_test = train_test_split(
            np.arange(len(label_rows)), test_size=TEST_FRAC,
            random_state=RANDOM_SEED, stratify=y)
        train_stems = {stems_only[i] for i in idx_train}
        test_stems = {stems_only[i] for i in idx_test}

        rf_train_pred = rf.predict(X_train_s)
        svm_train_pred = svm.predict(X_train_s)
        ens_train_pred = ((rf.predict_proba(X_train_s)[:, 1]
                          + svm.predict_proba(X_train_s)[:, 1]) / 2 >= DETECT_PROB_THRESHOLD).astype(int)
        rf_test_pred = rf.predict(X_test_s)
        svm_test_pred = svm.predict(X_test_s)
        ens_test_pred = ((rf.predict_proba(X_test_s)[:, 1]
                         + svm.predict_proba(X_test_s)[:, 1]) / 2 >= DETECT_PROB_THRESHOLD).astype(int)
        _write_metrics(y_train, rf_train_pred, svm_train_pred, ens_train_pred,
                       y_test, rf_test_pred, svm_test_pred, ens_test_pred)

        order = np.argsort(rf.feature_importances_)[::-1]
        print("\n  Top RF features: " + ", ".join(
            f"{ML_FEATURE_KEYS[i]}({rf.feature_importances_[i]:.2f})" for i in order[:8]))

    # ── Pass 2: predict + explain + save outputs for every crop ──────────────
    print("\n  Predicting + explaining…")
    summaries: List[dict] = []
    pixel_check_records: List[dict] = []
    for stem in stems:
        gray, shape = gray_cache[stem]
        cands = all_candidates[stem]
        lid = _extract_label_id(stem)
        true_label = labels.get(lid, "")
        split = "train" if stem in train_stems else ("test" if stem in test_stems else "unlabeled")

        if gray is None:
            summaries.append({"stem": stem, "split": split, "true_label": true_label,
                              "rf_prob": "", "svm_prob": "", "ensemble_prob": "",
                              "detected": "", "n_candidates": 0, "status": "unreadable"})
            continue

        result = predict_and_explain(image_vecs[stem], rf, svm, scaler)
        best_id = None
        best_mask = None
        if cands:
            best = max(cands, key=lambda c: c["rule_score"])
            best_id, best_mask = best["id"], best["mask"]

        _save_detection(stem, gray, cands, best_id, bool(result["detected"]))
        _save_explanation(stem, gray, best_mask if result["detected"] else None, result)

        summaries.append({
            "stem": stem, "split": split, "true_label": true_label,
            "rf_prob": round(result["rf_prob"], 4), "svm_prob": round(result["svm_prob"], 4),
            "ensemble_prob": round(result["ensemble_prob"], 4),
            "detected": bool(result["detected"]), "n_candidates": len(cands), "status": "ok",
        })
        pixel_check_records.append({"stem": stem, "gray_shape": shape,
                                    "detected": bool(result["detected"]), "best_mask": best_mask})

    _safe_write_csv(os.path.join(OUTPUT_DIR, "image_summary.csv"),
                    ["stem", "split", "true_label", "rf_prob", "svm_prob", "ensemble_prob",
                     "detected", "n_candidates", "status"], summaries)

    if RUN_PIXEL_CHECK:
        pixel_overlap_check(pixel_check_records)

    print(f"\n  Outputs -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
