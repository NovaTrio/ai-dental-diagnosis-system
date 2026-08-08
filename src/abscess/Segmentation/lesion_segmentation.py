"""
rf_lesion_segmentation.py
─────────────────────────
Supervised periapical-lesion segmentation with a Random Forest pixel classifier
(classical machine learning — no deep learning).

    Crown-crop image  +  RF tooth mask  ──►  per-pixel features
                                              (appearance + tooth anatomy)
    Lesion mask (GT)                    ──►  labels
                                              ↓
                                     train Random Forest
                                              ↓
                    lesion predicted NEAR the tooth apex  ──► compare with GT

The tooth mask produced by ``rf_tooth_segmentation.py`` supplies anatomy features
(distance-to-tooth, inside/outside, distance-to-apex, position along / across the
root axis) and confines the search to a peri-tooth region, so the lesion is found
next to the root apex instead of anywhere dark in the frame.

Inputs
    Crops        : ../../../data/abscess/raw/croun_crops/<stem>.png
    Tooth masks  : ../../../data/abscess/raw/rf_tooth_output/tooth_mask/tooth_mask_<stem>.png
    Lesion masks : ../../../data/abscess/raw/mask/<stem>_mask*.png            (labels)
Outputs (../../../data/abscess/raw/rf_lesion_output/)
    rf_lesion_model.joblib
    lesion_mask/lesion_mask_<stem>.png     predicted (white = lesion)
    overlay/overlay_<stem>.png
    debug/debug_<stem>.png                 GT (green) / pred (red) / overlap (yellow)
    metrics.csv                            per-image + mean Dice/IoU/precision/recall

Usage:
    python rf_tooth_segmentation.py             # first, to make the tooth masks
    python rf_lesion_segmentation.py            # train + predict + evaluate
    python rf_lesion_segmentation.py --predict  # load saved model, predict only
"""

from __future__ import annotations

import csv
import math
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk

import joblib
from sklearn.ensemble import RandomForestClassifier


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw")
INPUT_DIR = os.path.join(_RAW, "croun_crops")
TOOTH_MASK_DIR = os.path.join(_RAW, "rf_tooth_output", "tooth_mask")   # RF tooth masks
LESION_MASK_DIR = os.path.join(_RAW, "mask")                           # lesion labels
OUTPUT_DIR = os.path.join(_RAW, "lesion_output")
# Trained weights live under src/ (version-controlled) so a fresh clone does not
# have to retrain. Predictions/overlays stay in OUTPUT_DIR (gitignored data/).
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models", "lesion_segmentation")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_lesion_model.joblib")
MODEL_COMPRESS = 3          # keeps the saved forest ~75% smaller

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")
RANDOM_SEED = 42

TEST_FRAC = 0.2
PREDICT_ALL = True

POS_PER_IMG = 4000
NEG_PER_IMG = 8000

# Feature extraction
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSS_SIGMAS = (1.0, 3.0, 6.0)
LOCAL_WINDOWS = (7, 15)
GABOR_ORIENTATIONS = (0, 45, 90, 135)
GABOR_KSIZE, GABOR_SIGMA, GABOR_LAMBDA, GABOR_GAMMA = 15, 3.0, 8.0, 0.5
LBP_P, LBP_R = 8, 1
ENTROPY_DISK = 3
BLACKHAT_KSIZE = 25
APEX_END_FRAC = 0.15

# Peri-tooth region of interest (where a periapical lesion can lie)
ROI_DILATE_MULT = 2.0        # ROI radius = this × root width …
ROI_MIN_FRAC = 0.25          # … or this × min(H, W), whichever is larger

# Random Forest
RF_ESTIMATORS = 300
RF_MAX_DEPTH = 18
RF_MIN_LEAF = 4
RF_MAX_FEATURES = "sqrt"

# Inference post-processing
PROB_THRESH = 0.5
MORPH_FRAC = 0.02
MIN_LESION_AREA_FRAC = 0.003
KEEP_MODE = "confident"    # "confident" (max prob mass) | "largest" | "nearest"
FILL_HOLES = True

SUBDIRS = {"mask": "lesion_mask", "overlay": "overlay", "debug": "debug"}
_EPS = 1e-8

_GABOR_BANK = [
    cv2.getGaborKernel((GABOR_KSIZE, GABOR_KSIZE), GABOR_SIGMA,
                       math.radians(t), GABOR_LAMBDA, GABOR_GAMMA, 0, ktype=cv2.CV_64F)
    for t in GABOR_ORIENTATIONS
]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _odd(n: int) -> int:
    n = int(round(n))
    return max(1, n + 1 if n % 2 == 0 else n)


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))


def _safe_div(a: float, b: float) -> float:
    return float(a) / (float(b) + _EPS)


def _ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in SUBDIRS.values():
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


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


def _find_in(mask_dir: str, stem: str, shape: Tuple[int, int],
             require_align: bool) -> Optional[np.ndarray]:
    if not os.path.isdir(mask_dir):
        return None
    cand = [m for m in os.listdir(mask_dir)
            if m.lower().endswith(SUPPORTED_EXTS)
            and m.startswith((f"{stem}_mask", f"tooth_mask_{stem}"))]
    if not cand:
        return None
    m = cv2.imread(os.path.join(mask_dir, cand[0]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        if require_align:
            return None
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def find_lesion_mask(stem: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    return _find_in(LESION_MASK_DIR, stem, shape, require_align=True)   # labels must align


def find_tooth_mask(stem: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    return _find_in(TOOTH_MASK_DIR, stem, shape, require_align=False)


# ═════════════════════════════════════════════════════════════════════════════
# Tooth anatomy (PCA axis + apex) and peri-tooth ROI
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
    return {"centroid": centroid, "minor": minor, "direction": direction, "apex": apex,
            "root_len": root_len, "root_width": root_width}


def peri_tooth_roi(tooth: Optional[np.ndarray], valid: np.ndarray,
                   anat: Optional[dict]) -> np.ndarray:
    if tooth is None or anat is None:
        return valid.copy()
    h, w = valid.shape
    r = int(max(ROI_DILATE_MULT * anat["root_width"], ROI_MIN_FRAC * min(h, w)))
    return (cv2.dilate(tooth, _ellipse(2 * r + 1)) > 0) & valid


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction (appearance + tooth anatomy) — same in train & inference
# ═════════════════════════════════════════════════════════════════════════════
def feature_stack(gray: np.ndarray, tooth: Optional[np.ndarray],
                  anat: Optional[dict]) -> Tuple[np.ndarray, List[str]]:
    h, w = gray.shape
    g = cv2.bilateralFilter(gray, 5, 40, 40)
    f = g.astype(np.float64) / 255.0
    maps: List[np.ndarray] = []
    names: List[str] = []

    def add(name: str, m: np.ndarray) -> None:
        maps.append(m.astype(np.float32))
        names.append(name)

    # ── Appearance ──
    add("intensity", f)
    add("darkness", 1.0 - f)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    add("clahe", clahe.apply(g).astype(np.float64) / 255.0)
    for s in GAUSS_SIGMAS:
        add(f"gauss{int(s)}", cv2.GaussianBlur(f, (0, 0), s))
    for win in LOCAL_WINDOWS:
        mean = cv2.blur(f, (win, win))
        std = np.sqrt(np.clip(cv2.blur(f * f, (win, win)) - mean ** 2, 0.0, None))
        add(f"lmean{win}", mean)
        add(f"lstd{win}", std)
        add(f"contrast{win}", np.clip(mean - f, 0.0, None))
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    add("gradmag", np.sqrt(gx ** 2 + gy ** 2))
    add("laplacian", np.abs(cv2.Laplacian(f, cv2.CV_64F, ksize=3)))
    resp = np.stack([np.abs(cv2.filter2D(f, cv2.CV_64F, k)) for k in _GABOR_BANK], 0)
    add("gabor_mean", resp.mean(0))
    add("gabor_max", resp.max(0))
    add("lbp", local_binary_pattern(g, LBP_P, LBP_R, method="uniform") / float(LBP_P + 2))
    add("blackhat", cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, _ellipse(BLACKHAT_KSIZE)).astype(np.float64) / 255.0)
    add("entropy", local_entropy(g, disk(ENTROPY_DISK)).astype(np.float64) / 8.0)

    # ── Tooth anatomy (from the RF tooth mask) ──
    diag = math.hypot(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if tooth is not None and anat is not None:
        add("dist_to_tooth", cv2.distanceTransform((tooth == 0).astype(np.uint8),
                                                    cv2.DIST_L2, 5) / diag)
        add("inside_tooth", tooth.astype(np.float64))
        ax, ay = anat["apex"]
        add("dist_to_apex", np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2) / diag)
        cx, cy = anat["centroid"]; d = anat["direction"]; mn = anat["minor"]
        add("apical_proj", ((xx - ax) * d[0] + (yy - ay) * d[1]) / anat["root_len"])
        add("perp_dist", np.abs((xx - cx) * mn[0] + (yy - cy) * mn[1]) / anat["root_width"])
    else:                                            # no tooth mask → neutral anatomy
        add("dist_to_tooth", np.full((h, w), 0.5))
        add("inside_tooth", np.zeros((h, w)))
        add("dist_to_apex", np.full((h, w), 0.5))
        add("apical_proj", np.zeros((h, w)))
        add("perp_dist", np.full((h, w), 0.5))

    return np.stack(maps, axis=-1), names


# ═════════════════════════════════════════════════════════════════════════════
# Training-set construction
# ═════════════════════════════════════════════════════════════════════════════
def labeled_stems() -> List[str]:
    stems = []
    for f in sorted(os.listdir(INPUT_DIR)):
        if not f.lower().endswith(SUPPORTED_EXTS):
            continue
        stem = os.path.splitext(f)[0]
        loaded = load_gray_valid(os.path.join(INPUT_DIR, f))
        if loaded and find_lesion_mask(stem, loaded[0].shape) is not None:
            stems.append(stem)
    return stems


def build_training_set(stems: List[str], rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_parts, y_parts, names = [], [], None
    for stem in stems:
        loaded = load_gray_valid(os.path.join(INPUT_DIR, f"{stem}.png"))
        if not loaded:
            continue
        gray, valid = loaded
        lesion = find_lesion_mask(stem, gray.shape)
        if lesion is None:
            continue
        tooth = find_tooth_mask(stem, gray.shape)
        anat = tooth_anatomy(tooth)
        roi = peri_tooth_roi(tooth, valid, anat)
        feats, names = feature_stack(gray, tooth, anat)
        flat = feats.reshape(-1, feats.shape[-1])
        lab = lesion.reshape(-1)
        # positives = lesion pixels; negatives = non-lesion pixels inside the ROI
        pos = np.flatnonzero((lab == 1) & valid.reshape(-1))
        neg = np.flatnonzero((lab == 0) & roi.reshape(-1))
        if pos.size < 20 or neg.size < 20:
            continue
        pos = rng.choice(pos, min(POS_PER_IMG, pos.size), replace=False)
        neg = rng.choice(neg, min(NEG_PER_IMG, neg.size), replace=False)
        idx = np.concatenate([pos, neg])
        x_parts.append(flat[idx])
        y_parts.append(lab[idx])
    if not x_parts:
        return np.empty((0, 0)), np.empty((0,)), names or []
    return np.vstack(x_parts), np.concatenate(y_parts), names


# ═════════════════════════════════════════════════════════════════════════════
# Prediction (confined to the peri-tooth ROI, keep the lesion nearest the tooth)
# ═════════════════════════════════════════════════════════════════════════════
def predict_mask(model: RandomForestClassifier, gray: np.ndarray, valid: np.ndarray,
                 tooth: Optional[np.ndarray], anat: Optional[dict]) -> np.ndarray:
    h, w = gray.shape
    feats, _ = feature_stack(gray, tooth, anat)
    roi = peri_tooth_roi(tooth, valid, anat)
    flat = feats.reshape(-1, feats.shape[-1])
    ridx = np.flatnonzero(roi.reshape(-1))
    prob = np.zeros(h * w, np.float32)
    if ridx.size:
        prob[ridx] = model.predict_proba(flat[ridx])[:, 1]
    mask = ((prob >= PROB_THRESH).reshape(h, w).astype(np.uint8)) * 255

    ker = _ellipse(max(3, min(h, w) * MORPH_FRAC))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)

    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    min_area = MIN_LESION_AREA_FRAC * h * w
    cands = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    keep = np.zeros_like(mask)
    if cands:
        prob2d = prob.reshape(h, w)
        if KEEP_MODE == "confident":
            # highest total probability mass = confident AND sizeable (beats a
            # large low-confidence spurious blob).
            best = max(cands, key=lambda i: float(prob2d[lab == i].sum()))
        elif KEEP_MODE == "nearest" and tooth is not None:
            dt = cv2.distanceTransform((tooth == 0).astype(np.uint8), cv2.DIST_L2, 5)
            best = min(cands, key=lambda i: float(dt[lab == i].min()))
        else:
            best = max(cands, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        keep[lab == best] = 255
    if FILL_HOLES and keep.max() > 0:
        keep = ndi.binary_fill_holes(keep > 0).astype(np.uint8) * 255
    keep[~valid] = 0
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Reusable single-image inference (for run_pipeline.py and other callers)
# ─────────────────────────────────────────────────────────────────────────────
_LESION_MODEL = None


def load_lesion_model(model_path: str = MODEL_PATH) -> RandomForestClassifier:
    """Load (and cache) the trained RF lesion model. Raises if it is missing."""
    global _LESION_MODEL
    if _LESION_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"lesion model not found: {model_path}\n"
                f"Train it first:  python lesion_segmentation.py")
        _LESION_MODEL = joblib.load(model_path)["model"]
    return _LESION_MODEL


def _coerce_tooth(tooth_mask, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Accept a tooth mask as an ndarray, an image path, or None; return a 0/1
    mask resized to ``shape`` (or None)."""
    if isinstance(tooth_mask, str):
        m = cv2.imread(tooth_mask, cv2.IMREAD_GRAYSCALE)
    elif tooth_mask is not None:
        m = np.asarray(tooth_mask)
    else:
        return None
    if m is None:
        return None
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > (127 if m.max() > 1 else 0)).astype(np.uint8)


def segment_lesion_image(image_path: str, tooth_mask=None,
                         model: Optional[RandomForestClassifier] = None
                         ) -> Optional[np.ndarray]:
    """
    Predict the lesion mask for ONE crown-crop image (reusable inference).

    Parameters
    ----------
    image_path : path to the crop (grayscale / BGR / BGRA; alpha = valid region).
    tooth_mask : the RF/manual tooth mask as an ndarray, a path, or None. If None
                 it is looked up in TOOTH_MASK_DIR by the image stem. Supplies the
                 anatomy features + the peri-tooth ROI that localise the lesion.
    model      : a loaded RandomForestClassifier, or None to use the cached model
                 from MODEL_PATH.

    Returns
    -------
    uint8 {0, 255} lesion mask (same H×W as the crop), or None if unreadable.
    """
    loaded = load_gray_valid(image_path)
    if loaded is None:
        return None
    gray, valid = loaded

    tooth = _coerce_tooth(tooth_mask, gray.shape)
    if tooth is None and tooth_mask is None:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        tooth = find_tooth_mask(stem, gray.shape)

    anat = tooth_anatomy(tooth)
    mdl = model if model is not None else load_lesion_model()
    return predict_mask(mdl, gray, valid, tooth, anat)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    p, g = pred > 127, gt > 127
    tp = int(np.sum(p & g)); fp = int(np.sum(p & ~g)); fn = int(np.sum(~p & g))
    return (_safe_div(2 * tp, 2 * tp + fp + fn), _safe_div(tp, tp + fp + fn),
            _safe_div(tp, tp + fp), _safe_div(tp, tp + fn))


def save_prediction(stem: str, gray: np.ndarray, mask: np.ndarray,
                    gt: Optional[np.ndarray]) -> None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, SUBDIRS["mask"], f"lesion_mask_{stem}.png"), mask)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    if mask.max() > 0:
        tint = overlay.copy(); tint[mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.6, tint, 0.4, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, SUBDIRS["overlay"], f"overlay_{stem}.png"), overlay)

    debug = base.copy()
    if gt is not None:
        g = gt > 0; p = mask > 127
        debug[g & ~p] = (0, 200, 0)      # missed (GT only) = green
        debug[p & ~g] = (0, 0, 200)      # false positive = red
        debug[g & p] = (0, 255, 255)     # correct overlap = yellow
    else:
        debug[mask > 0] = (0, 0, 200)
    cv2.imwrite(os.path.join(OUTPUT_DIR, SUBDIRS["debug"], f"debug_{stem}.png"), debug)


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
    print("  RF lesion segmentation (crop + RF tooth mask -> lesion)")
    print("=" * 60)
    for d, name in [(INPUT_DIR, "crops"), (LESION_MASK_DIR, "lesion masks")]:
        if not os.path.isdir(d):
            print(f"[ERROR] {name} directory not found: {d}")
            return
    if not os.path.isdir(TOOTH_MASK_DIR):
        print(f"[WARN] tooth-mask dir not found ({TOOTH_MASK_DIR}); "
              f"run rf_tooth_segmentation.py — anatomy features will be neutral.")
    _ensure_dirs()

    stems = labeled_stems()
    if not stems:
        print("[ERROR] no crops with an aligned lesion mask.")
        return
    print(f"  Labeled images (aligned lesion mask): {len(stems)}")

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = list(stems); rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_FRAC))
    test_stems = set(shuffled[:n_test])
    train_stems = shuffled[n_test:]
    print(f"  Train: {len(train_stems)}   Test: {len(test_stems)}")

    if predict_only and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)["model"]
        print(f"  Loaded model from {MODEL_PATH}")
    else:
        print("\n  Sampling training pixels…")
        X, y, names = build_training_set(train_stems, rng)
        if X.size == 0:
            print("[ERROR] no training samples.")
            return
        print(f"  Training on {X.shape[0]:,} pixels x {X.shape[1]} features "
              f"({int(y.sum())} lesion / {int((y == 0).sum())} non-lesion)…")
        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_LEAF, max_features=RF_MAX_FEATURES,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
        model.fit(X, y)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({"model": model, "features": names}, MODEL_PATH,
                    compress=MODEL_COMPRESS)
        print(f"  Saved model -> {MODEL_PATH}")
        order = np.argsort(model.feature_importances_)[::-1]
        print("  Top features: " + ", ".join(
            f"{names[i]}({model.feature_importances_[i]:.2f})" for i in order[:8]))

    print("\n  Predicting…")
    rows, test_scores, train_scores = [], [], []
    targets = ([os.path.splitext(f)[0] for f in sorted(os.listdir(INPUT_DIR))
                if f.lower().endswith(SUPPORTED_EXTS)] if PREDICT_ALL else sorted(test_stems))
    for stem in targets:
        loaded = load_gray_valid(os.path.join(INPUT_DIR, f"{stem}.png"))
        if not loaded:
            continue
        gray, valid = loaded
        tooth = find_tooth_mask(stem, gray.shape)
        anat = tooth_anatomy(tooth)
        mask = predict_mask(model, gray, valid, tooth, anat)
        gt = find_lesion_mask(stem, gray.shape)
        save_prediction(stem, gray, mask, gt)
        if gt is not None and stem in stems:
            dice, iou, prec, rec = _metrics(mask, gt * 255)
            split = "test" if stem in test_stems else "train"
            rows.append({"stem": stem, "split": split, "dice": round(dice, 4),
                         "iou": round(iou, 4), "precision": round(prec, 4),
                         "recall": round(rec, 4)})
            (test_scores if split == "test" else train_scores).append((dice, iou, prec, rec))

    with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["stem", "split", "dice", "iou",
                                            "precision", "recall"])
        wr.writeheader()
        wr.writerows(rows)

    def summary(title, scores):
        print(f"\n  {title}  (n={len(scores)})")
        if scores:
            a = np.array(scores)
            for i, nm in enumerate(["Dice", "IoU", "Precision", "Recall"]):
                print(f"    {nm:10s}: {a[:, i].mean():.4f} ± {a[:, i].std():.4f}")

    print("\n" + "=" * 60)
    print("  RESULTS vs lesion masks")
    print("=" * 60)
    summary("TRAIN split (in-sample)", train_scores)
    summary("TEST split (held out) — trustworthy", test_scores)
    print(f"\n  Masks/overlays/debug -> {OUTPUT_DIR}")
    print(f"  Per-image metrics    -> {os.path.join(OUTPUT_DIR, 'metrics.csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
