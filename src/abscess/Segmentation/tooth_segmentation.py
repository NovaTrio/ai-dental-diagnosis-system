"""
rf_tooth_segmentation.py
────────────────────────
Supervised tooth segmentation with a Random Forest pixel classifier
(classical machine learning — no deep learning).

    Crown-cropped images  +  manual tooth masks
                 ↓
        per-pixel feature extraction
                 ↓
        train Random Forest (tooth vs. non-tooth)
                 ↓
        Tooth Segmentation Model  →  predicted tooth mask

Each pixel is described by handcrafted, multi-scale features (intensity, CLAHE,
Gaussian scales, local mean/std, gradient, Laplacian, Gabor, LBP, entropy and
normalized position). The forest learns to separate tooth from the equally-bright
alveolar bone using these cues plus spatial context. Images are split into
train/test so the reported metrics are honest.

Inputs
    Crops        : ../../../data/abscess/raw/croun_crops/<stem>.png
    Manual masks : ../../../data/abscess/raw/tooth_masks/<stem>_mask*.png
Outputs (../../../data/abscess/raw/rf_tooth_output/)
    rf_tooth_model.joblib              trained model + feature list
    tooth_mask/tooth_mask_<stem>.png   predicted mask (white = tooth)
    overlay/overlay_<stem>.png         mask outline on the crop
    metrics.csv                        per-image + mean Dice/IoU/precision/recall

Usage:
    python rf_tooth_segmentation.py            # train + predict + evaluate
    python rf_tooth_segmentation.py --predict  # load saved model, predict only
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
TOOTH_MASK_DIR = os.path.join(_RAW, "tooth_masks")          # manual tooth masks
OUTPUT_DIR = os.path.join(_RAW, "rf_tooth_output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_tooth_model.joblib")

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")
RANDOM_SEED = 42

# ── Train / test split ───────────────────────────────────────────────────────
TEST_FRAC = 0.2               # fraction of images held out for evaluation
PREDICT_ALL = True            # also predict + save masks for every image

# ── Pixel sampling for training ──────────────────────────────────────────────
SAMPLES_PER_CLASS_PER_IMG = 3000   # balanced tooth / non-tooth samples per image

# ── Feature extraction ───────────────────────────────────────────────────────
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSS_SIGMAS = (1.0, 3.0, 6.0)
LOCAL_WINDOWS = (7, 15)
GABOR_ORIENTATIONS = (0, 45, 90, 135)
GABOR_KSIZE, GABOR_SIGMA, GABOR_LAMBDA, GABOR_GAMMA = 15, 3.0, 8.0, 0.5
LBP_P, LBP_R = 8, 1
ENTROPY_DISK = 3

# ── Random Forest ────────────────────────────────────────────────────────────
RF_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_LEAF = 4
RF_MAX_FEATURES = "sqrt"

# ── Inference post-processing ────────────────────────────────────────────────
PROB_THRESH = 0.5
MORPH_FRAC = 0.02             # morphology kernel as a fraction of min(H, W)
KEEP_LARGEST = True           # keep only the largest predicted tooth component
FILL_HOLES = True

SUBDIRS = {"mask": "tooth_mask", "overlay": "overlay"}
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


# ═════════════════════════════════════════════════════════════════════════════
# Load
# ═════════════════════════════════════════════════════════════════════════════
def load_gray_valid(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (gray_uint8, valid_mask) or None. Transparent pixels excluded."""
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


def find_tooth_mask(stem: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load the manual tooth mask for ``stem`` (resized to the crop if needed)."""
    if not os.path.isdir(TOOTH_MASK_DIR):
        return None
    cand = [m for m in os.listdir(TOOTH_MASK_DIR)
            if m.startswith(f"{stem}_mask") and m.lower().endswith(SUPPORTED_EXTS)]
    if not cand:
        return None
    m = cv2.imread(os.path.join(TOOTH_MASK_DIR, cand[0]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction  (identical for training and inference)
# ═════════════════════════════════════════════════════════════════════════════
def feature_stack(gray: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Build the per-pixel feature volume (H, W, F) and the feature-name list.
    Multi-scale intensity + texture + edges + Gabor/LBP/entropy + position.
    """
    h, w = gray.shape
    g = cv2.bilateralFilter(gray, 5, 40, 40)
    f = g.astype(np.float64) / 255.0

    maps: List[np.ndarray] = []
    names: List[str] = []

    def add(name: str, m: np.ndarray) -> None:
        maps.append(m.astype(np.float32))
        names.append(name)

    add("intensity", f)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    add("clahe", clahe.apply(g).astype(np.float64) / 255.0)

    for s in GAUSS_SIGMAS:
        add(f"gauss{int(s)}", cv2.GaussianBlur(f, (0, 0), s))

    for win in LOCAL_WINDOWS:
        mean = cv2.blur(f, (win, win))
        mean_sq = cv2.blur(f * f, (win, win))
        std = np.sqrt(np.clip(mean_sq - mean ** 2, 0.0, None))
        add(f"lmean{win}", mean)
        add(f"lstd{win}", std)

    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    add("gradmag", np.sqrt(gx ** 2 + gy ** 2))
    add("laplacian", np.abs(cv2.Laplacian(f, cv2.CV_64F, ksize=3)))

    responses = np.stack([np.abs(cv2.filter2D(f, cv2.CV_64F, k)) for k in _GABOR_BANK], 0)
    add("gabor_mean", responses.mean(0))
    add("gabor_max", responses.max(0))

    add("lbp", local_binary_pattern(g, LBP_P, LBP_R, method="uniform") / float(LBP_P + 2))
    add("entropy", local_entropy(g, disk(ENTROPY_DISK)).astype(np.float64))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    add("x_norm", xx / max(w - 1, 1))
    add("y_norm", yy / max(h - 1, 1))
    add("x_centrality", 1.0 - np.abs(xx / max(w - 1, 1) - 0.5) * 2.0)

    return np.stack(maps, axis=-1), names


# ═════════════════════════════════════════════════════════════════════════════
# Dataset (sampled training pixels)
# ═════════════════════════════════════════════════════════════════════════════
def build_training_set(stems: List[str], rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Sample balanced tooth / non-tooth pixels from the training images."""
    X_parts, y_parts, names = [], [], None
    for stem in stems:
        loaded = load_gray_valid(os.path.join(INPUT_DIR, f"{stem}.png"))
        if loaded is None:
            continue
        gray, valid = loaded
        tooth = find_tooth_mask(stem, gray.shape)
        if tooth is None:
            continue
        feats, names = feature_stack(gray)
        flat = feats.reshape(-1, feats.shape[-1])
        lab = tooth.reshape(-1)
        val = valid.reshape(-1)

        pos = np.flatnonzero((lab == 1) & val)
        neg = np.flatnonzero((lab == 0) & val)
        n = min(SAMPLES_PER_CLASS_PER_IMG, len(pos), len(neg))
        if n < 10:
            continue
        pos = rng.choice(pos, n, replace=False)
        neg = rng.choice(neg, n, replace=False)
        idx = np.concatenate([pos, neg])
        X_parts.append(flat[idx])
        y_parts.append(lab[idx])

    if not X_parts:
        return np.empty((0, 0)), np.empty((0,)), names or []
    return np.vstack(X_parts), np.concatenate(y_parts), names


# ═════════════════════════════════════════════════════════════════════════════
# Predict one image
# ═════════════════════════════════════════════════════════════════════════════
def predict_mask(model: RandomForestClassifier, gray: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    """Per-pixel RF prediction → cleaned tooth mask (uint8 {0,255})."""
    h, w = gray.shape
    feats, _ = feature_stack(gray)
    flat = feats.reshape(-1, feats.shape[-1])
    vidx = np.flatnonzero(valid.reshape(-1))
    prob = np.zeros(h * w, np.float32)
    prob[vidx] = model.predict_proba(flat[vidx])[:, 1]
    mask = ((prob >= PROB_THRESH).reshape(h, w).astype(np.uint8)) * 255

    ker = _ellipse(max(3, min(h, w) * MORPH_FRAC))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    if KEEP_LARGEST:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        if n > 1:
            mask = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
    if FILL_HOLES and mask.max() > 0:
        mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255
    mask[~valid] = 0
    return mask


def save_prediction(stem: str, gray: np.ndarray, mask: np.ndarray) -> None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, SUBDIRS["mask"], f"tooth_mask_{stem}.png"), mask)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask.max() > 0:
        tint = overlay.copy(); tint[mask > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.7, tint, 0.3, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, SUBDIRS["overlay"], f"overlay_{stem}.png"), overlay)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    p, g = pred > 127, gt > 127
    tp = int(np.sum(p & g)); fp = int(np.sum(p & ~g)); fn = int(np.sum(~p & g))
    return (_safe_div(2 * tp, 2 * tp + fp + fn), _safe_div(tp, tp + fp + fn),
            _safe_div(tp, tp + fp), _safe_div(tp, tp + fn))


# ─────────────────────────────────────────────────────────────────────────────
# Reusable single-image inference (for run_pipeline.py and other callers)
# ─────────────────────────────────────────────────────────────────────────────
_TOOTH_MODEL = None


def load_tooth_model(model_path: str = MODEL_PATH) -> RandomForestClassifier:
    """Load (and cache) the trained RF tooth model. Raises if it is missing."""
    global _TOOTH_MODEL
    if _TOOTH_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"tooth model not found: {model_path}\n"
                f"Train it first:  python tooth_segmentation.py")
        _TOOTH_MODEL = joblib.load(model_path)["model"]
    return _TOOTH_MODEL


def segment_tooth_image(image_path: str,
                        model: Optional[RandomForestClassifier] = None
                        ) -> Optional[np.ndarray]:
    """
    Predict the tooth mask for ONE crown-crop image (reusable inference).

    Parameters
    ----------
    image_path : path to the crop (grayscale / BGR / BGRA; alpha = valid region).
    model      : a loaded RandomForestClassifier, or None to use the cached model
                 from MODEL_PATH.

    Returns
    -------
    uint8 {0, 255} tooth mask (same H×W as the crop), or None if unreadable.
    """
    loaded = load_gray_valid(image_path)
    if loaded is None:
        return None
    gray, valid = loaded
    mdl = model if model is not None else load_tooth_model()
    return predict_mask(mdl, gray, valid)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def list_labeled_stems() -> List[str]:
    """Crop stems that have a matching manual tooth mask."""
    stems = []
    for f in sorted(os.listdir(INPUT_DIR)):
        if not f.lower().endswith(SUPPORTED_EXTS):
            continue
        stem = os.path.splitext(f)[0]
        loaded = load_gray_valid(os.path.join(INPUT_DIR, f))
        if loaded and find_tooth_mask(stem, loaded[0].shape) is not None:
            stems.append(stem)
    return stems


def main() -> None:
    try:                                   # avoid cp1252 console crashes on Windows
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    predict_only = "--predict" in sys.argv
    print("=" * 60)
    print("  Random Forest tooth segmentation")
    print("=" * 60)
    for d, name in [(INPUT_DIR, "crops"), (TOOTH_MASK_DIR, "tooth masks")]:
        if not os.path.isdir(d):
            print(f"[ERROR] {name} directory not found: {d}")
            return
    _ensure_dirs()

    stems = list_labeled_stems()
    if not stems:
        print("[ERROR] no crops with a matching manual tooth mask found.")
        return
    print(f"  Labeled images: {len(stems)}")

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_FRAC))
    test_stems = set(shuffled[:n_test])
    train_stems = shuffled[n_test:]
    print(f"  Train: {len(train_stems)}   Test: {len(test_stems)}")

    # ── Train (or load) ──────────────────────────────────────────────────────
    if predict_only and os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        print(f"  Loaded model from {MODEL_PATH}")
    else:
        print("\n  Sampling training pixels…")
        X, y, names = build_training_set(train_stems, rng)
        if X.size == 0:
            print("[ERROR] no training samples.")
            return
        print(f"  Training on {X.shape[0]:,} pixels × {X.shape[1]} features…")
        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_LEAF, max_features=RF_MAX_FEATURES,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
        model.fit(X, y)
        joblib.dump({"model": model, "features": names}, MODEL_PATH)
        print(f"  Saved model → {MODEL_PATH}")
        order = np.argsort(model.feature_importances_)[::-1]
        print("  Top features: " + ", ".join(
            f"{names[i]}({model.feature_importances_[i]:.2f})" for i in order[:6]))

    # ── Predict + evaluate ───────────────────────────────────────────────────
    print("\n  Predicting…")
    rows, test_scores = [], []
    targets = stems if PREDICT_ALL else sorted(test_stems)
    for stem in targets:
        loaded = load_gray_valid(os.path.join(INPUT_DIR, f"{stem}.png"))
        if loaded is None:
            continue
        gray, valid = loaded
        mask = predict_mask(model, gray, valid)
        save_prediction(stem, gray, mask)

        gt = find_tooth_mask(stem, gray.shape)
        split = "test" if stem in test_stems else "train"
        if gt is not None:
            dice, iou, prec, rec = _metrics(mask, gt * 255)
            rows.append({"stem": stem, "split": split, "dice": round(dice, 4),
                         "iou": round(iou, 4), "precision": round(prec, 4),
                         "recall": round(rec, 4)})
            if split == "test":
                test_scores.append((dice, iou, prec, rec))

    with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["stem", "split", "dice", "iou",
                                            "precision", "recall"])
        wr.writeheader()
        wr.writerows(rows)

    print("\n" + "=" * 60)
    print("  TEST-SET performance (held out, not seen in training)")
    print("=" * 60)
    if test_scores:
        a = np.array(test_scores)
        for i, name in enumerate(["Dice", "IoU", "Precision", "Recall"]):
            print(f"    {name:10s}: {a[:, i].mean():.4f} ± {a[:, i].std():.4f}")
    print(f"\n  Masks + overlays → {OUTPUT_DIR}")
    print(f"  Per-image metrics → {os.path.join(OUTPUT_DIR, 'metrics.csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
