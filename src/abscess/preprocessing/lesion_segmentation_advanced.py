"""
lesion_segmentation_advanced.py
───────────────────────────────
An anatomically-constrained, multi-feature spatial Fuzzy C-Means
lesion-candidate segmentation and feature-based validation pipeline
for periapical lesion detection on crown-cropped dental radiographs.

This module supersedes the earlier darkest-cluster approach in
`lesion_segmentation.py`. The old method simply declared the FCM cluster
with the lowest intensity centroid to be the lesion. That is unreliable
because root canals, marrow spaces, the periodontal-ligament space,
inter-radicular bone and image borders are *also* radiolucent, and because
it can never return "no lesion". Here we instead:

  1.  Preprocess (alpha-aware CLAHE + edge-preserving denoise).
  2.  Restrict the search to an *apical ROI* where periapical lesions are
      clinically expected (orientation-driven; uses a detected root apex
      when a real tooth silhouette is available, otherwise a configurable
      fallback band and an apex *proxy* on the apical edge).
  3.  Describe every ROI pixel with a 12-D feature vector (intensity,
      local mean / std, gradient, Laplacian, multi-orientation Gabor, LBP,
      local entropy, normalized x / y, distance-from-apex).
  4.  Cluster with *spatial* Fuzzy C-Means (skfuzzy), m = 2, fixed seed.
  5.  Rank clusters by darkness + texture + local contrast + apical
      proximity (NOT darkest-only) to pick lesion-candidate cluster(s).
  6.  Clean, split into connected components, and describe every candidate
      with geometric / intensity / texture / anatomical features.
  7.  Score each candidate with a transparent weighted rule
      (lesion_score ∈ [0, 1]) plus hard gates, and accept / reject.
  8.  Keep candidates above a threshold (best or all); return an *empty*
      mask when nothing qualifies — the system never forces a lesion.
  9.  Optionally refine the boundary with Morphological Chan–Vese.
 10.  Save all debug / result images, a per-candidate CSV and an
      image-level summary CSV.
 11.  Optionally evaluate against ground-truth masks (Dice, IoU, precision,
      recall, specificity, accuracy, Hausdorff).

Input  : PNG / JPG / JPEG from  ../../../data/abscess/raw/croun_crops
Output :                        ../../../data/abscess/raw/lesion_segmented
GT     : (optional eval)        ../../../data/abscess/raw/mask

All tunable thresholds and weights live in the CONFIGURATION section below;
no unexplained magic numbers are buried inside functions.

Usage:
    python lesion_segmentation_advanced.py
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import skfuzzy as fuzz
from scipy import ndimage as ndi
from scipy.spatial.distance import directed_hausdorff

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters.rank import entropy as local_entropy_filter
from skimage.morphology import disk
from skimage.measure import shannon_entropy

# Chan–Vese is optional; guard the import so the pipeline still runs without it.
try:
    from skimage.segmentation import morphological_chan_vese
    _HAS_CHAN_VESE = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_CHAN_VESE = False


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  every tunable value lives here
# ═════════════════════════════════════════════════════════════════════════════

# ── Paths (relative to this script in src/abscess/preprocessing/) ────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw", "croun_crops")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw", "lesion_segmented")
MASK_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "abscess", "raw", "mask")

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")
RANDOM_SEED = 42  # global reproducibility

# ── Preprocessing ────────────────────────────────────────────────────────────
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
BILATERAL_D = 5           # neighbourhood diameter; <=0 lets sigmaSpace decide
BILATERAL_SIGMA_COLOR = 40.0
BILATERAL_SIGMA_SPACE = 40.0

# ── Anatomical ROI ───────────────────────────────────────────────────────────
# "mandibular" → roots/apex point DOWN  → apical ROI at the BOTTOM.
# "maxillary"  → roots/apex point UP     → apical ROI at the TOP.
# "auto"       → decide per-image from the more radiolucent (darker) end.
TOOTH_ORIENTATION = "auto"
APICAL_ROI_FRACTION = 0.65   # fraction of tooth height kept as the ROI band
# A real root-apex is only trusted when the tooth silhouette actually narrows.
# Crops whose foreground fills a near-full rectangle have no usable silhouette,
# so we fall back to a generic band and an apex proxy.
SILHOUETTE_NARROWING_MIN = 0.15  # min (1 - min_width/max_width) to trust apex
# ROI shape used when NO reliable apex can be detected (untrusted silhouette):
#   "centered"   → a mid-root band centred on the tooth. Best generic prior when
#                  the crop framing does not place the apex at a known edge — the
#                  apex could be at either end, and lesions are rarely at the very
#                  crown or the far margins. Still an anatomical restriction.
#   "apical_end" → the classic upper/lower band chosen by TOOTH_ORIENTATION.
# When a real apex IS detected, an apical-end band around it is always used.
FALLBACK_ROI_MODE = "centered"

# ── Per-pixel feature weights (scale each normalized feature before FCM) ──────
FEATURE_WEIGHTS: Dict[str, float] = {
    "intensity":  1.00,   # radiolucency is the dominant lesion cue
    "local_mean": 0.50,
    "local_std":  0.70,
    "gradient":   0.50,
    "laplacian":  0.40,
    "gabor_mean": 0.50,
    "gabor_max":  0.60,
    "lbp":        0.30,   # weak ordinal cue → low weight
    "entropy":    0.50,
    "x":          0.15,   # spatial cohesion only, never dominant
    "y":          0.15,
    "apex_dist":  0.40,
}
LOCAL_WINDOW = 7          # window (px) for local mean / std
GABOR_KSIZE = 15
GABOR_SIGMA = 3.0
GABOR_LAMBDA = 8.0
GABOR_GAMMA = 0.5
GABOR_ORIENTATIONS = (0, 45, 90, 135)  # degrees
LBP_P = 8
LBP_R = 1
ENTROPY_DISK = 3

# ── Spatial Fuzzy C-Means ────────────────────────────────────────────────────
N_CLUSTERS = 3            # background / healthy bone / (candidate) lesion
FUZZINESS = 2.0           # standard fuzziness exponent m
FCM_ERROR = 0.005
FCM_MAXITER = 1000

# ── Cluster ranking → lesion-candidate cluster(s) ────────────────────────────
CLUSTER_RANK_WEIGHTS: Dict[str, float] = {
    "darkness": 0.45,     # (1 - normalized intensity)
    "texture":  0.20,     # normalized local std / entropy
    "contrast": 0.20,     # darker than ROI average
    "apex":     0.15,     # closeness to apex
}
N_CANDIDATE_CLUSTERS = 1  # how many top-ranked clusters feed candidate stage
CANDIDATE_CLUSTER_REL = 0.90  # also include clusters >= REL * best rank score

# ── Morphological cleanup ────────────────────────────────────────────────────
MORPH_KERNEL = 3
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 2
FILL_HOLES = True
MIN_COMPONENT_AREA_FRAC = 0.0015  # drop blobs smaller than this * image area

# ── Candidate contrast ring ──────────────────────────────────────────────────
RING_PX = 12              # thickness of the surrounding ring for contrast

# ── Candidate scoring (weighted, transparent) ────────────────────────────────
SCORE_WEIGHTS: Dict[str, float] = {
    "apex":     0.20,     # apical proximity
    "contrast": 0.25,     # darker than surrounding tissue
    "texture":  0.15,     # texture abnormality
    "area":     0.15,     # size plausibility
    "shape":    0.15,     # solidity / circularity plausibility
    "roi":      0.10,     # overlap with apical ROI
}
SCORE_PENALTIES: Dict[str, float] = {
    "border":   0.25,     # subtract if candidate hugs the image border
    "fragment": 0.20,     # subtract if candidate is thin / fragmented
}
SCORE_THRESHOLD = 0.50    # keep candidates whose lesion_score >= this
SELECT_MODE = "all"       # "best" (single highest) or "all" accepted lesions

# Hard gates + score ramps (fractions are of total image area) ────────────────
MIN_AREA_FRAC = 0.0015    # below → rejected outright
MAX_AREA_FRAC = 0.45      # above → rejected outright
PREF_AREA_FRAC = (0.010, 0.150)   # plateau where area_score == 1
CONTRAST_NORM = 0.12      # intensity gap (0-1) mapped to full contrast score
TEXTURE_CONTRAST_NORM = 400.0     # GLCM contrast value mapped to full score
MIN_SOLIDITY = 0.50       # used by the soft shape score
GATE_ON_SOLIDITY = False  # if True, solidity < MIN_SOLIDITY is a HARD reject.
                          # Off by default: periapical lesions are frequently
                          # irregular, so solidity is a soft cue, not a veto.
CIRCULARITY_RANGE = (0.10, 1.20)  # acceptable circularity band
MIN_ROI_OVERLAP = 0.50    # candidate must overlap ROI at least this much
BORDER_TOUCH_MAX = 0.35   # reject if more than this fraction of the contour
                          # lies on the image border
BORDER_MARGIN_PX = 2      # pixels-from-edge counted as "on the border"

# ── Optional boundary refinement (Morphological Chan–Vese) ───────────────────
ENABLE_REFINEMENT = False
CHAN_VESE_ITER = 40
CHAN_VESE_SMOOTHING = 2
CHAN_VESE_LAMBDA1 = 1.0
CHAN_VESE_LAMBDA2 = 1.0

# ── GLCM ─────────────────────────────────────────────────────────────────────
GLCM_LEVELS = 32          # intensity quantisation for the GLCM
GLCM_DISTANCES = (1,)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

# ── Evaluation ───────────────────────────────────────────────────────────────
RUN_EVALUATION = True

# ── Output sub-directories (created under OUTPUT_DIR) ─────────────────────────
SUBDIRS = {
    "enhanced": "enhanced",
    "roi_mask": "roi_mask",
    "roi_vis": "roi_vis",
    "cluster_map": "cluster_map",
    "candidate_raw": "candidate_raw",
    "candidate_clean": "candidate_clean",
    "final_mask": "final_mask",
    "overlay": "overlay",
    "contours": "contours",
    "scores": "scores",
    "ring": "ring",
    "reports": "reports",
    "evaluation": "evaluation",
}

_EPS = 1e-8  # guards every division


# ═════════════════════════════════════════════════════════════════════════════
# Small numeric helpers
# ═════════════════════════════════════════════════════════════════════════════
def _clip01(x: float) -> float:
    """Clamp a scalar to [0, 1]."""
    return float(min(1.0, max(0.0, x)))


def _safe_div(num: float, den: float) -> float:
    """Division that never raises on a zero denominator."""
    return float(num) / (float(den) + _EPS)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1] (constant arrays → zeros)."""
    a = arr.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < _EPS:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _ensure_dirs() -> None:
    """Create the output directory tree."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in SUBDIRS.values():
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def _out_path(key: str, stem: str, prefix: str) -> str:
    """Build an output path that preserves the original filename stem."""
    return os.path.join(OUTPUT_DIR, SUBDIRS[key], f"{prefix}_{stem}.png")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Input & preprocessing
# ═════════════════════════════════════════════════════════════════════════════
def load_image(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Read an image and return (gray_uint8, valid_mask).

    Handles grayscale, BGR and BGRA. For BGRA, fully transparent pixels
    (alpha == 0) are excluded from processing via ``valid_mask``.

    Returns None if the file cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:                       # grayscale
        gray = img
        valid = np.ones(gray.shape, dtype=bool)
    elif img.shape[2] == 4:                 # BGRA
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        valid = img[:, :, 3] > 0
    else:                                   # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = np.ones(gray.shape[:2], dtype=bool)

    return gray.astype(np.uint8), valid


def preprocess_image(gray: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply CLAHE (local contrast) then bilateral filtering (edge-preserving
    denoise). Returns (enhanced_uint8, enhanced_float01).

    Invalid pixels are zeroed after enhancement so they never influence
    downstream clustering.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(
        enhanced, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )
    enhanced = enhanced.copy()
    enhanced[~valid] = 0
    enhanced_f = enhanced.astype(np.float64) / 255.0
    return enhanced, enhanced_f


# ═════════════════════════════════════════════════════════════════════════════
# 2. Anatomical ROI restriction
# ═════════════════════════════════════════════════════════════════════════════
def detect_orientation(enhanced_f: np.ndarray, valid: np.ndarray) -> str:
    """
    Resolve TOOTH_ORIENTATION == "auto" to "maxillary" or "mandibular".

    Heuristic: the crown (bright enamel/dentin) is radiopaque while the
    periapical/root end is comparatively radiolucent. We therefore treat the
    darker vertical end as the apical end. Darker TOP → apex up → maxillary;
    darker BOTTOM → apex down → mandibular. This is a transparent fallback,
    used only because these crops carry no usable tooth silhouette.
    """
    if TOOTH_ORIENTATION in ("mandibular", "maxillary"):
        return TOOTH_ORIENTATION

    h = enhanced_f.shape[0]
    band = max(1, int(h * 0.30))
    top = enhanced_f[:band][valid[:band]]
    bot = enhanced_f[-band:][valid[-band:]]
    top_mean = float(top.mean()) if top.size else 1.0
    bot_mean = float(bot.mean()) if bot.size else 1.0
    return "maxillary" if top_mean <= bot_mean else "mandibular"


def estimate_apex(valid: np.ndarray, orientation: str) -> Tuple[Tuple[int, int], bool]:
    """
    Estimate the root-apex point.

    When the foreground silhouette narrows toward one end (a real tooth
    outline), the apex is the midpoint of the narrowest apical row.
    Otherwise (near-rectangular alpha, as in these crops) we return a
    *proxy* apex at the centre of the apical edge and flag it as untrusted.

    Returns ((x, y), trusted).
    """
    h, w = valid.shape
    rows_with_fg = np.where(valid.any(axis=1))[0]
    if rows_with_fg.size == 0:
        cy = 0 if orientation == "maxillary" else h - 1
        return (w // 2, cy), False

    widths = valid.sum(axis=1).astype(np.float64)
    max_w = widths[rows_with_fg].max()
    min_w = widths[rows_with_fg][widths[rows_with_fg] > 0].min()
    narrowing = 1.0 - _safe_div(min_w, max_w)

    apical_row = rows_with_fg[0] if orientation == "maxillary" else rows_with_fg[-1]

    if narrowing >= SILHOUETTE_NARROWING_MIN:
        # Trust the silhouette: apex is the centroid of the apical-most rows.
        n_rows = max(1, int(h * 0.05))
        if orientation == "maxillary":
            band_rows = rows_with_fg[:n_rows]
        else:
            band_rows = rows_with_fg[-n_rows:]
        cols = [np.where(valid[r])[0] for r in band_rows]
        cols = [c for c in cols if c.size]
        if cols:
            ax = int(np.mean([c.mean() for c in cols]))
            ay = int(np.mean(band_rows))
            return (ax, ay), True

    # Fallback proxy on the apical edge.
    apical_cols = np.where(valid[apical_row])[0]
    ax = int(apical_cols.mean()) if apical_cols.size else w // 2
    return (ax, int(apical_row)), False


def build_apical_roi(
    valid: np.ndarray, orientation: str, apex_trusted: bool
) -> np.ndarray:
    """
    Build the search ROI: a band spanning ``APICAL_ROI_FRACTION`` of the tooth
    height, intersected with the valid (non-transparent) region.

    * A real apex (``apex_trusted``) or ``FALLBACK_ROI_MODE == "apical_end"``
      → band at the apical end chosen by ``orientation``.
    * Otherwise (untrusted apex, ``FALLBACK_ROI_MODE == "centered"``)
      → band centred on the tooth.

    Guaranteed non-empty when any foreground exists.
    """
    h, w = valid.shape
    rows_with_fg = np.where(valid.any(axis=1))[0]
    roi = np.zeros((h, w), dtype=bool)
    if rows_with_fg.size == 0:
        return roi

    y0, y1 = int(rows_with_fg[0]), int(rows_with_fg[-1])
    tooth_h = max(1, y1 - y0 + 1)
    band = max(1, int(round(tooth_h * APICAL_ROI_FRACTION)))

    use_end = apex_trusted or FALLBACK_ROI_MODE == "apical_end"
    if use_end and orientation == "maxillary":      # apex up → top band
        roi[y0:y0 + band] = True
    elif use_end:                                   # apex down → bottom band
        roi[max(y0, y1 - band + 1):y1 + 1] = True
    else:                                           # centred mid-root band
        cy = (y0 + y1) // 2
        half = band // 2
        roi[max(y0, cy - half):min(y1 + 1, cy + half + 1)] = True

    roi &= valid
    if not roi.any():                    # degenerate guard
        roi = valid.copy()
    return roi


# ═════════════════════════════════════════════════════════════════════════════
# 3. Multi-feature per-pixel representation
# ═════════════════════════════════════════════════════════════════════════════
def build_gabor_bank() -> List[np.ndarray]:
    """Pre-compute the Gabor kernel bank (one kernel per orientation)."""
    bank = []
    for theta_deg in GABOR_ORIENTATIONS:
        theta = math.radians(theta_deg)
        kern = cv2.getGaborKernel(
            (GABOR_KSIZE, GABOR_KSIZE), GABOR_SIGMA, theta,
            GABOR_LAMBDA, GABOR_GAMMA, 0, ktype=cv2.CV_64F,
        )
        bank.append(kern)
    return bank


_GABOR_BANK = build_gabor_bank()


def compute_feature_maps(
    enhanced_f: np.ndarray,
    enhanced_u8: np.ndarray,
    apex: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Compute every dense (full-image) feature map used for clustering.

    Returns a dict of float maps, all the same H×W as the input:
      intensity, local_mean, local_std, gradient, laplacian,
      gabor_mean, gabor_max, lbp, entropy, x, y, apex_dist
    """
    h, w = enhanced_f.shape
    maps: Dict[str, np.ndarray] = {}

    maps["intensity"] = enhanced_f

    # Local mean / std via box filters (E[x^2] - E[x]^2).
    ksz = (LOCAL_WINDOW, LOCAL_WINDOW)
    mean = cv2.blur(enhanced_f, ksz)
    mean_sq = cv2.blur(enhanced_f ** 2, ksz)
    var = np.clip(mean_sq - mean ** 2, 0.0, None)
    maps["local_mean"] = mean
    maps["local_std"] = np.sqrt(var)

    # Gradient magnitude (Sobel) and Laplacian response.
    gx = cv2.Sobel(enhanced_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced_f, cv2.CV_64F, 0, 1, ksize=3)
    maps["gradient"] = np.sqrt(gx ** 2 + gy ** 2)
    maps["laplacian"] = np.abs(cv2.Laplacian(enhanced_f, cv2.CV_64F, ksize=3))

    # Gabor bank: aggregate magnitude across orientations.
    responses = [np.abs(cv2.filter2D(enhanced_f, cv2.CV_64F, k)) for k in _GABOR_BANK]
    responses = np.stack(responses, axis=0)
    maps["gabor_mean"] = responses.mean(axis=0)
    maps["gabor_max"] = responses.max(axis=0)

    # Local Binary Pattern (uniform) — normalised ordinal texture cue.
    lbp = local_binary_pattern(enhanced_u8, LBP_P, LBP_R, method="uniform")
    maps["lbp"] = lbp.astype(np.float64) / float(LBP_P + 2)

    # Local Shannon entropy (skimage rank filter) — texture heterogeneity.
    ent = local_entropy_filter(enhanced_u8, disk(ENTROPY_DISK)).astype(np.float64)
    maps["entropy"] = ent

    # Spatial coordinates, normalised to [0, 1].
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    maps["x"] = xx / max(w - 1, 1)
    maps["y"] = yy / max(h - 1, 1)

    # Distance from the (possibly proxy) apex, normalised by the image diagonal.
    ax, ay = apex
    diag = math.hypot(h, w)
    maps["apex_dist"] = np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2) / (diag + _EPS)

    return maps


def stack_roi_features(
    maps: Dict[str, np.ndarray], roi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Sample the feature maps at ROI pixels, normalise each feature to [0, 1],
    apply FEATURE_WEIGHTS, and stack into a (F, N) matrix for skfuzzy.

    Returns:
        features_weighted : (F, N) weighted, normalised matrix for FCM
        roi_index         : flat indices (into H*W) of the sampled pixels
        raw_by_name       : per-feature raw (un-normalised) ROI values, used
                            later for interpretable cluster ranking
    """
    roi_flat = roi.reshape(-1)
    roi_index = np.nonzero(roi_flat)[0]

    feat_rows: List[np.ndarray] = []
    raw_by_name: Dict[str, np.ndarray] = {}
    for name, weight in FEATURE_WEIGHTS.items():
        raw = maps[name].reshape(-1)[roi_index]
        raw_by_name[name] = raw
        feat_rows.append(_normalize01(raw) * weight)

    features_weighted = np.vstack(feat_rows)  # (F, N)
    return features_weighted, roi_index, raw_by_name


# ═════════════════════════════════════════════════════════════════════════════
# 4. Spatial Fuzzy C-Means
# ═════════════════════════════════════════════════════════════════════════════
def run_spatial_fcm(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run skfuzzy spatial FCM on a (F, N) feature matrix.

    Returns (hard_labels[N], centroids[C, F], fpc). Prints the Fuzzy Partition
    Coefficient and every cluster centroid, as required for transparency.
    """
    cntr, u, _u0, _d, _jm, _p, fpc = fuzz.cluster.cmeans(
        features, N_CLUSTERS, FUZZINESS,
        error=FCM_ERROR, maxiter=FCM_MAXITER, init=None, seed=RANDOM_SEED,
    )
    labels = np.argmax(u, axis=0)
    print(f"    FPC (Fuzzy Partition Coefficient): {fpc:.4f}")
    for i, c in enumerate(cntr):
        pretty = ", ".join(f"{v:.3f}" for v in c)
        print(f"    Cluster {i} centroid (weighted feat-space): [{pretty}]")
    return labels, cntr, float(fpc)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Lesion-candidate cluster selection  (NOT darkest-only)
# ═════════════════════════════════════════════════════════════════════════════
def rank_clusters(
    labels: np.ndarray, raw_by_name: Dict[str, np.ndarray]
) -> Tuple[List[int], np.ndarray]:
    """
    Score every cluster by darkness + texture + local contrast + apical
    proximity, then return the candidate cluster indices (best, plus any within
    CANDIDATE_CLUSTER_REL of the best) alongside the full score vector.
    """
    intensity = raw_by_name["intensity"]
    texture = 0.5 * _normalize01(raw_by_name["local_std"]) \
        + 0.5 * _normalize01(raw_by_name["entropy"])
    roi_mean_int = float(intensity.mean())
    apex_prox = 1.0 - _normalize01(raw_by_name["apex_dist"])

    w = CLUSTER_RANK_WEIGHTS
    scores = np.full(N_CLUSTERS, -np.inf)
    for k in range(N_CLUSTERS):
        sel = labels == k
        if not np.any(sel):
            continue
        darkness = 1.0 - float(intensity[sel].mean())      # dark → high
        tex = float(texture[sel].mean())
        contrast = _clip01(_safe_div(roi_mean_int - float(intensity[sel].mean()),
                                     CONTRAST_NORM))
        apex = float(apex_prox[sel].mean())
        scores[k] = (w["darkness"] * darkness + w["texture"] * tex
                     + w["contrast"] * contrast + w["apex"] * apex)

    order = np.argsort(scores)[::-1]
    best = order[0]
    candidates = [int(best)]
    for k in order[1:]:
        if len(candidates) >= N_CANDIDATE_CLUSTERS:
            break
        if scores[k] >= CANDIDATE_CLUSTER_REL * scores[best]:
            candidates.append(int(k))
    # Also fold in any extra cluster near the best even beyond the count cap.
    for k in order:
        if k not in candidates and np.isfinite(scores[k]) \
                and scores[k] >= CANDIDATE_CLUSTER_REL * scores[best]:
            candidates.append(int(k))
    return candidates, scores


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological opening → closing → (optional) hole-fill → small-object
    removal. Input/output are uint8 {0, 255}, same shape as the image.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    m = mask.copy()
    if MORPH_OPEN_ITER > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=MORPH_OPEN_ITER)
    if MORPH_CLOSE_ITER > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=MORPH_CLOSE_ITER)
    if FILL_HOLES:
        m = (ndi.binary_fill_holes(m > 0).astype(np.uint8)) * 255

    min_area = MIN_COMPONENT_AREA_FRAC * mask.size
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 6. Candidate-level feature extraction
# ═════════════════════════════════════════════════════════════════════════════
def _geometry_features(contour: np.ndarray, comp: np.ndarray) -> Dict[str, float]:
    """Geometric descriptors of a single candidate from its largest contour."""
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True))
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = float(bw * bh)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    # Fall back to the pixel count when the polygon area collapses.
    pix_area = float(int(comp.sum()))
    area = area if area > 1.0 else pix_area

    circularity = _clip01(_safe_div(4.0 * math.pi * area, perim ** 2))
    compactness = _safe_div(perim ** 2, area)  # ≥ 4π; higher = more irregular
    solidity = _clip01(_safe_div(area, hull_area))
    extent = _clip01(_safe_div(area, bbox_area))
    aspect_ratio = _safe_div(bw, bh)
    equiv_diam = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0

    # Ellipse-based shape stats (needs ≥ 5 contour points).
    if len(contour) >= 5:
        (_, _), (MA, ma), angle = cv2.fitEllipse(contour)
        major, minor = max(MA, ma), min(MA, ma)
        ecc = math.sqrt(1.0 - (minor / major) ** 2) if major > 0 else 0.0
        orientation = float(angle)
    else:
        major = minor = equiv_diam
        ecc = 0.0
        orientation = 0.0

    return {
        "area": area,
        "perimeter": perim,
        "equivalent_diameter": equiv_diam,
        "bbox_w": float(bw),
        "bbox_h": float(bh),
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": float(ecc),
        "major_axis": float(major),
        "minor_axis": float(minor),
        "orientation": orientation,
        "compactness": compactness,
        "bbox_x": float(x),
        "bbox_y": float(y),
    }


def _intensity_features(
    comp: np.ndarray, enhanced_f: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Intensity statistics of the candidate plus its contrast against a
    surrounding ring. Returns (features, ring_mask) — ring_mask is reused for
    the ring visualisation.
    """
    vals = enhanced_f[comp]
    mean_i = float(vals.mean()) if vals.size else 0.0
    min_i = float(vals.min()) if vals.size else 0.0
    max_i = float(vals.max()) if vals.size else 0.0
    std_i = float(vals.std()) if vals.size else 0.0

    dil = cv2.dilate(comp.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_PX, RING_PX)))
    ring = (dil.astype(bool)) & (~comp)
    ring_vals = enhanced_f[ring]
    ring_mean = float(ring_vals.mean()) if ring_vals.size else mean_i

    return (
        {
            "mean_intensity": mean_i,
            "min_intensity": min_i,
            "max_intensity": max_i,
            "std_intensity": std_i,
            "ring_mean_intensity": ring_mean,
            "contrast_vs_ring": ring_mean - mean_i,  # >0 → candidate darker
        },
        ring,
    )


def _glcm_texture_features(comp: np.ndarray, enhanced_u8: np.ndarray) -> Dict[str, float]:
    """GLCM (contrast/homogeneity/energy/correlation), LBP-hist and entropy."""
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return dict.fromkeys((
            "glcm_contrast", "glcm_homogeneity", "glcm_energy",
            "glcm_correlation", "lbp_mean", "lbp_std", "entropy"), 0.0)

    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = enhanced_u8[y0:y1, x0:x1]
    q = (patch.astype(np.float64) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    q = np.clip(q, 0, GLCM_LEVELS - 1)

    try:
        glcm = graycomatrix(q, distances=list(GLCM_DISTANCES),
                            angles=list(GLCM_ANGLES), levels=GLCM_LEVELS,
                            symmetric=True, normed=True)
        g_contrast = float(graycoprops(glcm, "contrast").mean())
        g_homog = float(graycoprops(glcm, "homogeneity").mean())
        g_energy = float(graycoprops(glcm, "energy").mean())
        g_corr = float(graycoprops(glcm, "correlation").mean())
    except Exception:
        g_contrast = g_homog = g_energy = g_corr = 0.0

    lbp = local_binary_pattern(patch, LBP_P, LBP_R, method="uniform")
    region_lbp = lbp[comp[y0:y1, x0:x1]]
    lbp_mean = float(region_lbp.mean()) if region_lbp.size else 0.0
    lbp_std = float(region_lbp.std()) if region_lbp.size else 0.0

    ent = float(shannon_entropy(enhanced_u8[comp])) if comp.any() else 0.0

    return {
        "glcm_contrast": g_contrast,
        "glcm_homogeneity": g_homog,
        "glcm_energy": g_energy,
        "glcm_correlation": g_corr,
        "lbp_mean": lbp_mean,
        "lbp_std": lbp_std,
        "entropy": ent,
    }


def _anatomical_features(
    comp: np.ndarray, roi: np.ndarray,
    apex: Tuple[int, int], axis_x: float, img_shape: Tuple[int, int],
) -> Dict[str, float]:
    """Apex distance, ROI overlap, border relationship and axis position."""
    h, w = img_shape
    M = cv2.moments(comp.astype(np.uint8), binaryImage=True)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cy_arr, cx_arr = np.where(comp)
        cx, cy = float(cx_arr.mean()), float(cy_arr.mean())

    ax, ay = apex
    diag = math.hypot(h, w)
    apex_dist = math.hypot(cx - ax, cy - ay)

    overlap = _safe_div(int((comp & roi).sum()), int(comp.sum()))

    # Border relationship: fraction of candidate pixels within the margin.
    border = np.zeros_like(comp)
    m = BORDER_MARGIN_PX
    border[:m, :] = border[-m:, :] = border[:, :m] = border[:, -m:] = True
    on_border = int((comp & border).sum())
    border_frac = _safe_div(on_border, int(comp.sum()))
    touches_border = 1.0 if on_border > 0 else 0.0

    dist_to_border = float(min(cx, cy, w - 1 - cx, h - 1 - cy))

    axis_offset = (cx - axis_x) / (w / 2.0 + _EPS)  # signed, ≈[-1, 1]

    return {
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "apex_distance": float(apex_dist),
        "apex_distance_norm": _clip01(apex_dist / (diag + _EPS)),
        "roi_overlap": _clip01(overlap),
        "border_touch_frac": _clip01(border_frac),
        "touches_border": touches_border,
        "dist_to_border": dist_to_border,
        "dist_to_border_norm": _clip01(dist_to_border / (0.5 * diag + _EPS)),
        "axis_offset_signed": float(axis_offset),
        "axis_offset_abs": _clip01(abs(axis_offset)),
    }


def extract_candidate_features(
    comp: np.ndarray, enhanced_f: np.ndarray, enhanced_u8: np.ndarray,
    roi: np.ndarray, apex: Tuple[int, int], axis_x: float,
) -> Optional[Tuple[Dict[str, float], np.ndarray]]:
    """
    Full descriptor for one connected candidate. Returns (features, ring_mask)
    or None if the component has no extractable contour.
    """
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)

    feats: Dict[str, float] = {}
    feats.update(_geometry_features(contour, comp))
    intensity_feats, ring = _intensity_features(comp, enhanced_f)
    feats.update(intensity_feats)
    feats.update(_glcm_texture_features(comp, enhanced_u8))
    feats.update(_anatomical_features(comp, roi, apex, axis_x,
                                      enhanced_f.shape))
    return feats, ring


# ═════════════════════════════════════════════════════════════════════════════
# 7. Feature-based candidate validation  (transparent weighted score)
# ═════════════════════════════════════════════════════════════════════════════
def _area_score(area_frac: float) -> float:
    """Plateau at 1 inside PREF_AREA_FRAC, ramp linearly to 0 at the hard gates."""
    lo_pref, hi_pref = PREF_AREA_FRAC
    if lo_pref <= area_frac <= hi_pref:
        return 1.0
    if area_frac < lo_pref:
        return _clip01(_safe_div(area_frac - MIN_AREA_FRAC, lo_pref - MIN_AREA_FRAC))
    return _clip01(_safe_div(MAX_AREA_FRAC - area_frac, MAX_AREA_FRAC - hi_pref))


def _shape_score(solidity: float, circularity: float) -> float:
    """High when solid and within the circularity band; low for thin/ragged blobs."""
    sol = _clip01(_safe_div(solidity - MIN_SOLIDITY, 1.0 - MIN_SOLIDITY))
    lo, hi = CIRCULARITY_RANGE
    circ = 1.0 if lo <= circularity <= hi else _clip01(
        1.0 - min(abs(circularity - lo), abs(circularity - hi)))
    return 0.5 * sol + 0.5 * circ


def score_candidate(
    feats: Dict[str, float], img_area: int
) -> Tuple[float, bool, List[str], Dict[str, float]]:
    """
    Combine sub-scores into a lesion likelihood ∈ [0, 1] and apply hard gates.

    lesion_score =  w_apex·apex + w_contrast·contrast + w_texture·texture
                  + w_area·area + w_shape·shape + w_roi·roi
                  − p_border·border_touch − p_fragment·fragmentation

    Returns (score, accepted, rejection_reasons, sub_scores).
    """
    area_frac = _safe_div(feats["area"], img_area)

    apex_s = 1.0 - feats["apex_distance_norm"]
    contrast_s = _clip01(_safe_div(feats["contrast_vs_ring"], CONTRAST_NORM))
    texture_s = _clip01(0.6 * _safe_div(feats["glcm_contrast"], TEXTURE_CONTRAST_NORM)
                        + 0.4 * _safe_div(feats["entropy"], math.log2(256)))
    area_s = _area_score(area_frac)
    shape_s = _shape_score(feats["solidity"], feats["circularity"])
    roi_s = feats["roi_overlap"]

    border_pen = feats["border_touch_frac"]
    # Fragmentation: low solidity and/or very elongated → high penalty.
    frag_pen = _clip01(0.5 * (1.0 - feats["solidity"])
                       + 0.5 * _clip01(feats["compactness"] / (8.0 * math.pi)))

    w, p = SCORE_WEIGHTS, SCORE_PENALTIES
    sub = {
        "apex": apex_s, "contrast": contrast_s, "texture": texture_s,
        "area": area_s, "shape": shape_s, "roi": roi_s,
        "border_penalty": border_pen, "fragment_penalty": frag_pen,
    }
    score = (w["apex"] * apex_s + w["contrast"] * contrast_s
             + w["texture"] * texture_s + w["area"] * area_s
             + w["shape"] * shape_s + w["roi"] * roi_s
             - p["border"] * border_pen - p["fragment"] * frag_pen)
    score = _clip01(score)

    # ── Hard gates → rejection reasons ───────────────────────────────────────
    reasons: List[str] = []
    if area_frac < MIN_AREA_FRAC:
        reasons.append("area_too_small")
    if area_frac > MAX_AREA_FRAC:
        reasons.append("area_too_large")
    if feats["roi_overlap"] < MIN_ROI_OVERLAP:
        reasons.append("insufficient_roi_overlap")
    if feats["border_touch_frac"] > BORDER_TOUCH_MAX:
        reasons.append("touches_border")
    if feats["contrast_vs_ring"] <= 0:
        reasons.append("not_darker_than_surround")
    if GATE_ON_SOLIDITY and feats["solidity"] < MIN_SOLIDITY:
        reasons.append("low_solidity")
    if score < SCORE_THRESHOLD:
        reasons.append("score_below_threshold")

    accepted = len(reasons) == 0
    return score, accepted, reasons, sub


# ═════════════════════════════════════════════════════════════════════════════
# 8. Candidate selection
# ═════════════════════════════════════════════════════════════════════════════
def select_final_lesions(
    candidates: List[dict], shape: Tuple[int, int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Build the final binary lesion mask from accepted candidates.

    SELECT_MODE == "best" keeps the single highest-scoring accepted candidate;
    "all" keeps every accepted candidate. If none is accepted, returns an empty
    mask — the pipeline never forces a lesion.
    """
    final = np.zeros(shape, dtype=np.uint8)
    accepted = [c for c in candidates if c["accepted"]]
    if not accepted:
        return final, []

    if SELECT_MODE == "best":
        chosen = [max(accepted, key=lambda c: c["lesion_score"])]
    else:
        chosen = accepted

    ids = []
    for c in chosen:
        final[c["mask"]] = 255
        ids.append(c["candidate_id"])
    return final, ids


# ═════════════════════════════════════════════════════════════════════════════
# 9. Optional boundary refinement
# ═════════════════════════════════════════════════════════════════════════════
def refine_chan_vese(
    enhanced_f: np.ndarray, init_mask: np.ndarray, roi: np.ndarray
) -> np.ndarray:
    """
    Morphological Chan–Vese refinement seeded by ``init_mask`` and confined to
    the apical ROI. Returns a refined uint8 {0,255} mask, or ``init_mask``
    unchanged when refinement is disabled/unavailable or the seed is empty.
    """
    if not (ENABLE_REFINEMENT and _HAS_CHAN_VESE) or init_mask.max() == 0:
        return init_mask
    try:
        ls = morphological_chan_vese(
            enhanced_f, num_iter=CHAN_VESE_ITER,
            init_level_set=(init_mask > 0).astype(np.uint8),
            smoothing=CHAN_VESE_SMOOTHING,
            lambda1=CHAN_VESE_LAMBDA1, lambda2=CHAN_VESE_LAMBDA2,
        )
        refined = (ls.astype(bool) & roi).astype(np.uint8) * 255
        return refined if refined.max() > 0 else init_mask
    except Exception:
        return init_mask


# ═════════════════════════════════════════════════════════════════════════════
# 10. Visualisation helpers
# ═════════════════════════════════════════════════════════════════════════════
def _gray_to_bgr(enhanced_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(enhanced_u8, cv2.COLOR_GRAY2BGR)


def save_roi_vis(enhanced_u8, roi, apex, axis_x, path) -> None:
    vis = _gray_to_bgr(enhanced_u8)
    tint = vis.copy()
    tint[roi] = (0, 180, 0)
    vis = cv2.addWeighted(vis, 0.7, tint, 0.3, 0)
    cnts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
    h = vis.shape[0]
    cv2.line(vis, (int(axis_x), 0), (int(axis_x), h - 1), (255, 128, 0), 1)
    cv2.drawMarker(vis, (int(apex[0]), int(apex[1])), (0, 0, 255),
                   cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.imwrite(path, vis)


def save_cluster_map(labels, roi_index, shape, path) -> None:
    lab_img = np.full(shape[0] * shape[1], 0, dtype=np.uint8)
    lab_img[roi_index] = (labels + 1) * (255 // (N_CLUSTERS + 1))
    lab_img = lab_img.reshape(shape)
    colored = cv2.applyColorMap(lab_img, cv2.COLORMAP_JET)
    colored[lab_img == 0] = 0
    cv2.imwrite(path, colored)


def save_candidate_vis(enhanced_u8, candidates, path, annotate_score=False) -> None:
    vis = _gray_to_bgr(enhanced_u8)
    for c in candidates:
        color = (0, 255, 0) if c["accepted"] else (0, 0, 255)
        cnts, _ = cv2.findContours(c["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
        cx = int(c["features"]["centroid_x"])
        cy = int(c["features"]["centroid_y"])
        label = f"#{c['candidate_id']}"
        if annotate_score:
            label += f" {c['lesion_score']:.2f}"
        cv2.putText(vis, label, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(path, vis)


def save_ring_vis(enhanced_u8, candidates, path) -> None:
    vis = _gray_to_bgr(enhanced_u8)
    for c in candidates:
        vis[c["ring"]] = (0, 255, 255)   # ring → yellow
        vis[c["mask"]] = (255, 0, 0)     # candidate → blue
    cv2.imwrite(path, vis)


def save_overlay(enhanced_u8, final_mask, path) -> None:
    vis = _gray_to_bgr(enhanced_u8)
    if final_mask.max() > 0:
        tint = vis.copy()
        tint[final_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    cv2.imwrite(path, vis)


# ═════════════════════════════════════════════════════════════════════════════
# Orchestration for a single image
# ═════════════════════════════════════════════════════════════════════════════
def process_image(path: str, filename: str) -> Tuple[dict, List[dict]]:
    """
    Run the full pipeline on one image. Returns (image_summary, candidate_rows).

    Never raises for expected data problems (unreadable / tiny / empty ROI);
    those are reported through the summary's ``status`` / ``failure_reason``.
    """
    stem = os.path.splitext(filename)[0]
    summary = {
        "filename": filename, "num_candidates": 0, "num_accepted": 0,
        "final_candidate_ids": "", "final_area": 0, "final_centroid": "",
        "fpc": float("nan"), "orientation": "", "apex_trusted": False,
        "status": "ok", "failure_reason": "",
    }

    loaded = load_image(path)
    if loaded is None:
        summary.update(status="failed", failure_reason="unreadable_image")
        print(f"  [WARN] Could not read {filename}")
        return summary, []

    gray, valid = loaded
    h, w = gray.shape
    if h < 16 or w < 16 or valid.sum() < N_CLUSTERS * 20:
        summary.update(status="failed", failure_reason="image_too_small_or_empty")
        print(f"  [WARN] {filename}: too small / too little foreground")
        return summary, []

    # 1. Preprocess.
    enhanced_u8, enhanced_f = preprocess_image(gray, valid)
    cv2.imwrite(_out_path("enhanced", stem, "enhanced"), enhanced_u8)

    # 2. Orientation + apex + ROI.
    orientation = detect_orientation(enhanced_f, valid)
    apex, apex_trusted = estimate_apex(valid, orientation)
    roi = build_apical_roi(valid, orientation, apex_trusted)
    axis_x = w / 2.0
    # With an untrusted apex and a centred ROI, distance-from-apex is most
    # meaningful relative to the ROI centre rather than an arbitrary edge point.
    if not apex_trusted and FALLBACK_ROI_MODE == "centered":
        ys_roi, xs_roi = np.where(roi)
        if ys_roi.size:
            apex = (int(xs_roi.mean()), int(ys_roi.mean()))
    summary.update(orientation=orientation, apex_trusted=bool(apex_trusted))

    cv2.imwrite(_out_path("roi_mask", stem, "roi_mask"),
                (roi.astype(np.uint8) * 255))
    save_roi_vis(enhanced_u8, roi, apex, axis_x, _out_path("roi_vis", stem, "roi_vis"))

    if roi.sum() < N_CLUSTERS * 20:
        summary.update(status="failed", failure_reason="empty_roi")
        cv2.imwrite(_out_path("final_mask", stem, "final_mask"),
                    np.zeros((h, w), np.uint8))
        print(f"  [WARN] {filename}: ROI too small after restriction")
        return summary, []

    # 3. Feature maps + ROI feature matrix.
    maps = compute_feature_maps(enhanced_f, enhanced_u8, apex)
    features, roi_index, raw_by_name = stack_roi_features(maps, roi)

    # 4. Spatial FCM.
    labels, _cntr, fpc = run_spatial_fcm(features)
    summary["fpc"] = fpc
    save_cluster_map(labels, roi_index, (h, w), _out_path("cluster_map", stem, "cluster_map"))

    # 5. Candidate cluster(s) → raw candidate mask.
    cand_clusters, _cscores = rank_clusters(labels, raw_by_name)
    cand_flat = np.zeros(h * w, dtype=np.uint8)
    sel = np.isin(labels, cand_clusters)
    cand_flat[roi_index[sel]] = 255
    raw_mask = cand_flat.reshape(h, w)
    cv2.imwrite(_out_path("candidate_raw", stem, "candidate_raw"), raw_mask)

    cleaned = clean_mask(raw_mask)
    cleaned &= (roi.astype(np.uint8) * 255)   # keep candidates inside the ROI
    cv2.imwrite(_out_path("candidate_clean", stem, "candidate_clean"), cleaned)

    # 6-7. Per-component features + scoring.
    n_lab, lab_img = cv2.connectedComponents((cleaned > 0).astype(np.uint8), 8)
    img_area = h * w
    candidates: List[dict] = []
    rows: List[dict] = []
    for cid in range(1, n_lab):
        comp = lab_img == cid
        result = extract_candidate_features(comp, enhanced_f, enhanced_u8,
                                            roi, apex, axis_x)
        if result is None:
            continue
        feats, ring = result
        score, accepted, reasons, sub = score_candidate(feats, img_area)
        cand = {
            "candidate_id": cid, "mask": comp, "ring": ring,
            "features": feats, "lesion_score": score,
            "accepted": accepted, "reasons": reasons, "sub_scores": sub,
            "is_final": False,
        }
        candidates.append(cand)

    # 8. Final selection.
    final_mask, final_ids = select_final_lesions(candidates, (h, w))
    for c in candidates:
        c["is_final"] = c["candidate_id"] in final_ids

    # 9. Optional refinement (kept inside the ROI).
    final_mask = refine_chan_vese(enhanced_f, final_mask, roi)

    # 10. Remaining visual outputs.
    cv2.imwrite(_out_path("final_mask", stem, "final_mask"), final_mask)
    save_overlay(enhanced_u8, final_mask, _out_path("overlay", stem, "overlay"))
    save_candidate_vis(enhanced_u8, candidates, _out_path("contours", stem, "contours"))
    save_candidate_vis(enhanced_u8, candidates, _out_path("scores", stem, "scores"),
                       annotate_score=True)
    save_ring_vis(enhanced_u8, candidates, _out_path("ring", stem, "ring"))

    # 11. CSV rows for this image.
    for c in candidates:
        row = {"filename": filename, "candidate_id": c["candidate_id"]}
        row.update(c["features"])
        row.update({f"score_{k}": v for k, v in c["sub_scores"].items()})
        row["lesion_score"] = c["lesion_score"]
        row["status"] = "accepted" if c["accepted"] else "rejected"
        row["rejection_reasons"] = ";".join(c["reasons"])
        row["is_final_lesion"] = c["is_final"]
        rows.append(row)

    # Summary bookkeeping.
    ys, xs = np.where(final_mask > 0)
    summary["num_candidates"] = len(candidates)
    summary["num_accepted"] = sum(1 for c in candidates if c["accepted"])
    summary["final_candidate_ids"] = ";".join(str(i) for i in final_ids)
    summary["final_area"] = int((final_mask > 0).sum())
    summary["final_centroid"] = (
        f"({xs.mean():.1f},{ys.mean():.1f})" if xs.size else "")

    print(f"  {filename}: orient={orientation} apex_trusted={apex_trusted} "
          f"cand={len(candidates)} accepted={summary['num_accepted']} "
          f"final={summary['final_candidate_ids'] or 'NONE'} "
          f"area={summary['final_area']}")
    return summary, rows


# ═════════════════════════════════════════════════════════════════════════════
# 13. Evaluation (kept separate from the segmentation logic)
# ═════════════════════════════════════════════════════════════════════════════
def _match_gt(stem: str) -> Optional[str]:
    """Find the ground-truth mask whose name starts with ``{stem}_mask``."""
    if not os.path.isdir(MASK_DIR):
        return None
    for f in os.listdir(MASK_DIR):
        if f.lower().endswith(SUPPORTED_EXTS) and f.startswith(f"{stem}_mask"):
            return os.path.join(MASK_DIR, f)
    return None


def _hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two binary masks (NaN if either empty)."""
    pc = np.argwhere(pred > 0)
    gc = np.argwhere(gt > 0)
    if pc.size == 0 or gc.size == 0:
        return float("nan")
    return float(max(directed_hausdorff(pc, gc)[0], directed_hausdorff(gc, pc)[0]))


def _pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Dice, IoU, precision, recall, specificity, accuracy, Hausdorff."""
    p = (pred > 127).astype(np.uint8)
    g = (gt > 127).astype(np.uint8)
    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))
    return {
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "hausdorff": _hausdorff(p, g),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def run_evaluation() -> None:
    """Evaluate every predicted final mask against its ground-truth mask."""
    print("\n" + "=" * 70)
    print("  EVALUATION vs ground-truth masks")
    print("=" * 70)
    pred_dir = os.path.join(OUTPUT_DIR, SUBDIRS["final_mask"])
    if not os.path.isdir(MASK_DIR):
        print(f"  [WARN] Ground-truth dir not found: {MASK_DIR} — skipping.")
        return

    rows: List[dict] = []
    for f in sorted(os.listdir(pred_dir)):
        if not f.lower().endswith(SUPPORTED_EXTS):
            continue
        stem = f[len("final_mask_"):] if f.startswith("final_mask_") else f
        stem = os.path.splitext(stem)[0]
        gt_path = _match_gt(stem)
        if gt_path is None:
            continue
        pred = cv2.imread(os.path.join(pred_dir, f), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        m = _pixel_metrics(pred, gt)
        m["filename"] = stem
        rows.append(m)
        print(f"  {stem:28s} Dice={m['dice']:.3f} IoU={m['iou']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f}")

    if not rows:
        print("  [WARN] No prediction/ground-truth pairs matched.")
        return

    df = pd.DataFrame(rows)
    per_image = os.path.join(OUTPUT_DIR, SUBDIRS["evaluation"], "per_image_metrics.csv")
    df.to_csv(per_image, index=False)

    metric_cols = ["dice", "iou", "precision", "recall",
                   "specificity", "accuracy", "hausdorff"]
    summary = pd.DataFrame({
        "metric": metric_cols,
        "mean": [df[c].mean(skipna=True) for c in metric_cols],
        "std": [df[c].std(skipna=True) for c in metric_cols],
    })
    summary_path = os.path.join(OUTPUT_DIR, SUBDIRS["evaluation"], "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)

    print("\n  Mean over %d matched image(s):" % len(df))
    for c in metric_cols:
        print(f"    {c:12s}: {df[c].mean(skipna=True):.4f} "
              f"± {df[c].std(skipna=True):.4f}")
    print(f"  -> {per_image}")
    print(f"  -> {summary_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("  Anatomically-constrained multi-feature spatial FCM")
    print("  lesion-candidate segmentation + feature-based validation")
    print("=" * 70)
    print(f"  Input      : {INPUT_DIR}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Orientation: {TOOTH_ORIENTATION}   ROI frac: {APICAL_ROI_FRACTION}")
    print(f"  Clusters   : {N_CLUSTERS}   Select: {SELECT_MODE}   "
          f"Score thr: {SCORE_THRESHOLD}")
    print(f"  Chan-Vese  : {'on' if (ENABLE_REFINEMENT and _HAS_CHAN_VESE) else 'off'}")
    print("=" * 70)

    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return
    _ensure_dirs()

    files = sorted(f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(SUPPORTED_EXTS))
    if not files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return
    print(f"\nFound {len(files)} image(s).\n")

    all_candidate_rows: List[dict] = []
    all_summaries: List[dict] = []
    for filename in files:
        try:
            summary, rows = process_image(os.path.join(INPUT_DIR, filename), filename)
        except Exception as exc:  # never let one image kill the batch
            summary = {"filename": filename, "status": "failed",
                       "failure_reason": f"exception:{type(exc).__name__}:{exc}"}
            rows = []
            print(f"  [ERROR] {filename}: {type(exc).__name__}: {exc}")
        all_summaries.append(summary)
        all_candidate_rows.extend(rows)

    # CSV reports.
    reports = os.path.join(OUTPUT_DIR, SUBDIRS["reports"])
    cand_csv = os.path.join(reports, "candidates.csv")
    summ_csv = os.path.join(reports, "image_summary.csv")
    pd.DataFrame(all_candidate_rows).to_csv(cand_csv, index=False)
    pd.DataFrame(all_summaries).to_csv(summ_csv, index=False)
    print(f"\n  -> Candidate report : {cand_csv}")
    print(f"  -> Image summary    : {summ_csv}")

    if RUN_EVALUATION:
        run_evaluation()

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
