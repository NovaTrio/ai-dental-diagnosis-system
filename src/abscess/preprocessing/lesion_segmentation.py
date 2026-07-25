"""
lesion_segmentation.py
──────────────────────
Performs lesion segmentation on the output images from texture_enhance.py
using Fuzzy C-Means (FCM) clustering.

Pipeline (based on GeeksforGeeks FCM Image Segmentation guide):
  Step 1: Import Necessary Libraries
  Step 2: Load and Preprocess the Image  -> convert to grayscale
  Step 3: Reshape the Image              -> flatten to 1-D pixel array
  Step 4: Apply Fuzzy C-Means Clustering -> skfuzzy cmeans
  Step 5: Reshape the Clustered Data Back to Image Shape
  (Step 6: Assign Colors — SKIPPED as requested)
  Step 7: Save the Segmented Image

Input  : texture_removed_*.png  from  data/abscess/raw/texture_output/
Output : lesion_seg_*.png        into  data/abscess/raw/lesion_segmented/
"""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Import Necessary Libraries
# ─────────────────────────────────────────────────────────────────────────────
import os
import cv2
import numpy as np
import skfuzzy as fuzz

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR  = '../../../data/abscess/raw/croun_crops'
OUTPUT_DIR = '../../../data/abscess/raw/lesion_segmented'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FCM parameters
N_CLUSTERS    = 3    # Number of fuzzy clusters (background / healthy / lesion)
FUZZINESS     = 2      # Fuzziness exponent  m  (standard value = 2)
FCM_ERROR     = 0.005  # Convergence threshold
FCM_MAXITER   = 1000   # Maximum number of iterations
RANDOM_SEED   = 42     # Seeds cmeans() so results are reproducible run-to-run

# Spatially-aware clustering: cluster on [intensity, x, y] instead of intensity
# alone, so two similarly-dark but spatially unrelated regions (e.g. the true
# periapical lesion vs. an unrelated patch of trabecular bone elsewhere in the
# crop) are less likely to be pulled into the same "lesion" cluster. Intensity
# stays the dominant axis; position only acts as a cohesion/tie-break term.
INTENSITY_WEIGHT      = 1.0
SPATIAL_WEIGHT        = 0.0
MIN_FOREGROUND_PIXELS = N_CLUSTERS * 10   # guard for near-empty/degenerate crops

# Only process the texture-removed images (not the gabor outputs)
INPUT_PREFIX = ''


# ─────────────────────────────────────────────────────────────────────────────
# Helper: identify which cluster index corresponds to the lesion
#         Lesions in periapical X-rays appear as dark (low intensity) regions.
#         We pick the cluster whose centroid has the LOWEST pixel intensity.
# ─────────────────────────────────────────────────────────────────────────────
def _lesion_cluster_index(intensity_centroids: np.ndarray) -> int:
    """Return the index of the cluster with the lowest intensity centroid.

    `intensity_centroids` is the intensity column of `cntr` (shape (c,)) —
    with the spatial features added, `cntr` itself is (c, 3), so callers must
    pass `cntr[:, 0]`, not the raw `cntr` array.
    """
    return int(np.argmin(intensity_centroids))


# ─────────────────────────────────────────────────────────────────────────────
# Core segmentation function
# ─────────────────────────────────────────────────────────────────────────────
def segment_lesion(image_path: str, filename: str) -> None:
    print(f"\nProcessing: {filename}")

    # ── Step 2: Load and Preprocess the Image ────────────────────────────────
    img_original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_original is None:
        print(f"  [WARN] Could not read {image_path} — skipping.")
        return

    # Convert to grayscale (handle RGBA, RGB, or already-gray images)
    if len(img_original.shape) == 2:
        gray_image = img_original.astype(np.float64) / 255.0
    elif img_original.shape[2] == 4:          # RGBA
        rgb = img_original[:, :, :3]
        gray_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:                                      # BGR
        gray_image = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    original_shape = gray_image.shape
    h, w = original_shape
    print(f"  - Image size : {w} x {h} px")

    # ── Foreground mask ────────────────────────────────────────────────────────
    # Exclude alpha=0 background pixels from clustering so pure-black corner
    # pixels never drag the "darkest" centroid toward 0.
    if len(img_original.shape) == 3 and img_original.shape[2] == 4:
        alpha = img_original[:, :, 3]
        fg_mask = alpha > 0
    else:
        alpha = None
        fg_mask = np.ones((h, w), dtype=bool)

    fg_idx = np.nonzero(fg_mask.reshape(-1))[0]
    if fg_idx.size < MIN_FOREGROUND_PIXELS:
        print(f"  [WARN] Only {fg_idx.size} foreground px in {filename} — skipping.")
        return

    # ── Step 3: Reshape the Image into a [intensity, x, y] feature vector ───────
    yy, xx = np.mgrid[0:h, 0:w]
    x_norm = (xx / max(w - 1, 1)).reshape(-1)
    y_norm = (yy / max(h - 1, 1)).reshape(-1)
    intensity_flat = gray_image.reshape(-1)

    features = np.vstack([
        intensity_flat[fg_idx] * INTENSITY_WEIGHT,
        x_norm[fg_idx]         * SPATIAL_WEIGHT,
        y_norm[fg_idx]         * SPATIAL_WEIGHT,
    ])                                              # shape: (3, N_fg)

    # ── Step 4: Apply Fuzzy C-Means Clustering ────────────────────────────────
    print(f"  - Running FCM  (clusters={N_CLUSTERS}, m={FUZZINESS}, "
          f"spatial_weight={SPATIAL_WEIGHT}) …")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        features,           # shape: (3, N_fg)  — features × samples
        N_CLUSTERS,
        FUZZINESS,
        error=FCM_ERROR,
        maxiter=FCM_MAXITER,
        init=None,
        seed=RANDOM_SEED
    )
    print(f"  - FPC (Fuzzy Partition Coefficient): {fpc:.4f}")
    cntr_intensity = cntr[:, 0] / INTENSITY_WEIGHT
    print(f"  - Cluster centroids (intensity, x, y): "
          f"{[f'({c[0]/INTENSITY_WEIGHT:.3f}, {c[1]/SPATIAL_WEIGHT:.3f}, {c[2]/SPATIAL_WEIGHT:.3f})' for c in cntr]}")

    # Hard assignment: each foreground pixel → cluster with highest membership
    cluster_membership_fg = np.argmax(u, axis=0)   # shape: (N_fg,)

    # ── Step 5: Reshape Clustered Data Back to Image Shape ────────────────────
    # Background pixels get a sentinel label (N_CLUSTERS) that can never be
    # picked as the lesion cluster by _lesion_cluster_index.
    cluster_membership_full = np.full(h * w, N_CLUSTERS, dtype=np.int64)
    cluster_membership_full[fg_idx] = cluster_membership_fg
    segmented_image = cluster_membership_full.reshape(original_shape)

    # ── (Step 6 SKIPPED — no color assignment) ────────────────────────────────

    # ── Create lesion mask ────────────────────────────────────────────────────
    # Identify which cluster is the lesion (lowest intensity centroid)
    lesion_idx = _lesion_cluster_index(cntr_intensity)
    print(f"  - Lesion cluster index : {lesion_idx}  "
          f"(centroid ~= {cntr_intensity[lesion_idx]:.4f})")

    lesion_mask = (segmented_image == lesion_idx).astype(np.uint8) * 255

    # ── Remove transparent background from lesion mask ────────────────────────
    if alpha is not None:
        lesion_mask[alpha == 0] = 0

    # ── Save outputs ──────────────────────────────────────────────────────────
    # 1. Full cluster-label map (grayscale, one shade per cluster; background
    #    sentinel stays 0). Debug artifact only — no downstream reader depends
    #    on its exact gray levels.
    label_map = np.zeros((h, w), dtype=np.uint8)
    for k in range(N_CLUSTERS):
        label_map[segmented_image == k] = int(round((k + 1) * 255 / N_CLUSTERS))
    label_path = os.path.join(OUTPUT_DIR, f"segmap_{filename}")
    cv2.imwrite(label_path, label_map)

    # 2. Binary lesion mask (white = lesion, black = background)
    mask_path = os.path.join(OUTPUT_DIR, f"lesion_mask_{filename}")
    cv2.imwrite(mask_path, lesion_mask)

    # 3. Lesion overlay — original grayscale with lesion region highlighted
    gray_8u = (gray_image * 255).astype(np.uint8)
    overlay  = cv2.cvtColor(gray_8u, cv2.COLOR_GRAY2BGR)
    # Draw lesion region in a distinct highlight (bright cyan)
    overlay[lesion_mask == 255] = [255, 255, 0]   # BGR → yellow highlight
    overlay_path = os.path.join(OUTPUT_DIR, f"lesion_overlay_{filename}")
    cv2.imwrite(overlay_path, overlay)

    print(f"  -> Cluster label map saved  : {label_path}")
    print(f"  -> Lesion binary mask saved : {mask_path}")
    print(f"  -> Lesion overlay saved     : {overlay_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  Lesion Segmentation via Fuzzy C-Means (FCM) Clustering")
    print("=" * 65)
    print(f"  Input  directory : {INPUT_DIR}")
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  N clusters       : {N_CLUSTERS}")
    print("=" * 65)

    supported_exts = ('.png', '.jpg', '.jpeg')

    # Collect only texture_removed_* images from the input directory
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().startswith(INPUT_PREFIX) and f.lower().endswith(supported_exts)
    ]

    if not files:
        print(f"\n[ERROR] No '{INPUT_PREFIX}*' images found in:\n  {INPUT_DIR}")
        print("        Run texture_enhance.py first to generate the input images.")
    else:
        print(f"\nFound {len(files)} image(s) to process.\n")
        for filename in sorted(files):
            image_path = os.path.join(INPUT_DIR, filename)
            segment_lesion(image_path, filename)


    print("\n" + "=" * 65)
    print("  Segmentation complete!")
    print("=" * 65)
