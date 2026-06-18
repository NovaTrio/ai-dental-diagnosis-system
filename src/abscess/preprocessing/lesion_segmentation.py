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

# Only process the texture-removed images (not the gabor outputs)
INPUT_PREFIX = ''


# ─────────────────────────────────────────────────────────────────────────────
# Helper: identify which cluster index corresponds to the lesion
#         Lesions in periapical X-rays appear as dark (low intensity) regions.
#         We pick the cluster whose centroid has the LOWEST pixel intensity.
# ─────────────────────────────────────────────────────────────────────────────
def _lesion_cluster_index(cntr: np.ndarray) -> int:
    """Return the index of the cluster with the lowest centroid value."""
    return int(np.argmin(cntr.flatten()))


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
    print(f"  - Image size : {original_shape[1]} x {original_shape[0]} px")

    # ── Step 3: Reshape the Image ─────────────────────────────────────────────
    # FCM expects data as (features, samples) → shape (1, H*W)
    pixels = gray_image.reshape(-1, 1)          # shape: (N, 1)

    # ── Step 4: Apply Fuzzy C-Means Clustering ────────────────────────────────
    print(f"  - Running FCM  (clusters={N_CLUSTERS}, m={FUZZINESS}) …")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        pixels.T,           # shape: (1, N)  — features × samples
        N_CLUSTERS,
        FUZZINESS,
        error=FCM_ERROR,
        maxiter=FCM_MAXITER,
        init=None
    )
    print(f"  - FPC (Fuzzy Partition Coefficient): {fpc:.4f}")
    print(f"  - Cluster centroids (pixel intensity): "
          f"{[f'{c[0]:.4f}' for c in cntr]}")

    # Hard assignment: each pixel → cluster with highest membership
    cluster_membership = np.argmax(u, axis=0)   # shape: (N,)

    # ── Step 5: Reshape Clustered Data Back to Image Shape ────────────────────
    segmented_image = cluster_membership.reshape(original_shape)

    # ── (Step 6 SKIPPED — no color assignment) ────────────────────────────────

    # ── Create lesion mask ────────────────────────────────────────────────────
    # Identify which cluster is the lesion (lowest intensity centroid)
    lesion_idx = _lesion_cluster_index(cntr)
    print(f"  - Lesion cluster index : {lesion_idx}  "
          f"(centroid ~= {cntr[lesion_idx][0]:.4f})")

    lesion_mask = (segmented_image == lesion_idx).astype(np.uint8) * 255

    # ── Remove transparent background from lesion mask ────────────────────────
    if len(img_original.shape) == 3 and img_original.shape[2] == 4:
        alpha = img_original[:, :, 3]
        lesion_mask[alpha == 0] = 0

    # ── Save outputs ──────────────────────────────────────────────────────────
    # 1. Full cluster-label map (grayscale, values 0 / 85 / 170 … scaled)
    label_map = (segmented_image * (255 // (N_CLUSTERS - 1))).astype(np.uint8)
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
