"""
lesion_segmentation.py
──────────────────────
Performs lesion segmentation on the extracted features using a highly optimized,
vectorized FCM-FRWS (Fuzzy C-Means with Feature Range Weighting Scheme).
Enforces exact (H, W) shapes strictly, applying mathematical heuristics to precisely 
identify texturally active, dark abscess regions.
"""

import os
import cv2
import numpy as np

# ======================================================
# CONFIGURATION
# ======================================================
FEATURE_DIR = "../../../data/abscess/raw/features"
IMAGE_DIR   = "../../../data/abscess/raw/texture_output"
OUTPUT_DIR  = "../../../data/abscess/raw/lesion_segmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FCM parameters
NUM_CLUSTERS = 3        # lesion, healthy, background
FUZZY_M = 2.0
MAX_ITER = 150
ERROR = 0.005

# ======================================================
# FEATURE WEIGHTS (FRWS)
# ======================================================
# Order matching feature_extraction.py: 
# [0: Intensity, 1: Local Mean, 2: Local Variance, 3: Gradient, 4: Entropy, 5: LBP]
feature_weights = np.array([
    0.8,   # intensity
    0.6,   # local mean
    0.7,   # local variance
    1.2,   # gradient (IMPORTANT for edges)
    1.3,   # entropy (VERY IMPORTANT for texture lesions)
    0.9    # LBP
])

# Normalize weights
feature_weights = feature_weights / np.sum(feature_weights)


def fcm_frws(data, n_clusters=3, m=2, max_iter=100, error=1e-5):
    """
    Fully vectorized Weighted Fuzzy C-Means implementation.
    """
    N, D = data.shape
    
    # Apply global feature weights to the entire dataset
    weighted_data = data * feature_weights

    # Random membership initialization
    U = np.random.dirichlet(np.ones(n_clusters), size=N)

    for iteration in range(max_iter):
        U_old = U.copy()
        centers = np.zeros((n_clusters, D))

        # Update centroids
        for c in range(n_clusters):
            um = U[:, c] ** m
            centers[c] = np.sum(um[:, np.newaxis] * weighted_data, axis=0) / (np.sum(um) + 1e-8)

        # Vectorized membership update
        dists = np.zeros((N, n_clusters))
        for c in range(n_clusters):
            # Compute Euclidean distance for all pixels to centroid `c`
            dists[:, c] = np.linalg.norm(weighted_data - centers[c], axis=1) + 1e-8
            
        for c in range(n_clusters):
            # Update U matrix based on distances
            U[:, c] = 1.0 / np.sum((dists[:, c:c+1] / dists) ** (2 / (m - 1)), axis=1)

        # Convergence check
        if np.linalg.norm(U - U_old) < error:
            print(f"  - Converged at iteration {iteration}")
            break

    return U, centers


def segment_lesion(feature_path, filename):
    print(f"\nProcessing: {filename}")
    
    # Load features safely (structured dictionary)
    try:
        data = np.load(feature_path, allow_pickle=True).item()
    except Exception as e:
        print(f"  [ERROR] Failed to load {feature_path}: {e}")
        return

    features = data.get("features")
    shape = data.get("shape")
    
    if features is None or shape is None:
        print("  [ERROR] Invalid feature file structure. Missing 'features' or 'shape' keys.")
        return
        
    H, W = shape
    print(f"  - Loaded Matrix Size: {features.shape} for Image Shape {W}x{H}")
    
    # Normalizing features globally [0, 1] isn't strictly necessary as they were 
    # individually normalized in feature_extraction.py, but we apply a minor clip.
    features = np.clip(features, 0, 1)
    
    # Run FCM-FRWS
    print(f"  - Running FCM-FRWS (clusters={NUM_CLUSTERS}, m={FUZZY_M})...")
    U, centers = fcm_frws(features, n_clusters=NUM_CLUSTERS, m=FUZZY_M, max_iter=MAX_ITER, error=ERROR)
    
    # Hard assignment
    labels = np.argmax(U, axis=1)
    
    # ==================================================
    # CLUSTER SELECTION LOGIC
    # Lesion = Texturally active (High Entropy + Gradient) AND Dark (Low Intensity)
    # Score = Mean(Gradient) + Mean(Entropy) - Mean(Intensity)
    # ==================================================
    cluster_scores = []
    print("  - Cluster Analysis:")
    
    for c in range(NUM_CLUSTERS):
        mask_c = (labels == c)
        if not np.any(mask_c):
            cluster_scores.append(-np.inf)
            continue
            
        mean_intensity = np.mean(features[mask_c][:, 0])
        mean_gradient  = np.mean(features[mask_c][:, 3])
        mean_entropy   = np.mean(features[mask_c][:, 4])
        
        score = mean_gradient + mean_entropy - mean_intensity
        cluster_scores.append(score)
        print(f"    Cluster {c} -> Int: {mean_intensity:.3f}, Grad: {mean_gradient:.3f}, Ent: {mean_entropy:.3f} | Score: {score:.3f}")
        
    lesion_cluster = np.argmax(cluster_scores)
    print(f"  - Identified Lesion Cluster: {lesion_cluster}")
    
    # Create mask and explicitly reshape using original H, W
    mask_1d = (labels == lesion_cluster).astype(np.uint8) * 255
    mask_img = mask_1d.reshape((H, W))
    segmented_image = labels.reshape((H, W))
    
    # ==================================================
    # POST PROCESSING (Morphology)
    # ==================================================
    kernel = np.ones((5, 5), np.uint8)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
    
    # ==================================================
    # SAVE OUTPUTS (Mask, Segmap, Overlay)
    # ==================================================
    base_name = filename.replace(".npy", "")
    
    # 1. Binary lesion mask
    mask_path = os.path.join(OUTPUT_DIR, f"lesion_mask_{base_name}.png")
    cv2.imwrite(mask_path, mask_img)
    
    # 2. Cluster label map (visualize the 3 zones distinctly)
    label_map = (segmented_image * (255 // (NUM_CLUSTERS - 1))).astype(np.uint8)
    label_path = os.path.join(OUTPUT_DIR, f"segmap_{base_name}.png")
    cv2.imwrite(label_path, label_map)
    
    # 3. Lesion overlay (needs original grayscale image)
    orig_img_path = os.path.join(IMAGE_DIR, f"{base_name}.png")
    if not os.path.exists(orig_img_path):
        orig_img_path = os.path.join(IMAGE_DIR, f"{base_name}.jpg")
        
    if os.path.exists(orig_img_path):
        orig_img = cv2.imread(orig_img_path, cv2.IMREAD_UNCHANGED)
        
        # Enforce grayscale overlay base
        if len(orig_img.shape) == 3:
            if orig_img.shape[2] == 4:
                orig_img = cv2.cvtColor(orig_img[:,:,:3], cv2.COLOR_BGR2GRAY)
            else:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                
        # Handle case if the loaded texture map is slightly off in size
        if orig_img.shape != (H, W):
            orig_img = cv2.resize(orig_img, (W, H))
            
        overlay = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        overlay[mask_img == 255] = [255, 255, 0]  # Bright Cyan highlight in BGR
        
        overlay_path = os.path.join(OUTPUT_DIR, f"lesion_overlay_{base_name}.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"  -> Lesion overlay saved     : {overlay_path}")
    else:
        print(f"  [WARN] Original image {orig_img_path} not found. Cannot create overlay.")
        
    print(f"  -> Cluster label map saved  : {label_path}")
    print(f"  -> Lesion binary mask saved : {mask_path}")


if __name__ == '__main__':
    print("=" * 65)
    print("  Lesion Segmentation via FCM-FRWS")
    print("=" * 65)
    
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy")]
    
    if not files:
        print(f"\n[ERROR] No feature files found in: {FEATURE_DIR}")
        print("        Run feature_extraction.py first.")
    else:
        print(f"\nFound {len(files)} feature file(s) to process.\n")
        for filename in sorted(files):
            feature_path = os.path.join(FEATURE_DIR, filename)
            segment_lesion(feature_path, filename)
            
    print("\n" + "=" * 65)
    print("  Segmentation complete!")
    print("=" * 65)
