"""
feature_extraction.py
──────────────────────
Extracts a 6-dimensional feature vector for each pixel from the preprocessed images.
Ensures rigorous normalization per feature map before stacking into a final (N, 6) matrix.
"""

import os
import cv2
import numpy as np

from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk

# ======================================================
# PATH CONFIGURATION
# ======================================================
INPUT_DIR = "../../../data/abscess/raw/texture_output"
OUTPUT_DIR = "../../../data/abscess/raw/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process only texture removed images
INPUT_PREFIX = 'texture_removed_'

# ======================================================
# FEATURE EXTRACTION FUNCTION
# ======================================================
def extract_features(image):
    """
    Extracts 6 features from the image, normalizes them independently,
    and returns an (H*W, 6) matrix.
    """
    
    # 1. Safely convert to grayscale
    if len(image.shape) == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to float for math operations
    gray_float = gray.astype(np.float32) / 255.0

    H, W = gray.shape

    # -------------------------------------------------
    # Feature 1 : Intensity
    # -------------------------------------------------
    intensity = gray_float.copy()

    # Normalize map [0, 1]
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

    # -------------------------------------------------
    # Feature 2 : Local Mean
    # -------------------------------------------------
    local_mean = cv2.blur(gray_float, (5, 5))
    local_mean = (local_mean - local_mean.min()) / (local_mean.max() - local_mean.min() + 1e-8)

    # -------------------------------------------------
    # Feature 3 : Local Variance
    # -------------------------------------------------
    sqr_mean = cv2.blur(gray_float * gray_float, (5, 5))
    local_variance = sqr_mean - (local_mean * local_mean)
    
    # Clip tiny negative numbers arising from float precision
    local_variance = np.clip(local_variance, 0, None)
    local_variance = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-8)

    # -------------------------------------------------
    # Feature 4 : Gradient Magnitude
    # -------------------------------------------------
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)

    # -------------------------------------------------
    # Feature 5 : Entropy
    # -------------------------------------------------
    # skimage entropy requires uint8 (8-bit) image
    gray_uint8 = (gray_float * 255).astype(np.uint8)
    entropy_img = entropy(gray_uint8, disk(5)).astype(np.float32)
    entropy_img = (entropy_img - entropy_img.min()) / (entropy_img.max() - entropy_img.min() + 1e-8)

    # -------------------------------------------------
    # Feature 6 : Local Binary Pattern
    # -------------------------------------------------
    radius = 2
    n_points = radius * 8
    lbp = local_binary_pattern(gray_uint8, n_points, radius, method="uniform").astype(np.float32)
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)

    # -------------------------------------------------
    # Stack and Reshape (H*W, 6)
    # -------------------------------------------------
    feature_image = np.dstack((
        intensity,
        local_mean,
        local_variance,
        gradient,
        entropy_img,
        lbp
    ))
    
    # Reshape into N x 6 matrix
    feature_matrix = feature_image.reshape(H * W, 6)

    return feature_matrix, (H, W)


# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  Feature Extraction Started")
    print("=" * 65)

    supported_exts = ('.png', '.jpg', '.jpeg')
    
    files = [f for f in os.listdir(INPUT_DIR) 
             if f.lower().startswith(INPUT_PREFIX) and f.lower().endswith(supported_exts)]
    
    if not files:
        print(f"\n[ERROR] No '{INPUT_PREFIX}*' images found in:\n  {INPUT_DIR}")
    else:
        for filename in sorted(files):
            image_path = os.path.join(INPUT_DIR, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"  [WARN] Cannot read {filename}")
                continue
                
            print(f"Processing: {filename}")
            
            # Extract features (returns (N, 6) matrix and original shape tuple)
            feature_matrix, original_shape = extract_features(image)
            
            # Save Feature Matrix safely
            save_name = os.path.splitext(filename)[0] + ".npy"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            np.save(save_path, {
                "features": feature_matrix,
                "shape": original_shape
            })
            
            print(f"  - Shape  : {original_shape[1]}x{original_shape[0]}")
            print(f"  - Matrix : {feature_matrix.shape}")
            print(f"  - Saved  : {save_path}\n")

    print("=" * 65)
    print("  Feature Extraction Completed")
    print("=" * 65)
