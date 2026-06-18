"""
texture_processing.py
─────────────────────
This script applies the texture analysis techniques from the Python Basic Image Texture Analysis Guide:
1. Extracts texture features (Contrast, Energy, Homogeneity) using GLCM (Gray-Level Co-occurrence Matrix).
2. Applies a Gabor filter to detect and extract texture.
3. REMOVES texture from the image using edge-preserving smoothing (Bilateral Filtering), 
   which smooths out textures while keeping the structural edges (like tooth boundaries) intact.

Outputs are saved to the 'texture_processed' directory.
"""

import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
# from skimage.filters import gabor


INPUT_DIR = '../../../data/abscess/raw/BlackRemove'
OUTPUT_DIR = '../../../data/abscess/raw/texture_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_and_remove_texture(image_path, filename):
    print(f"\nProcessing: {filename}")
    
    # Load image in grayscale (as recommended by the guide)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Could not read {image_path}")
        return

    # Load original image for texture removal to preserve colors/alpha if needed
    img_original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 1. TEXTURE ANALYSIS (GLCM) - From the Guide

    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    print(f"  - Contrast:    {contrast:.2f}")
    print(f"  - Energy:      {energy:.4f}")
    print(f"  - Homogeneity: {homogeneity:.4f}")
   
    # 2. APPLY GABOR FILTER - From the Guide
    # filt_real, filt_imag = gabor(img_gray, frequency=0.6)
    
    # Convert the real part of the Gabor filter to a visible image format (0-255)
    # gabor_vis = cv2.normalize(filt_real, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Save the extracted texture (Gabor)
    # gabor_output_path = os.path.join(OUTPUT_DIR, f"gabor_texture_{filename}")
    # cv2.imwrite(gabor_output_path, gabor_vis)

    # 3. REMOVE TEXTURES (Smoothing)
    
    if len(img_original.shape) == 3 and img_original.shape[2] == 4:
        # If it has an alpha channel (RGBA), process only RGB, then merge alpha back
        b, g, r, a = cv2.split(img_original)
        img_rgb = cv2.merge([b, g, r])
        
        # 1. Remove fine-grained noise (rough texture)
        denoised_rgb = cv2.fastNlMeansDenoisingColored(img_rgb, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Apply bilateral filter
        texture_removed_rgb = cv2.bilateralFilter(denoised_rgb, d=9, sigmaColor=50, sigmaSpace=50)
        
        # Merge alpha back
        texture_removed = cv2.merge([
            texture_removed_rgb[:,:,0], 
            texture_removed_rgb[:,:,1], 
            texture_removed_rgb[:,:,2], 
            a
        ])
    else:
        # Standard BGR or Grayscale
        if len(img_original.shape) == 3:
            # 1. Remove fine-grained noise (rough texture)
            denoised = cv2.fastNlMeansDenoisingColored(img_original, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        else:
            # 1. Remove fine-grained noise (rough texture)
            denoised = cv2.fastNlMeansDenoising(img_original, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
        # 2. Apply bilateral filter
        texture_removed = cv2.bilateralFilter(denoised, d=9, sigmaColor=50, sigmaSpace=50)

    # Save the texture-removed image
    removed_output_path = os.path.join(OUTPUT_DIR, f"texture_removed_{filename}")
    cv2.imwrite(removed_output_path, texture_removed)
    
    # print(f"  -> Saved Gabor texture extraction to: {gabor_output_path}")
    print(f"  -> Saved Texture-REMOVED image to:    {removed_output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Image Texture Analysis and Removal Pipeline")
    print("=" * 60)
    
    supported_exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(supported_exts)]
    
    if not files:
        print(f"No images found in {INPUT_DIR}.")
    else:
        for filename in files:
            image_path = os.path.join(INPUT_DIR, filename)
            analyze_and_remove_texture(image_path, filename)
            
    print("\nProcessing complete!")
