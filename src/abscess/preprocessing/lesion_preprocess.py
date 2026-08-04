import os
from pathlib import Path
import cv2
import numpy as np
from skimage.exposure import match_histograms


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
input_dir = str(_PROJECT_ROOT / "data/common/processed/common_selected_tooth_roi")
output_dir = str(_PROJECT_ROOT / "data/abscess/raw/BlackRemove")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    

def remove_black_background_threshold(image_path, output_path, threshold=50):
    """Remove black background and add alpha channel"""
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Could not read {image_path}")
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([threshold, threshold, threshold])
    mask = cv2.inRange(img_rgb, lower_black, upper_black)
    mask_inv = cv2.bitwise_not(mask)
    
    b, g, r = cv2.split(img_rgb)
    rgba = cv2.merge([b, g, r, mask_inv])
    
    cv2.imwrite(output_path, rgba)
    return True

def sigmoid_enhancement(image, k=12, mid=0.38):
    """Sigmoid transformation for contrast enhancement"""
    has_alpha = False
    alpha_channel = None
    
    if len(image.shape) == 3 and image.shape[2] == 4:
        has_alpha = True
        b, g, r, a = cv2.split(image)
        alpha_channel = a
        img_rgb = cv2.merge([b, g, r])
    else:
        img_rgb = image.copy()
    
    img_float = img_rgb.astype(np.float32) / 255.0
    sigmoid_img = 1.0 / (1.0 + np.exp(-k * (img_float - mid)))
    sigmoid_img = np.clip(sigmoid_img, 0, 1)
    sigmoid_img = (sigmoid_img * 255).astype(np.uint8)
    result = cv2.cvtColor(sigmoid_img, cv2.COLOR_RGB2BGR)
    
    if has_alpha:
        result = cv2.merge([result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha_channel])
    
    return result

def clahe_enhancement(image, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE for local contrast enhancement"""
    has_alpha = False
    alpha_channel = None
    
    if len(image.shape) == 3 and image.shape[2] == 4:
        has_alpha = True
        b, g, r, a = cv2.split(image)
        alpha_channel = a
        img_rgb = cv2.merge([b, g, r])
    else:
        img_rgb = image.copy()
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if has_alpha:
        enhanced = cv2.merge([enhanced[:, :, 0], enhanced[:, :, 1], enhanced[:, :, 2], alpha_channel])
    
    return enhanced

def clahe_sigmoid_combined(image):
    """
    Apply CLAHE first, then Sigmoid for maximum lesion visibility
    """
    # First apply CLAHE
    clahe_result = clahe_enhancement(image, clipLimit=2.0, tileGridSize=(8, 8))
    
    # Then apply sigmoid on CLAHE result
    final_result = sigmoid_enhancement(clahe_result, k=12, mid=0.38)
    
    return final_result


# ==============================================================
# X-RAY PREPROCESSING FRAMEWORK
# Stage 1: Intensity Normalization (Histogram Matching)
# Stage 2: Linear Averaging Filter (Noise Reduction)
# Stage 3: Morphological Closing (Smoothing)
# ==============================================================

def _split_alpha(image):
    """Utility: separate alpha channel if present.
    Returns (colour_image_BGR, alpha_or_None, had_alpha_bool).
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        return cv2.merge([b, g, r]), a, True
    return image.copy(), None, False


def _merge_alpha(image, alpha):
    """Utility: re-attach alpha channel if it was present."""
    if alpha is not None:
        return cv2.merge([image[:, :, 0], image[:, :, 1],
                          image[:, :, 2], alpha])
    return image


def histogram_matching_normalization(image, reference_image):
    """
    Stage 1 – Intensity Normalization via Histogram Matching.

    Adjusts the input image's intensity distribution to match a
    carefully selected reference image using CDF-based transformation:
        s = F_s^{-1}( F_r(r) )
    where F_r and F_s are the CDFs of the input and reference images.

    This equalizes brightness across the dataset and amplifies
    anatomical details such as enamel boundaries and lesion textures.
    """
    img_bgr, alpha, has_alpha = _split_alpha(image)
    ref_bgr, _, _ = _split_alpha(reference_image)

    # match_histograms expects channel-last arrays (H, W, C)
    matched = match_histograms(img_bgr, ref_bgr, channel_axis=-1)
    matched = np.clip(matched, 0, 255).astype(np.uint8)

    return _merge_alpha(matched, alpha)


# def linear_averaging_filter(image, kernel_size=5):
#     """
#     Stage 2 – Linear Averaging Filter for Noise Reduction.

#     Applies a uniform (box) averaging filter:
#         g_ij = Σ_{k,l} w_{kl} · f_{i+k, j+l}
#     with a kernel_size × kernel_size window (default 5×5, m=2).

#     Effectively reduces random and Gaussian noise amplified by
#     intensity normalization while preserving edge information and
#     caries contours.
#     """
#     img_bgr, alpha, has_alpha = _split_alpha(image)

#     # Uniform weight kernel  –  all weights = 1/(kernel_size^2)
#     filtered = cv2.blur(img_bgr, (kernel_size, kernel_size))

#     return _merge_alpha(filtered, alpha)


def morphological_closing(image, struct_size=6):
    """
    Stage 3 – Morphological Smoothing via Closing Operator.

    Defined as:  A • B = (A ⊕ B) ⊖ B
    where A is the input image and B is a rectangular structuring
    element of the given size (default 6×6).

    The closing operation (dilation followed by erosion) fills small
    gaps, removes dark noise patches, and produces smoother, continuous
    object boundaries — improving dental region integrity and caries
    delineation.
    """
    img_bgr, alpha, has_alpha = _split_alpha(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (struct_size, struct_size))
    closed = cv2.morphologyEx(img_bgr, cv2.MORPH_CLOSE, kernel)

    return _merge_alpha(closed, alpha)


def xray_preprocessing_pipeline(image, reference_image,
                                 kernel_size=5, struct_size=6):
    """
    Complete three-step X-ray preprocessing framework.

    1. Histogram matching  → intensity normalization
    2. Linear averaging    → noise suppression
    3. Morphological close → structural smoothing

    Parameters
    ----------
    image : ndarray
        Input X-ray image (BGR or BGRA).
    reference_image : ndarray
        Reference image with clear caries features and strong contrast.
    kernel_size : int
        Size of the averaging filter window (default 5).
    struct_size : int
        Size of the rectangular structuring element (default 6).

    Returns
    -------
    preprocessed : ndarray
        Preprocessed image ready for segmentation / classification.
    """
    # Stage 1 – Intensity Normalization
    normalized = histogram_matching_normalization(image, reference_image)

    # Stage 2 – Noise Reduction
    # filtered = linear_averaging_filter(normalized, kernel_size=kernel_size)

    # Stage 3 – Morphological Smoothing
    preprocessed = morphological_closing(normalized, struct_size=struct_size)

    return preprocessed


# ==============================================================
# Reusable single-image preprocessing (for run_pipeline.py)
# ==============================================================
def preprocess_single_image(input_path, output_path, reference_image_path=None):
    """
    Run the full preprocessing chain on ONE image and save the result.

    Reuses the existing stage functions — background removal →
    xray_preprocessing_pipeline (histogram matching + morphological closing) →
    clahe_sigmoid_combined — without touching the batch loop.

    Parameters
    ----------
    input_path : str
        Path to a single selected-tooth-ROI image.
    output_path : str
        Where the final CLAHE+Sigmoid image is written (parent dirs are created).
    reference_image_path : str, optional
        Reference image for histogram matching. If None (or unreadable), the
        input itself is used as its own reference, which makes the matching a
        no-op and leaves the remaining stages unchanged.

    Returns
    -------
    str
        ``output_path`` on success.

    Raises
    ------
    FileNotFoundError
        If the input image cannot be read or background removal fails.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Step 1: background removal → temp BGRA beside the output
    temp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)),
                             f"_temp_{os.path.basename(output_path)}")
    if not remove_black_background_threshold(input_path, temp_path, threshold=50):
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    try:
        # Step 2: read back with the alpha channel
        img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Background removal failed for: {input_path}")

        # Step 3: reference for histogram matching (falls back to the input)
        reference_image = None
        if reference_image_path:
            reference_image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
        if reference_image is None:
            reference_image = img

        # Step 4: X-ray preprocessing (histogram matching + morphological closing)
        preprocessed_img = xray_preprocessing_pipeline(
            img, reference_image, kernel_size=5, struct_size=6)

        # Step 5: CLAHE + Sigmoid
        enhanced_img = clahe_sigmoid_combined(preprocessed_img)

        # Step 6: save
        cv2.imwrite(output_path, enhanced_img)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return output_path


if __name__ == "__main__":
    # ============================================
    # REFERENCE IMAGE FOR HISTOGRAM MATCHING
    # ============================================
    # Select the first suitable image as the reference.
    # Replace with a specific path if you have a hand-picked
    # reference image with clear caries features and strong contrast.
    reference_image_path = None
    for _fn in os.listdir(input_dir):
        if _fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            reference_image_path = os.path.join(input_dir, _fn)
            break

    if reference_image_path is None:
        raise FileNotFoundError(f"No images found in {input_dir} to use as reference.")

    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
    print(f"Reference image for histogram matching: {reference_image_path}")

    # ============================================
    # Process images - Output ONLY CLAHE+Sigmoid
    # ============================================
    print("=" * 60)
    print("PROCESSING IMAGES - X-RAY PREPROCESSING + CLAHE + SIGMOID")
    print("=" * 60)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            base_name = filename.rsplit('.', 1)[0]

            # Step 1: Remove black background (temporary)
            temp_path = os.path.join(output_dir, '_temp_.png')
            remove_black_background_threshold(input_path, temp_path, threshold=50)

            # Step 2: Read image with alpha channel
            img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)

            if img is not None:
                # Step 3: X-ray preprocessing (histogram matching +
                #    _     linear averaging filter + morphological closing)
                preprocessed_img = xray_preprocessing_pipeline(
                    img, reference_image,
                    kernel_size=5,   # 5×5 averaging filter (m=2)
                    struct_size=6    # 6×6 rectangular structuring element
                )

                # Step 4: Apply CLAHE + Sigmoid combined enhancement
                enhanced_img = clahe_sigmoid_combined(preprocessed_img)

                # Step 5: Save final output
                output_path = os.path.join(output_dir, f"{base_name}_clahe_sigmoid.png")
                cv2.imwrite(output_path, enhanced_img)

                # Step 6: Remove temporary file
                os.remove(temp_path)

                print(f"✓ Created: {base_name}_clahe_sigmoid.png")
            else:
                print(f"✗ Failed: {filename}")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nPipeline: Background Removal → Histogram Matching → ")
    print("          Averaging Filter → Morphological Closing → ")
    print("          CLAHE → Sigmoid Enhancement")
    print("\n Output files in 'output_images' folder:")
    print("    [filename]_clahe_sigmoid.png - Full pipeline output")
    print("\n This combination is optimized for lesion visibility.")
    print("=" * 60)
