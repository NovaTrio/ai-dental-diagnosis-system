"""
n4_preprocess.py
────────────────
Apply N4 bias-field correction to crown-cropped periapical X-ray images.

Input:
    ../../../data/abscess/raw/croun_crops

Output:
    ../../../data/abscess/raw/N4Output

The script:
1. Reads PNG, JPG, and JPEG images.
2. Supports grayscale, BGR, and BGRA images.
3. Uses the alpha channel as the foreground mask when available.
4. Otherwise creates a foreground mask by removing near-black pixels.
5. Applies SimpleITK N4 bias-field correction.
6. Saves corrected images with the original filenames.

Usage:
    python n4_preprocess.py
"""

import os

import cv2
import numpy as np
import SimpleITK as sitk


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR = "../../../data/abscess/raw/croun_crops"
OUTPUT_DIR = "../../../data/abscess/raw/N4Output"

BLACK_THRESHOLD = 10

# N4 parameters
N4_SHRINK_FACTOR = 2
N4_ITERATIONS = [50, 50, 30]
N4_FWHM = 0.15

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_foreground_mask(
    original_image: np.ndarray,
    gray_image: np.ndarray,
) -> np.ndarray:
    """
    Create a foreground mask for N4 correction.

    If the input contains an alpha channel, alpha > 0 is used as foreground.
    Otherwise, near-black pixels are treated as background.

    Returns
    -------
    np.ndarray
        uint8 mask where 255 = foreground and 0 = background.
    """
    if original_image.ndim == 3 and original_image.shape[2] == 4:
        alpha = original_image[:, :, 3]
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    else:
        mask = np.where(
            gray_image > BLACK_THRESHOLD,
            255,
            0,
        ).astype(np.uint8)

    # Small cleanup to remove isolated mask noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask


def normalize_to_uint8(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Normalize corrected floating-point intensities to uint8.

    Only foreground pixels are used for normalization.
    """
    output = np.zeros_like(image, dtype=np.uint8)

    foreground_values = image[mask > 0]

    if foreground_values.size == 0:
        return output

    low = float(np.percentile(foreground_values, 1))
    high = float(np.percentile(foreground_values, 99))

    if high <= low:
        clipped = np.clip(image, 0, 255)
        output = clipped.astype(np.uint8)
    else:
        normalized = (image - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)
        output = (normalized * 255.0).astype(np.uint8)

    output[mask == 0] = 0
    return output


def apply_n4_bias_correction(
    gray_image: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply SimpleITK N4 bias-field correction.

    Parameters
    ----------
    gray_image
        uint8 grayscale image.
    foreground_mask
        uint8 mask where non-zero pixels define the correction region.

    Returns
    -------
    np.ndarray
        Corrected uint8 grayscale image.
    """
    if gray_image.ndim != 2:
        raise ValueError("gray_image must be a 2D grayscale image.")

    binary_mask = (foreground_mask > 0).astype(np.uint8)

    if np.count_nonzero(binary_mask) == 0:
        print("  [WARN] Empty foreground mask. Using the entire image.")
        binary_mask = np.ones_like(binary_mask, dtype=np.uint8)

    image_sitk = sitk.GetImageFromArray(
        gray_image.astype(np.float32)
    )

    mask_sitk = sitk.GetImageFromArray(binary_mask)
    mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

    if N4_SHRINK_FACTOR > 1:
        shrink_factors = [
            N4_SHRINK_FACTOR
        ] * image_sitk.GetDimension()

        small_image = sitk.Shrink(
            image_sitk,
            shrink_factors,
        )

        small_mask = sitk.Shrink(
            mask_sitk,
            shrink_factors,
        )
    else:
        small_image = image_sitk
        small_mask = mask_sitk

    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()

    n4_filter.SetMaximumNumberOfIterations(
        N4_ITERATIONS
    )

    n4_filter.SetBiasFieldFullWidthAtHalfMaximum(
        N4_FWHM
    )

    # Estimate the bias field using the smaller image
    n4_filter.Execute(
        small_image,
        small_mask,
    )

    # Get estimated bias field at original image size
    log_bias_field = n4_filter.GetLogBiasFieldAsImage(
        image_sitk
    )

    image_array = sitk.GetArrayFromImage(
        image_sitk
    )

    log_bias_array = sitk.GetArrayFromImage(
        log_bias_field
    )

    bias_field = np.exp(log_bias_array)

    corrected_float = np.divide(
        image_array,
        bias_field,
        out=np.zeros_like(image_array, dtype=np.float32),
        where=bias_field > 1e-8,
    )

    corrected_uint8 = normalize_to_uint8(
        corrected_float,
        foreground_mask,
    )

    return corrected_uint8


def process_image(
    input_path: str,
    output_path: str,
) -> bool:
    """
    Read one image, apply N4 correction, and save the result.
    """
    original = cv2.imread(
        input_path,
        cv2.IMREAD_UNCHANGED,
    )

    if original is None:
        print(f"  [FAIL] Could not read: {input_path}")
        return False

    if original.ndim == 2:
        gray = original.copy()
        alpha = None

    elif original.shape[2] == 4:
        gray = cv2.cvtColor(
            original[:, :, :3],
            cv2.COLOR_BGR2GRAY,
        )
        alpha = original[:, :, 3]

    else:
        gray = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2GRAY,
        )
        alpha = None

    mask = create_foreground_mask(
        original,
        gray,
    )

    corrected = apply_n4_bias_correction(
        gray,
        mask,
    )

    # Preserve alpha channel when the input is BGRA
    if alpha is not None:
        corrected_output = cv2.merge([
            corrected,
            corrected,
            corrected,
            alpha,
        ])
    else:
        corrected_output = corrected

    saved = cv2.imwrite(
        output_path,
        corrected_output,
    )

    if not saved:
        print(f"  [FAIL] Could not save: {output_path}")
        return False

    return True


def main() -> None:
    print("=" * 70)
    print("N4 BIAS-FIELD CORRECTION")
    print("=" * 70)
    print(f"Input directory  : {INPUT_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Shrink factor    : {N4_SHRINK_FACTOR}")
    print(f"Iterations       : {N4_ITERATIONS}")
    print("=" * 70)

    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input directory not found:\n{INPUT_DIR}")
        return

    files = sorted(
        filename
        for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith(SUPPORTED_EXTENSIONS)
    )

    if not files:
        print(f"[ERROR] No images found in:\n{INPUT_DIR}")
        return

    successful = 0

    for filename in files:
        input_path = os.path.join(
            INPUT_DIR,
            filename,
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename,
        )

        print(f"\nProcessing: {filename}")

        try:
            if process_image(input_path, output_path):
                successful += 1
                print(f"  [OK] Saved: {output_path}")

        except Exception as error:
            print(f"  [FAIL] {filename}: {error}")

    print("\n" + "=" * 70)
    print(
        f"N4 PROCESSING COMPLETE: "
        f"{successful}/{len(files)} images"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()