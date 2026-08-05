import cv2
import numpy as np

def to_uint8(img_float):
    """
    Convert normalized float image 0-1 into uint8 0-255.
    """
    return np.clip(img_float * 255, 0, 255).astype(np.uint8)


def gamma_brighten(img_uint8, gamma=0.75):
    """
    Brighten darker bone regions without over-brightening the tooth.
    gamma < 1 brightens the image.
    gamma > 1 darkens the image.
    """

    img_float = img_uint8.astype(np.float32) / 255.0

    corrected = np.power(img_float, gamma)

    return np.clip(corrected * 255, 0, 255).astype(np.uint8)


def apply_mild_clahe(img_uint8, clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    Mild CLAHE.
    Strong CLAHE makes bone trabeculae too dark/noisy.
    Use low clip_limit for dental X-rays.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(img_uint8)


def extract_pdl_dark_lines(img_uint8, kernel_size=9):
    """
    Black-hat morphology extracts thin dark structures.
    This is useful for PDL-like dark gaps.

    blackhat = closing(image) - image
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        img_uint8,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    blackhat = cv2.normalize(
        blackhat,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return blackhat.astype(np.uint8)


def selective_pdl_enhancement(img_uint8, pdl_map, strength=0.35):
    """
    Enhance only thin dark PDL-like structures.
    This avoids darkening the whole bone region.

    Higher strength = darker PDL lines.
    Recommended: 0.25 to 0.45
    """

    pdl_effect = (pdl_map.astype(np.float32) * strength).astype(np.uint8)

    enhanced = cv2.subtract(img_uint8, pdl_effect)

    return enhanced


def sharpen_image(img_uint8, amount=0.8):
    """
    Mild sharpening.
    Too much sharpening increases bone noise.
    """

    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)

    sharpened = cv2.addWeighted(
        img_uint8,
        1.0 + amount,
        blurred,
        -amount,
        0
    )

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def fracture_specific_preprocess_image(image):
    """Apply fracture-specific enhancement without common resize/preprocessing."""
    if image is None or image.size == 0:
        raise ValueError("Input fracture image is empty")
    if image.ndim == 3:
        img_uint8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        img_uint8 = image.astype(np.uint8, copy=False)
    else:
        raise ValueError("Fracture image must be grayscale or BGR")

    # Preserve the source dimensions and intensities. Only the processing
    # specific to PDL/fracture visibility is applied in this diagnostic flow.
    brightened = gamma_brighten(img_uint8, gamma=0.75)
    clahe_img = apply_mild_clahe(
        brightened,
        clip_limit=1.5,
        tile_grid_size=(8, 8),
    )
    pdl_map = extract_pdl_dark_lines(clahe_img, kernel_size=9)
    pdl_enhanced = selective_pdl_enhancement(
        clahe_img,
        pdl_map,
        strength=0.35,
    )
    return sharpen_image(pdl_enhanced, amount=0.6)


def fracture_specific_preprocess(image_path):
    """
    Bone-preserving fracture preprocessing.

    Goal:
    - Keep bone from becoming too dark
    - Improve PDL visibility
    - Avoid over-enhancing trabecular bone texture
    """

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return fracture_specific_preprocess_image(image)
