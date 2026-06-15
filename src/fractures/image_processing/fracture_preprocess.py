import cv2
import numpy as np

from src.common.preprocessing.base_preprocess import base_preprocess


def to_uint8(img_float):
    """
    Convert normalized float image 0-1 into uint8 0-255.
    Required because many OpenCV algorithms work better with uint8.
    """
    img = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    return img


def apply_clahe(img_uint8, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to enhance local contrast.
    Useful for highlighting PDL space, lamina dura, root boundaries, and bone texture.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(img_uint8)


def sharpen_image(img_uint8):
    """
    Unsharp masking.
    Enhances root edges, PDL boundaries, and bone margins.
    """
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    sharpened = cv2.addWeighted(img_uint8, 1.5, blurred, -0.5, 0)
    return sharpened


def fracture_specific_preprocess(image_path):
    """
    Full fracture-specific preprocessing pipeline.
    Uses common base_preprocess first, then applies fracture-specific enhancement.
    """
    img_float = base_preprocess(image_path)

    img_uint8 = to_uint8(img_float)

    enhanced = apply_clahe(img_uint8)

    sharpened = sharpen_image(enhanced)

    return sharpened