import cv2
import numpy as np

from src.common.preprocessing.base_preprocess import base_preprocess


def to_uint8(img):
    if img.max() <= 1.0:
        img = img * 255
    return img.astype(np.uint8)


def apply_clahe(img):
    img = to_uint8(img)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)


def sharpen_image(img):
    """
    Sharpen edges such as root boundary, lamina dura and PDL margins.
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def unsharp_mask(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharpened = cv2.addWeighted(img, 1.8, blurred, -0.8, 0)
    return sharpened


def fracture_preprocess(image_path):
    """
    Fracture-specific preprocessing:
    1. Common preprocessing
    2. CLAHE contrast enhancement
    3. Edge sharpening
    """
    img = base_preprocess(image_path)
    img = apply_clahe(img)

    # Use one sharpening method
    img = unsharp_mask(img)
    # img = sharpen_image(img)

    return img