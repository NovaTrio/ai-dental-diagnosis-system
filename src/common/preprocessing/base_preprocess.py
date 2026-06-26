import cv2
import numpy as np
import os

def get_dataset_path(task, split="raw"):
    return os.path.join("data", task, split)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    return img


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# def resize_image(img, size=(256, 256)):
#     return cv2.resize(img, size)


def normalize_image(img):
    img = img.astype("float32")

    min_val = img.min()
    max_val = img.max()

    if max_val - min_val == 0:
        return img  # avoid division error

    img = (img - min_val) / (max_val - min_val)

    return img

def contrast_stretch(img, low_pct=2, high_pct=98):
    """
    Percentile-based contrast stretching.
    Ignores extreme bright/dark outliers (common in X-rays).
    Input/Output: normalized float (0-1)
    """
    low = np.percentile(img, low_pct)
    high = np.percentile(img, high_pct)

    # Stretch so low→0 and high→1
    stretched = (img - low) / (high - low + 1e-6)

    return np.clip(stretched, 0, 1)

def sharpen_image(img):
    """
    Unsharp masking.
    Enhances root edges, PDL boundaries, and bone margins.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened


# def gaussian_denoise(img):
#     return cv2.GaussianBlur(img, (5,5), 0)

def median_denoise(img):
    return cv2.medianBlur(img, 5)

def base_preprocess(path):
    """
    Base preprocessing pipeline:
    - Load image
    - Convert to grayscale
    - Resize
    - Denoise (while still uint8)
    - Convert to float and normalize
    """
    img = load_image(path)
    img = to_grayscale(img)
    # img = resize_image(img)
    img = median_denoise(img)  
    img = img.astype("float32") / 255.0 
    img = contrast_stretch(img, 2, 98)
    img = normalize_image(img)
    # img = sharpen_image((img))

    return img