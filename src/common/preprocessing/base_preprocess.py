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


def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)


def normalize_image(img):
    img = img.astype("float32")

    min_val = img.min()
    max_val = img.max()

    if max_val - min_val == 0:
        return img  # avoid division error

    img = (img - min_val) / (max_val - min_val)

    return img

def gaussian_denoise(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def median_denoise(img):
    return cv2.medianBlur(img, 5)

def base_preprocess(path):
    """
    Base preprocessing pipeline:
    - Load image
    - Convert to grayscale
    - Resize
    - Normalize
    """
    img = load_image(path)
    img = to_grayscale(img)
    img = resize_image(img)
    img = normalize_image(img)
    img = median_denoise(img)  
    # img = gaussian_denoise(img) 

    return img