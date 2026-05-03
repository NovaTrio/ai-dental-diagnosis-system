import os
import cv2
import numpy as np
from PIL import Image

input_dir = '../../../data/abscess/raw/images'
output_dir = '../../../data/abscess/raw/BlackRemove'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def sharpen_image(image):
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply the kernel to the image
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def remove_black_background_threshold(image_path, output_path, threshold=50):
    # Read image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read {image_path}")
        return False

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Sharpen the image before processing
    img_rgb = sharpen_image(img_rgb)

    # Create mask: pixels where all channels < threshold are considered black
    # (0,0,0) to (threshold, threshold, threshold)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([threshold+5, threshold+5, threshold+5])
    mask = cv2.inRange(img_rgb, lower_black, upper_black)

    # Invert mask (keep non-black areas)
    mask_inv = cv2.bitwise_not(mask)

    # Split channels
    b, g, r = cv2.split(img_rgb)

    # Add alpha channel
    rgba = cv2.merge([b, g, r, mask_inv])

    # Save as PNG
    cv2.imwrite(output_path, rgba)
    return True

# Define a function for histogram equalization
def apply_histogram_equalization(image_path, output_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not read {image_path}")
        return False

    # Apply histogram equalization
    equalized_img = cv2.equalizeHist(img)

    # Save the equalized image
    cv2.imwrite(output_path, equalized_img)
    return True

# Directory for histogram equalized images
hist_eq_dir = '../../../data/abscess/raw/HistogramEqualized'
if not os.path.exists(hist_eq_dir):
    os.makedirs(hist_eq_dir)

# Apply histogram equalization to images in input_dir
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        hist_eq_path = os.path.join(hist_eq_dir, filename)

        # Apply histogram equalization
        success = apply_histogram_equalization(input_path, hist_eq_path)

        if success:
            print(f"Histogram equalized: {filename}")
        else:
            print(f"Failed to equalize: {filename}")

# Process histogram-equalized images for black background removal
for filename in os.listdir(hist_eq_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        hist_eq_path = os.path.join(hist_eq_dir, filename)
        output_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + '.png')

        # Remove black background
        success = remove_black_background_threshold(hist_eq_path, output_path, threshold=50)

        if success:
            print(f"Black background removed: {filename}")
        else:
            print(f"Failed to remove black background: {filename}")

