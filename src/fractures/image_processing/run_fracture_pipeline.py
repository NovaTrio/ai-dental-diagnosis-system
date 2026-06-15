import os
import cv2

from src.fractures.image_processing.fracture_preprocess import fracture_preprocess
from src.fractures.image_processing.roi_extraction import crop_roi_from_annotations


RAW_DIR = "data/fractures/raw/images"
PREPROCESSED_DIR = "data/fractures/processed/preprocessed"


def run_preprocessing():
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    image_files = [
        file for file in os.listdir(RAW_DIR)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No images found in {RAW_DIR}")
        return

    for file in image_files:
        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(PREPROCESSED_DIR, file)

        processed = fracture_preprocess(input_path)
        cv2.imwrite(output_path, processed)

        print(f"Processed: {file}")

    print("Fracture preprocessing completed.")


if __name__ == "__main__":
    run_preprocessing()
    crop_roi_from_annotations()