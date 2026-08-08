import os
import cv2

from src.fractures.image_processing.fracture_preprocess import fracture_specific_preprocess


RAW_DIR = "data/fractures/raw/images"
PREPROCESSED_DIR = "data/fractures/processed/preprocessed"


def run_preprocessing():
    """
    Reads all raw fracture images,
    applies fracture preprocessing,
    and saves outputs into data/fractures/processed/preprocessed.
    """

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    image_files = [
        file for file in os.listdir(RAW_DIR)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No images found in: {RAW_DIR}")
        return

    for file_name in image_files:
        input_path = os.path.join(RAW_DIR, file_name)
        output_path = os.path.join(PREPROCESSED_DIR, file_name)

        try:
            processed = fracture_specific_preprocess(input_path)

            cv2.imwrite(output_path, processed)

            print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

    print("Fracture preprocessing completed.")


if __name__ == "__main__":
    run_preprocessing()