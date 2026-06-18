import os

from src.abscess.image_processing.anatomy_segmentation import extract_anatomical_region


INPUT_DIR = "data/common/processed/common_selected_tooth_roi"
OUTPUT_DIR = "data/abscess/preprocessing/crop_croun"
DEBUG_DIR = "data/abscess/preprocessing/debug_anatomical_region"


def run():
    print("Starting crown-margin-based root extraction...")
    print("INPUT_DIR:", INPUT_DIR)

    if not os.path.exists(INPUT_DIR):
        print("ERROR: Input image directory does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    files = [
        file_name for file_name in os.listdir(INPUT_DIR)
        if any(file_name.lower().endswith(ext) for ext in valid_ext)
    ]

    print("Valid image files found:", len(files))

    processed_count = 0

    for file_name in files:
        input_path = os.path.join(INPUT_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        debug_path = os.path.join(DEBUG_DIR, file_name)

        extract_anatomical_region(
            image_path=input_path,
            save_path=output_path,
            debug_path=debug_path,

            # Detect first stable bright crown region
            central_width_ratio=0.75,
            bright_percentile=88,
            min_bright_ratio=0.16,
            consecutive_rows=6,

            # Remove crown after detecting upper crown margin
            crown_depth_ratio=0.18,
            min_crown_depth=30,
            max_crown_depth_ratio=0.28
        )

        processed_count += 1
        print(f"Processed: {file_name}")

    print("Total processed:", processed_count)
    print("Output saved to:", OUTPUT_DIR)
    print("Debug saved to:", DEBUG_DIR)


if __name__ == "__main__":
    run()