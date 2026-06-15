import os

from src.fractures.image_processing.anatomy_segmentation import extract_anatomical_region


RAW_DIR = "data/fractures/raw/images"
OUTPUT_DIR = "data/fractures/processed/anatomical_region"
DEBUG_DIR = "data/fractures/processed/debug_anatomical_region"


def run():
    print("Starting anatomical region extraction...")
    print("RAW_DIR:", RAW_DIR)

    if not os.path.exists(RAW_DIR):
        print("ERROR: Raw image directory does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png"]

    files = os.listdir(RAW_DIR)
    print("Files found:", len(files))

    processed_count = 0

    for file_name in files:
        if not any(file_name.lower().endswith(ext) for ext in valid_ext):
            continue

        input_path = os.path.join(RAW_DIR, file_name)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        debug_path = os.path.join(DEBUG_DIR, file_name)

        extract_anatomical_region(
            image_path=input_path,
            save_path=output_path,
            debug_path=debug_path
        )

        processed_count += 1
        print(f"Processed: {file_name}")

    print("Total processed:", processed_count)


if __name__ == "__main__":
    run()