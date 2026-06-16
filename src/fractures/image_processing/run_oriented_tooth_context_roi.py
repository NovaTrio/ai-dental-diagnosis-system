import os

from src.fractures.image_processing.oriented_tooth_context_roi import (
    extract_oriented_tooth_context_rois
)


INPUT_DIR = "data/fractures/processed/anatomical_region"
OUTPUT_DIR = "data/fractures/processed/oriented_tooth_context_roi"
DEBUG_DIR = "data/fractures/processed/debug_oriented_tooth_context_roi"


def run():
    print("Starting oriented tooth-context ROI extraction...")

    if not os.path.exists(INPUT_DIR):
        print("ERROR: Anatomical region directory does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png"]

    files = os.listdir(INPUT_DIR)
    print("Files found:", len(files))

    processed_count = 0

    for file_name in files:
        if not any(file_name.lower().endswith(ext) for ext in valid_ext):
            continue

        image_path = os.path.join(INPUT_DIR, file_name)
        base_name = os.path.splitext(file_name)[0]

        output_dir = os.path.join(OUTPUT_DIR, base_name)
        debug_path = os.path.join(DEBUG_DIR, file_name)

        try:
            rois, boxes, angle = extract_oriented_tooth_context_rois(
                image_path=image_path,
                output_dir=output_dir,
                debug_path=debug_path
            )

            print(
                f"Processed: {file_name} | "
                f"ROIs: {len(rois)} | "
                f"Angle: {angle:.2f}"
            )

        except Exception as e:
            print(f"Failed: {file_name} | Error: {e}")

        processed_count += 1

    print("Total processed:", processed_count)


if __name__ == "__main__":
    run()