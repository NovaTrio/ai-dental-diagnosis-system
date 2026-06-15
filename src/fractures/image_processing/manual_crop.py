import cv2
import os
import pandas as pd

IMAGE_DIR = "data/fractures/processed/preprocessed"
OUTPUT_CSV = "data/fractures/annotations/roi_annotations.csv"

ZOOM_SCALE = 2.0

def manual_crop():
    records = []

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_file in image_files:
        path = os.path.join(IMAGE_DIR, image_file)
        image = cv2.imread(path)

        if image is None:
            print(f"Could not load {image_file}")
            continue

        # Enlarge image for easier accurate selection
        zoomed = cv2.resize(
            image,
            None,
            fx=ZOOM_SCALE,
            fy=ZOOM_SCALE,
            interpolation=cv2.INTER_CUBIC
        )

        print(f"Select ROI for: {image_file}")

        roi = cv2.selectROI(
            "Select Root ROI",
            zoomed,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyAllWindows()

        x, y, w, h = roi

        # Convert zoomed coordinates back to original image coordinates
        x_min = int(x / ZOOM_SCALE)
        y_min = int(y / ZOOM_SCALE)
        x_max = int((x + w) / ZOOM_SCALE)
        y_max = int((y + h) / ZOOM_SCALE)

        records.append({
            "image_name": image_file,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print("ROI annotations saved.")

if __name__ == "__main__":
    manual_crop()