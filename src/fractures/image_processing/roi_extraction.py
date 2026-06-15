import os
import cv2
import pandas as pd


def crop_roi_from_annotations(
    annotations_path="data/fractures/annotations/roi_annotations.csv",
    image_dir="data/fractures/processed/preprocessed",
    output_dir="data/fractures/processed/roi"
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(annotations_path)

    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row["image_name"])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Image not found: {image_path}")
            continue

        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])

        roi = image[y_min:y_max, x_min:x_max]

        output_path = os.path.join(output_dir, row["image_name"])
        cv2.imwrite(output_path, roi)

        print(f"ROI saved: {row['image_name']}")