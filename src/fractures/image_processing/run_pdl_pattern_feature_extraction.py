import os
import pandas as pd

from src.fractures.image_processing.pdl_pattern_feature_extraction import (
    extract_pdl_pattern_features,
    save_features_csv,
    find_mask_file
)


IMAGE_DIR = "data/fractures/processed/anatomical_region"

ROOT_MASK_DIR = "data/fractures/processed/polynomial_root_masks"
PDL_MASK_DIR = "data/fractures/processed/dark_pdl_from_polynomial_root_masks"

DEBUG_DIR = "data/fractures/processed/debug_pdl_pattern_features"

OUTPUT_CSV_PATH = "data/fractures/processed/pdl_pattern_features.csv"

# Optional dentist annotation CSV.
# Put your uploaded fracture_dataset_numeric.csv here if you want to compare.
DENTIST_LABEL_CSV_PATH = "data/fractures/labels/fracture_dataset_numeric.csv"

MERGED_OUTPUT_CSV_PATH = "data/fractures/processed/pdl_pattern_features_with_dentist_labels.csv"


def merge_with_dentist_labels(feature_csv_path, label_csv_path, output_csv_path):
    if not os.path.exists(label_csv_path):
        print("Dentist label CSV not found. Skipping merge:", label_csv_path)
        return

    features_df = pd.read_csv(feature_csv_path)
    labels_df = pd.read_csv(label_csv_path)

    if "image_name" not in labels_df.columns:
        print("ERROR: Dentist label CSV must contain 'image_name' column.")
        return

    # Rename dentist labels so they do not conflict with rule-based scores.
    rename_map = {}

    if "pdl_pattern_score" in labels_df.columns:
        rename_map["pdl_pattern_score"] = "dentist_pdl_pattern_score"

    if "fracture_probability_score" in labels_df.columns:
        rename_map["fracture_probability_score"] = "dentist_fracture_probability_score"

    if "rct_success_score" in labels_df.columns:
        rename_map["rct_success_score"] = "dentist_rct_success_score"

    labels_df = labels_df.rename(columns=rename_map)

    merged = pd.merge(
        features_df,
        labels_df,
        on="image_name",
        how="left"
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    merged.to_csv(output_csv_path, index=False)

    print("Saved merged feature + dentist label CSV:", output_csv_path)

    if "dentist_pdl_pattern_score" in merged.columns:
        matched = merged.dropna(subset=["dentist_pdl_pattern_score"])

        if len(matched) > 0:
            accuracy = (
                matched["pdl_pattern_score_rule"].astype(int)
                == matched["dentist_pdl_pattern_score"].astype(int)
            ).mean()

            print("Rule vs dentist PDL pattern agreement:", round(float(accuracy), 4))


def run():
    print("Starting PDL pattern feature extraction...")
    print("IMAGE_DIR:", IMAGE_DIR)
    print("ROOT_MASK_DIR:", ROOT_MASK_DIR)
    print("PDL_MASK_DIR:", PDL_MASK_DIR)

    if not os.path.exists(IMAGE_DIR):
        print("ERROR: IMAGE_DIR does not exist.")
        return

    if not os.path.exists(ROOT_MASK_DIR):
        print("ERROR: ROOT_MASK_DIR does not exist.")
        return

    if not os.path.exists(PDL_MASK_DIR):
        print("ERROR: PDL_MASK_DIR does not exist.")
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    files = [
        file_name for file_name in os.listdir(IMAGE_DIR)
        if any(file_name.lower().endswith(ext) for ext in valid_ext)
    ]

    files.sort()

    print("Valid images found:", len(files))

    all_features = []

    for file_name in files:
        image_path = os.path.join(IMAGE_DIR, file_name)

        root_mask_path = find_mask_file(
            ROOT_MASK_DIR,
            image_file_name=file_name,
            prefix="polynomial_root_"
        )

        pdl_mask_path = find_mask_file(
            PDL_MASK_DIR,
            image_file_name=file_name,
            prefix="dark_pdl_"
        )

        if root_mask_path is None:
            print("SKIPPED: missing root mask for", file_name)
            continue

        if pdl_mask_path is None:
            print("SKIPPED: missing PDL mask for", file_name)
            continue

        name, ext = os.path.splitext(file_name)

        debug_path = os.path.join(
            DEBUG_DIR,
            f"debug_pdl_pattern_{name}.png"
        )

        features = extract_pdl_pattern_features(
            image_path=image_path,
            root_mask_path=root_mask_path,
            pdl_mask_path=pdl_mask_path,
            debug_path=debug_path,
            upscale_factor=2
        )

        if features is not None:
            all_features.append(features)

    save_features_csv(
        features_list=all_features,
        csv_path=OUTPUT_CSV_PATH
    )

    merge_with_dentist_labels(
        feature_csv_path=OUTPUT_CSV_PATH,
        label_csv_path=DENTIST_LABEL_CSV_PATH,
        output_csv_path=MERGED_OUTPUT_CSV_PATH
    )

    print("Done.")
    print("Feature CSV:", OUTPUT_CSV_PATH)
    print("Debug images:", DEBUG_DIR)


if __name__ == "__main__":
    run()