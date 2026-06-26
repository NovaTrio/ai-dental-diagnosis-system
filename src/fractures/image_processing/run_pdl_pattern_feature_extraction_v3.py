import os
import pandas as pd

from src.fractures.image_processing.pdl_pattern_feature_extraction_v3 import (
    extract_pdl_pattern_features_v3,
    save_features_csv,
    find_mask_file,
)


# ============================================================
# Paths
# ============================================================

IMAGE_DIR = "data/fractures/processed/anatomical_region"

ROOT_MASK_DIR = "data/fractures/processed/polynomial_root_masks"

PDL_MASK_DIR = "data/fractures/processed/dark_pdl_from_polynomial_root_masks"

DEBUG_DIR = "data/fractures/processed/debug_pdl_pattern_features_v3"

OUTPUT_CSV_PATH = "data/fractures/processed/pdl_pattern_features_v3.csv"

DENTIST_LABEL_CSV_PATH = "data/fractures/labels/fracture_dataset_numeric.csv"

MERGED_OUTPUT_CSV_PATH = (
    "data/fractures/processed/"
    "pdl_pattern_features_v3_with_dentist_labels.csv"
)


# ============================================================
# Utilities
# ============================================================

def get_stem(path_or_name):
    return os.path.splitext(os.path.basename(str(path_or_name)))[0].lower()


def find_image_name_column(df):
    candidates = [
        "image_name",
        "image",
        "filename",
        "file_name",
        "Image_Name",
        "Image Name",
        "name",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: use first column
    return df.columns[0]


def merge_with_dentist_labels(feature_csv_path, label_csv_path, output_csv_path):
    """
    Merges extracted V3 features with dentist labels.

    Expected dentist label columns:
        pdl_pattern_score
        fracture_probability_score
        rct_success_score
    """

    if not os.path.exists(feature_csv_path):
        print("Feature CSV not found:", feature_csv_path)
        return

    if not os.path.exists(label_csv_path):
        print("Dentist label CSV not found. Skipping merge:", label_csv_path)
        return

    features_df = pd.read_csv(feature_csv_path)
    labels_df = pd.read_csv(label_csv_path)

    if "image_name" not in features_df.columns:
        print("ERROR: image_name column missing in feature CSV.")
        return

    label_image_col = find_image_name_column(labels_df)

    features_df["_merge_stem"] = features_df["image_name"].apply(get_stem)
    labels_df["_merge_stem"] = labels_df[label_image_col].apply(get_stem)

    merged_df = pd.merge(
        features_df,
        labels_df,
        on="_merge_stem",
        how="left",
        suffixes=("", "_dentist"),
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    merged_df.to_csv(output_csv_path, index=False)

    print("Merged V3 feature CSV saved:")
    print(output_csv_path)

    print("\nMerged rows:", len(merged_df))

    if "pdl_pattern_score" in merged_df.columns:
        missing_labels = merged_df["pdl_pattern_score"].isna().sum()
        print("Missing pdl_pattern_score labels:", missing_labels)

    elif "pdl_pattern_score_dentist" in merged_df.columns:
        missing_labels = merged_df["pdl_pattern_score_dentist"].isna().sum()
        print("Missing pdl_pattern_score labels:", missing_labels)

    else:
        print("WARNING: pdl_pattern_score column not found after merge.")


# ============================================================
# Main runner
# ============================================================

def run():
    print("Starting V3 PDL pattern feature extraction...")
    print("IMAGE_DIR:", IMAGE_DIR)
    print("ROOT_MASK_DIR:", ROOT_MASK_DIR)
    print("PDL_MASK_DIR:", PDL_MASK_DIR)
    print("DEBUG_DIR:", DEBUG_DIR)

    if not os.path.exists(IMAGE_DIR):
        print("ERROR: IMAGE_DIR does not exist:", IMAGE_DIR)
        return

    if not os.path.exists(ROOT_MASK_DIR):
        print("ERROR: ROOT_MASK_DIR does not exist:", ROOT_MASK_DIR)
        return

    if not os.path.exists(PDL_MASK_DIR):
        print("ERROR: PDL_MASK_DIR does not exist:", PDL_MASK_DIR)
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    image_files = sorted(image_files)

    print("\nFound images:", len(image_files))

    all_features = []

    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(IMAGE_DIR, image_file)

        root_mask_path = find_mask_file(image_file, ROOT_MASK_DIR)
        pdl_mask_path = find_mask_file(image_file, PDL_MASK_DIR)

        print(f"\n[{idx}/{len(image_files)}] Processing:", image_file)

        if root_mask_path is None:
            print("  WARNING: Root mask not found.")
            all_features.append(
                {
                    "image_name": image_file,
                    "feature_error": "root_mask_not_found",
                }
            )
            continue

        if pdl_mask_path is None:
            print("  WARNING: PDL mask not found.")
            all_features.append(
                {
                    "image_name": image_file,
                    "feature_error": "pdl_mask_not_found",
                }
            )
            continue

        debug_path = os.path.join(
            DEBUG_DIR,
            os.path.splitext(image_file)[0] + "_debug_v3.png",
        )

        features = extract_pdl_pattern_features_v3(
            image_path=image_path,
            root_mask_path=root_mask_path,
            pdl_mask_path=pdl_mask_path,
            debug_path=debug_path,
        )

        all_features.append(features)

        if features.get("feature_error", ""):
            print("  Feature error:", features.get("feature_error"))
        else:
            print("  Rows:", features.get("num_profile_rows"))
            print("  Uniformity score:", round(features.get("uniformity_score", 0), 4))
            print("  Side dominance score:", round(features.get("side_dominance_score", 0), 4))
            print("  Localized spike score:", round(features.get("localized_spike_score", 0), 4))

    features_df = save_features_csv(
        features_list=all_features,
        output_csv_path=OUTPUT_CSV_PATH,
    )

    print("\nV3 feature extraction complete.")
    print("Saved feature CSV:")
    print(OUTPUT_CSV_PATH)
    print("Rows:", len(features_df))
    print("Columns:", len(features_df.columns))

    merge_with_dentist_labels(
        feature_csv_path=OUTPUT_CSV_PATH,
        label_csv_path=DENTIST_LABEL_CSV_PATH,
        output_csv_path=MERGED_OUTPUT_CSV_PATH,
    )


if __name__ == "__main__":
    run()