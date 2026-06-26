from src.fractures.image_processing.pdl_pattern_feature_extraction_v2 import (
    process_dataset_v2
)


IMAGE_DIR = "data/fractures/processed/anatomical_region"

ROOT_MASK_DIR = "data/fractures/processed/polynomial_root_masks"

PDL_MASK_DIR = "data/fractures/processed/dark_pdl_from_polynomial_root_masks"

DEBUG_DIR = "data/fractures/processed/debug_pdl_pattern_features_v2"

OUTPUT_CSV_PATH = "data/fractures/processed/pdl_pattern_features_v2.csv"

DENTIST_LABEL_CSV_PATH = "data/fractures/labels/fracture_dataset_numeric.csv"

MERGED_OUTPUT_CSV_PATH = "data/fractures/processed/pdl_pattern_features_v2_with_dentist_labels.csv"


def run():
    process_dataset_v2(
        image_dir=IMAGE_DIR,
        root_mask_dir=ROOT_MASK_DIR,
        pdl_mask_dir=PDL_MASK_DIR,
        output_csv_path=OUTPUT_CSV_PATH,
        debug_dir=DEBUG_DIR,
        dentist_label_csv_path=DENTIST_LABEL_CSV_PATH,
        merged_output_csv_path=MERGED_OUTPUT_CSV_PATH
    )


if __name__ == "__main__":
    run()