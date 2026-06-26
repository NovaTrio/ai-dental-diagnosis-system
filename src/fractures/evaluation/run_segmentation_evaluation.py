from src.fractures.evaluation.segmentation_evaluation import (
    run_segmentation_evaluation
)


IMAGE_DIR = "data/fractures/processed/anatomical_region"

# Automatic masks
PRED_PDL_MASK_DIR = "data/fractures/processed/dark_pdl_from_polynomial_root_masks"
PRED_ROOT_MASK_DIR = "data/fractures/processed/polynomial_root_masks"

# Manual ground-truth masks
GT_PDL_MASK_DIR = "data/fractures/manual/pdl_masks"
GT_ROOT_MASK_DIR = "data/fractures/manual/root_masks"

# Outputs
OUTPUT_CSV_PATH = "data/fractures/evaluation/segmentation_evaluation_details.csv"
SUMMARY_CSV_PATH = "data/fractures/evaluation/segmentation_evaluation_summary.csv"
DEBUG_DIR = "data/fractures/evaluation/debug_segmentation_evaluation"


def run():
    run_segmentation_evaluation(
        image_dir=IMAGE_DIR,

        pred_pdl_mask_dir=PRED_PDL_MASK_DIR,
        gt_pdl_mask_dir=GT_PDL_MASK_DIR,

        pred_root_mask_dir=PRED_ROOT_MASK_DIR,
        gt_root_mask_dir=GT_ROOT_MASK_DIR,

        output_csv_path=OUTPUT_CSV_PATH,
        summary_csv_path=SUMMARY_CSV_PATH,
        debug_dir=DEBUG_DIR,

        evaluate_pdl=True,
        evaluate_root=True
    )


if __name__ == "__main__":
    run()