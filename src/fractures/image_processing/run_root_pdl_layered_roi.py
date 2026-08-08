import os

from src.fractures.image_processing.root_pdl_layered_roi import (
    extract_root_pdl_layered_roi
)


INPUT_DIR = "data/fractures/processed/anatomical_region"

# Main ROI outputs
ROI_DIR = "data/fractures/processed/bone_pdl_root_region_roi"
DEBUG_DIR = "data/fractures/processed/debug_bone_pdl_root_region"

# Mask outputs required for next stages
FINAL_MASK_DIR = "data/fractures/processed/bone_pdl_root_region_masks"
ROOT_MASK_DIR = "data/fractures/processed/root_region_masks"
PDL_MASK_DIR = "data/fractures/processed/estimated_pdl_region_masks"
OUTER_MASK_DIR = "data/fractures/processed/root_pdl_outer_region_masks"


def run():
    print("Starting bone + PDL + root ROI extraction...")
    print("INPUT_DIR:", INPUT_DIR)

    if not os.path.exists(INPUT_DIR):
        print("ERROR: Input directory does not exist.")
        return

    os.makedirs(ROI_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    os.makedirs(FINAL_MASK_DIR, exist_ok=True)
    os.makedirs(ROOT_MASK_DIR, exist_ok=True)
    os.makedirs(PDL_MASK_DIR, exist_ok=True)
    os.makedirs(OUTER_MASK_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    files = [
        file_name for file_name in os.listdir(INPUT_DIR)
        if any(file_name.lower().endswith(ext) for ext in valid_ext)
    ]

    files.sort()

    print("Valid image files found:", len(files))

    for file_name in files:
        input_path = os.path.join(INPUT_DIR, file_name)

        name, ext = os.path.splitext(file_name)
        output_file_name = f"bone_pdl_root_region_{name}{ext}"

        roi_path = os.path.join(ROI_DIR, output_file_name)
        debug_path = os.path.join(DEBUG_DIR, output_file_name)

        final_mask_path = os.path.join(FINAL_MASK_DIR, output_file_name)
        root_mask_path = os.path.join(ROOT_MASK_DIR, output_file_name)
        pdl_mask_path = os.path.join(PDL_MASK_DIR, output_file_name)
        outer_mask_path = os.path.join(OUTER_MASK_DIR, output_file_name)

        extract_root_pdl_layered_roi(
            image_path=input_path,

            # Save masks
            mask_path=final_mask_path,
            root_mask_path=root_mask_path,
            pdl_mask_path=pdl_mask_path,
            outer_mask_path=outer_mask_path,

            # Save ROI and debug
            roi_path=roi_path,
            debug_path=debug_path,

            # Final ROI space outside the yellow PDL/lamina line.
            outer_space_px=4,

            # Smooth final mask edge.
            edge_smooth_px=5,

            # Apex detection.
            min_end_ratio=0.50,
            max_end_ratio=0.93,
            convergence_ratio=0.36,
            end_padding_px=5,

            # Red root boundary search.
            min_half_width_ratio=0.040,
            max_half_width_ratio=0.36,
            expected_half_width_ratio=0.19,

            # Yellow PDL / lamina search.
            pdl_min_px=1,
            pdl_max_px=16,
            lamina_search_px=20,
            fallback_pdl_width_px=7,

            # Smooth closing near apex.
            root_taper_ratio=0.12,
            outer_taper_ratio=0.12,
            root_end_half_width_px=2,
            outer_end_half_width_px=5
        )

        print(f"Processed: {file_name}")

    print("Done.")
    print("ROIs saved to:", ROI_DIR)
    print("Final ROI masks saved to:", FINAL_MASK_DIR)
    print("Root masks saved to:", ROOT_MASK_DIR)
    print("Estimated PDL masks saved to:", PDL_MASK_DIR)
    print("Outer masks saved to:", OUTER_MASK_DIR)
    print("Debug images saved to:", DEBUG_DIR)


if __name__ == "__main__":
    run()