import os

from src.fractures.image_processing.pdl_from_final_mask_polynomial import (
    extract_polynomial_root_and_dark_pdl_from_final_mask
)


IMAGE_DIR = "data/fractures/processed/anatomical_region"

# Only mask input used by this algorithm
FINAL_MASK_DIR = "data/fractures/processed/bone_pdl_root_region_masks"

# New outputs
POLYNOMIAL_ROOT_MASK_DIR = "data/fractures/processed/polynomial_root_masks"
DARK_PDL_MASK_DIR = "data/fractures/processed/dark_pdl_from_polynomial_root_masks"
DEBUG_DIR = "data/fractures/processed/debug_polynomial_root_dark_pdl"


def run():
    print("Starting polynomial root + complete dark PDL detection...")
    print("IMAGE_DIR:", IMAGE_DIR)
    print("FINAL_MASK_DIR:", FINAL_MASK_DIR)

    if not os.path.exists(IMAGE_DIR):
        print("ERROR: IMAGE_DIR does not exist.")
        return

    if not os.path.exists(FINAL_MASK_DIR):
        print("ERROR: FINAL_MASK_DIR does not exist.")
        return

    os.makedirs(POLYNOMIAL_ROOT_MASK_DIR, exist_ok=True)
    os.makedirs(DARK_PDL_MASK_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    files = [
        file_name for file_name in os.listdir(IMAGE_DIR)
        if any(file_name.lower().endswith(ext) for ext in valid_ext)
    ]

    files.sort()

    print("Valid image files found:", len(files))

    for file_name in files:
        image_path = os.path.join(IMAGE_DIR, file_name)

        name, ext = os.path.splitext(file_name)

        # Must match the output naming from your previous ROI algorithm:
        # bone_pdl_root_region_{name}{ext}
        final_mask_file_name = f"bone_pdl_root_region_{name}{ext}"

        final_mask_path = os.path.join(
            FINAL_MASK_DIR,
            final_mask_file_name
        )

        if not os.path.exists(final_mask_path):
            print(f"SKIPPED: missing final mask for {file_name}")
            print("Expected:", final_mask_path)
            continue

        root_mask_path = os.path.join(
            POLYNOMIAL_ROOT_MASK_DIR,
            f"polynomial_root_{name}{ext}"
        )

        pdl_mask_path = os.path.join(
            DARK_PDL_MASK_DIR,
            f"dark_pdl_{name}{ext}"
        )

        debug_path = os.path.join(
            DEBUG_DIR,
            f"debug_polynomial_root_dark_pdl_{name}{ext}"
        )

        extract_polynomial_root_and_dark_pdl_from_final_mask(
            image_path=image_path,
            final_mask_path=final_mask_path,

            root_mask_path=root_mask_path,
            pdl_mask_path=pdl_mask_path,
            debug_path=debug_path,

            # Bright root detection
            root_percentile=8,

            # Polynomial root boundary
            polynomial_degree=3,
            root_end_half_width_px=2,
            apex_synthetic_weight=10,
            root_taper_ratio=0.13,

            # Real root end from bright root area
            apex_padding_px=4,

            # Wider search for uneven/asymmetric PDL
            min_pdl_search_px=1,
            max_pdl_search_px=24,

            # Complete dark-region detection
            expected_pdl_px=6,
            max_pdl_px=60,
            dark_percentile=70,
            adaptive_offset=0.035,
            probability_threshold=0.10,
            gap_tolerance_px=2,

            # Cleanup
            min_component_area_px=8
        )

        print(f"Processed: {file_name}")

    print("Done.")
    print("Polynomial root masks saved to:", POLYNOMIAL_ROOT_MASK_DIR)
    print("Dark PDL masks saved to:", DARK_PDL_MASK_DIR)
    print("Debug images saved to:", DEBUG_DIR)


if __name__ == "__main__":
    run()