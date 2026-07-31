from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fractures.feature_extraction.pdl_width_features import (
    PDLFeatureConfig,
    extract_pdl_features,
    features_to_dict,
    load_binary_mask,
)


# ============================================================
# Input directories
# ============================================================

ROOT_MASK_DIR = Path(
    "data/fractures/processed/polynomial_root_masks"
)

PDL_MASK_DIR = Path(
    "data/fractures/processed/dark_pdl_from_polynomial_root_masks"
)


# ============================================================
# Output directories
# ============================================================

OUTPUT_DIR = Path(
    "data/fractures/processed/pdl_width_feature_extraction_v2"
)

PROFILE_DIR = (
    OUTPUT_DIR
    / "profiles"
)

DEBUG_DIR = (
    OUTPUT_DIR
    / "debug_profiles"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "pdl_width_features_v2.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIG = PDLFeatureConfig(

    max_scan_root_ratio=0.50,

    min_scan_px=5,

    max_scan_cap_px=80,

    gap_tolerance=2,

    median_kernel_size=3,

    dominance_margin_ratio=0.03,

    local_window_radius=3,

    local_ratio_threshold=1.35,

    local_absolute_margin_ratio=0.03,

    asymmetry_ratio_threshold=1.30,

    asymmetry_absolute_margin_ratio=0.025,

    min_valid_rows=10,
)


# ============================================================
# Valid image formats
# ============================================================

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Sample ID
# ============================================================

def extract_sample_id(
    filename: str,
) -> str:

    stem = Path(
        filename
    ).stem.lower()

    match = re.search(
        r"frac\d+",
        stem,
    )

    if match:
        return match.group(0)

    return stem


# ============================================================
# Match root and PDL files
# ============================================================

def find_matching_file(
    directory: Path,
    sample_id: str,
):

    sample_id = (
        sample_id.lower()
    )

    matches = []

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in VALID_EXTENSIONS
        ):
            continue

        if (
            sample_id
            in path.stem.lower()
        ):
            matches.append(
                path
            )

    if not matches:
        return None

    matches = sorted(
        matches
    )

    if len(matches) > 1:

        print(
            f"  WARNING: Multiple PDL masks "
            f"found for {sample_id}"
        )

        for path in matches:
            print(
                "   ",
                path.name,
            )

        print(
            "  Using:",
            matches[0].name,
        )

    return matches[0]


# ============================================================
# Save row-wise profile
# ============================================================

def save_profile_csv(
    sample_id: str,
    profiles: dict,
):

    profile_df = pd.DataFrame(
        {

            "y_position":
                profiles[
                    "y_positions"
                ],

            "normalized_y":
                profiles[
                    "normalized_y"
                ],

            "root_width_px":
                profiles[
                    "root_width_px"
                ],

            "root_width_smoothed_px":
                profiles[
                    "root_width_smoothed_px"
                ],

            "left_width_raw_px":
                profiles[
                    "left_raw_px"
                ],

            "right_width_raw_px":
                profiles[
                    "right_raw_px"
                ],

            "left_width_smoothed_px":
                profiles[
                    "left_smoothed_px"
                ],

            "right_width_smoothed_px":
                profiles[
                    "right_smoothed_px"
                ],

            "overall_width_px":
                profiles[
                    "overall_width_px"
                ],

            "asymmetry_px":
                profiles[
                    "asymmetry_px"
                ],

            "left_width_normalized":
                profiles[
                    "left_normalized"
                ],

            "right_width_normalized":
                profiles[
                    "right_normalized"
                ],

            "overall_width_normalized":
                profiles[
                    "overall_normalized"
                ],

            "asymmetry_normalized":
                profiles[
                    "asymmetry_normalized"
                ],

            "side_profile":
                profiles[
                    "side_profile"
                ],

            "width_spike":
                profiles[
                    "width_spike_flags"
                ].astype(int),

            "asymmetry_spike":
                profiles[
                    "asymmetry_spike_flags"
                ].astype(int),
        }
    )

    output_path = (
        PROFILE_DIR
        / f"{sample_id}_profile.csv"
    )

    profile_df.to_csv(
        output_path,
        index=False,
    )


# ============================================================
# RAW width debug plot
# ============================================================

def save_raw_width_plot(
    sample_id: str,
    profiles: dict,
):

    y = profiles[
        "normalized_y"
    ]

    left = profiles[
        "left_smoothed_px"
    ]

    right = profiles[
        "right_smoothed_px"
    ]

    overall = profiles[
        "overall_width_px"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.plot(
        left,
        y,
        label="Left PDL"
    )

    ax.plot(
        right,
        y,
        label="Right PDL"
    )

    ax.plot(
        overall,
        y,
        label="Mean PDL"
    )

    ax.set_xlabel(
        "Raw Width (pixels)"
    )

    ax.set_ylabel(
        "Normalized Root Position"
    )

    ax.set_title(
        f"Raw PDL Width Profile - {sample_id}"
    )

    ax.invert_yaxis()

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        DEBUG_DIR
        / f"{sample_id}_raw_width.png",
        dpi=150,
    )

    plt.close(
        fig
    )


# ============================================================
# NORMALIZED width debug plot
# ============================================================

def save_normalized_width_plot(
    sample_id: str,
    profiles: dict,
):

    y = profiles[
        "normalized_y"
    ]

    left = profiles[
        "left_normalized"
    ]

    right = profiles[
        "right_normalized"
    ]

    overall = profiles[
        "overall_normalized"
    ]

    spike_flags = profiles[
        "width_spike_flags"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.plot(
        left,
        y,
        label="Left Normalized PDL"
    )

    ax.plot(
        right,
        y,
        label="Right Normalized PDL"
    )

    ax.plot(
        overall,
        y,
        label="Mean Normalized PDL"
    )

    if np.any(
        spike_flags
    ):

        ax.scatter(
            overall[
                spike_flags
            ],
            y[
                spike_flags
            ],
            marker="x",
            label="Localized Width Spike",
        )

    ax.set_xlabel(
        "PDL Width / Local Root Width"
    )

    ax.set_ylabel(
        "Normalized Root Position"
    )

    ax.set_title(
        f"Normalized PDL Width Profile - {sample_id}"
    )

    ax.invert_yaxis()

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        DEBUG_DIR
        / f"{sample_id}_normalized_width.png",
        dpi=150,
    )

    plt.close(
        fig
    )


# ============================================================
# NORMALIZED asymmetry plot
# ============================================================

def save_normalized_asymmetry_plot(
    sample_id: str,
    profiles: dict,
):

    y = profiles[
        "normalized_y"
    ]

    asymmetry = profiles[
        "asymmetry_normalized"
    ]

    spikes = profiles[
        "asymmetry_spike_flags"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.plot(
        asymmetry,
        y,
        label="Normalized Asymmetry"
    )

    if np.any(
        spikes
    ):

        ax.scatter(
            asymmetry[
                spikes
            ],
            y[
                spikes
            ],
            marker="x",
            label="Asymmetry Spike",
        )

    ax.set_xlabel(
        "Normalized |Left - Right|"
    )

    ax.set_ylabel(
        "Normalized Root Position"
    )

    ax.set_title(
        f"Normalized PDL Asymmetry - {sample_id}"
    )

    ax.invert_yaxis()

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        DEBUG_DIR
        / f"{sample_id}_normalized_asymmetry.png",
        dpi=150,
    )

    plt.close(
        fig
    )


# ============================================================
# Save all plots
# ============================================================

def save_debug_plots(
    sample_id: str,
    profiles: dict,
):

    save_raw_width_plot(
        sample_id,
        profiles,
    )

    save_normalized_width_plot(
        sample_id,
        profiles,
    )

    save_normalized_asymmetry_plot(
        sample_id,
        profiles,
    )


# ============================================================
# Main
# ============================================================

def run():

    print()
    print(
        "Starting resolution-normalized PDL feature extraction..."
    )

    print(
        "ROOT_MASK_DIR:",
        ROOT_MASK_DIR
    )

    print(
        "PDL_MASK_DIR:",
        PDL_MASK_DIR
    )

    print(
        "OUTPUT_CSV:",
        OUTPUT_CSV
    )

    # --------------------------------------------------------
    # Directory validation
    # --------------------------------------------------------

    if not ROOT_MASK_DIR.exists():

        print(
            "ERROR: Root mask directory does not exist."
        )

        return

    if not PDL_MASK_DIR.exists():

        print(
            "ERROR: PDL mask directory does not exist."
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Root files
    # --------------------------------------------------------

    root_files = sorted(
        [
            path
            for path
            in ROOT_MASK_DIR.iterdir()

            if (
                path.is_file()
                and path.suffix.lower()
                in VALID_EXTENSIONS
            )
        ]
    )

    print()
    print(
        "Root masks found:",
        len(root_files)
    )

    if not root_files:

        print(
            "ERROR: No root masks found."
        )

        return

    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    all_features = []

    success_count = 0
    failure_count = 0

    for index, root_path in enumerate(
        root_files,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(root_files)}] "
            f"Processing: {root_path.name}"
        )

        sample_id = extract_sample_id(
            root_path.name
        )

        print(
            "  Sample ID:",
            sample_id
        )

        pdl_path = find_matching_file(
            PDL_MASK_DIR,
            sample_id,
        )

        if pdl_path is None:

            print(
                "  SKIPPED: Matching PDL mask not found."
            )

            failure_count += 1
            continue

        print(
            "  Root:",
            root_path.name
        )

        print(
            "  PDL:",
            pdl_path.name
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        try:

            root_mask = load_binary_mask(
                root_path
            )

            pdl_mask = load_binary_mask(
                pdl_path
            )

        except Exception as exc:

            print(
                "  ERROR loading masks:",
                exc
            )

            failure_count += 1
            continue

        print(
            "  Shape:",
            root_mask.shape
        )

        if (
            root_mask.shape
            != pdl_mask.shape
        ):

            print(
                "  ERROR: Mask dimensions differ."
            )

            failure_count += 1
            continue

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        try:

            features, profiles = (
                extract_pdl_features(
                    root_mask=root_mask,
                    pdl_mask=pdl_mask,
                    image_name=sample_id,
                    config=CONFIG,
                )
            )

        except Exception as exc:

            print(
                "  ERROR extracting features:",
                exc
            )

            failure_count += 1
            continue

        row = features_to_dict(
            features
        )

        all_features.append(
            row
        )

        # ----------------------------------------------------
        # Save details
        # ----------------------------------------------------

        save_profile_csv(
            sample_id,
            profiles,
        )

        save_debug_plots(
            sample_id,
            profiles,
        )

        # ----------------------------------------------------
        # Terminal summary
        # ----------------------------------------------------

        print(
            "  Valid rows:",
            features.valid_row_count
        )

        print(
            "  Mean root width:",
            round(
                features.mean_root_width_px,
                3,
            )
        )

        print(
            "  Raw mean left:",
            round(
                features.mean_left_width_px,
                3,
            )
        )

        print(
            "  Raw mean right:",
            round(
                features.mean_right_width_px,
                3,
            )
        )

        print(
            "  Normalized mean left:",
            round(
                features.mean_left_width_normalized,
                4,
            )
        )

        print(
            "  Normalized mean right:",
            round(
                features.mean_right_width_normalized,
                4,
            )
        )

        print(
            "  Normalized variance:",
            round(
                features.variance_overall_width_normalized,
                6,
            )
        )

        print(
            "  Normalized mean asymmetry:",
            round(
                features.mean_asymmetry_normalized,
                4,
            )
        )

        print(
            "  Normalized L-R difference:",
            round(
                features.left_right_mean_difference_normalized,
                4,
            )
        )

        print(
            "  Dominant side:",
            features.dominant_side
        )

        print(
            "  Side dominant ratio:",
            round(
                features.side_dominant_length_ratio,
                3,
            )
        )

        print(
            "  Side consistency:",
            round(
                features.side_consistency,
                3,
            )
        )

        print(
            "  Width spike ratio:",
            round(
                features.width_spike_ratio,
                3,
            )
        )

        print(
            "  Asymmetry spike ratio:",
            round(
                features.asymmetry_spike_ratio,
                3,
            )
        )

        success_count += 1

    # --------------------------------------------------------
    # Final CSV
    # --------------------------------------------------------

    if not all_features:

        print()
        print(
            "No features were extracted."
        )

        return

    df = pd.DataFrame(
        all_features
    )

    # Sort frac01 -> frac34
    def sort_key(value):

        match = re.search(
            r"\d+",
            str(value),
        )

        if match:
            return int(
                match.group(0)
            )

        return 999999

    df["_sort"] = (
        df[
            "image_name"
        ].apply(
            sort_key
        )
    )

    df = (
        df
        .sort_values(
            "_sort"
        )
        .drop(
            columns=[
                "_sort"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Resolution-normalized PDL feature extraction completed."
    )

    print(
        "Successful samples:",
        success_count
    )

    print(
        "Failed / skipped:",
        failure_count
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(
            df.columns
        )
    )

    print()
    print(
        "Feature dataset:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Profiles:"
    )

    print(
        PROFILE_DIR
    )

    print()
    print(
        "Debug plots:"
    )

    print(
        DEBUG_DIR
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()