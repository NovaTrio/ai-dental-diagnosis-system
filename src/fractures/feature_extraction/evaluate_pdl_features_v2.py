from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Input files
# ============================================================

FEATURE_CSV = Path(
    "data/fractures/processed/"
    "pdl_width_feature_extraction_v2/"
    "pdl_width_features_v2.csv"
)

LABEL_CSV = Path(
    "data/fractures/processed/"
    "pdl_dentist_labels_clean.csv"
)


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = Path(
    "data/fractures/processed/"
    "pdl_width_feature_extraction_v2/evaluation"
)

MERGED_CSV = OUTPUT_DIR / "pdl_features_with_dentist_labels.csv"

SUMMARY_CSV = OUTPUT_DIR / "feature_summary_by_class.csv"

SEPARABILITY_CSV = OUTPUT_DIR / "feature_separability_ranking.csv"

PLOT_DIR = OUTPUT_DIR / "feature_plots"


# ============================================================
# Important features
# ============================================================

IMPORTANT_FEATURES = [

    "variance_overall_width_normalized",

    "mean_asymmetry_normalized",

    "asymmetry_variance_normalized",

    "left_right_mean_difference_normalized",

    "left_dominant_ratio",

    "right_dominant_ratio",

    "side_dominant_length_ratio",

    "side_consistency",

    "width_spike_ratio",

    "asymmetry_spike_ratio",

    "left_right_correlation",
]


# ============================================================
# Utility
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
):

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )


# ============================================================
# Feature summary by dentist class
# ============================================================

def build_class_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in IMPORTANT_FEATURES:

        for class_id in sorted(
            df["pdl_pattern_score"].unique()
        ):

            values = (
                df.loc[
                    df["pdl_pattern_score"] == class_id,
                    feature
                ]
                .dropna()
                .astype(float)
            )

            if len(values) == 0:
                continue

            rows.append({
                "feature": feature,

                "class_id": int(class_id),

                "class_label":
                    df.loc[
                        df["pdl_pattern_score"] == class_id,
                        "pdl_pattern_label"
                    ].iloc[0],

                "count": len(values),

                "mean": float(values.mean()),

                "median": float(values.median()),

                "std": float(values.std(ddof=0)),

                "min": float(values.min()),

                "q25": float(values.quantile(0.25)),

                "q75": float(values.quantile(0.75)),

                "max": float(values.max()),
            })

    return pd.DataFrame(rows)


# ============================================================
# Simple separability score
# ============================================================

def calculate_pairwise_separation(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> float:
    """
    Effect-size-like score.

    Larger value means the two class distributions
    are more separated relative to their internal spread.
    """

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))

    std_a = float(np.std(values_a))
    std_b = float(np.std(values_b))

    pooled_std = np.sqrt(
        (std_a ** 2 + std_b ** 2) / 2.0
    )

    if pooled_std < 1e-8:
        return 0.0

    return abs(
        mean_a - mean_b
    ) / pooled_std


def build_separability_ranking(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in IMPORTANT_FEATURES:

        feature_values = {}

        for class_id in [1, 2, 3]:

            feature_values[class_id] = (
                df.loc[
                    df["pdl_pattern_score"] == class_id,
                    feature
                ]
                .dropna()
                .astype(float)
                .to_numpy()
            )

        if any(
            len(feature_values[c]) == 0
            for c in [1, 2, 3]
        ):
            continue

        sep_1_2 = calculate_pairwise_separation(
            feature_values[1],
            feature_values[2],
        )

        sep_1_3 = calculate_pairwise_separation(
            feature_values[1],
            feature_values[3],
        )

        sep_2_3 = calculate_pairwise_separation(
            feature_values[2],
            feature_values[3],
        )

        mean_separation = np.mean([
            sep_1_2,
            sep_1_3,
            sep_2_3,
        ])

        rows.append({
            "feature": feature,

            "uniform_vs_side": sep_1_2,

            "uniform_vs_irregular": sep_1_3,

            "side_vs_irregular": sep_2_3,

            "mean_separation": mean_separation,
        })

    result = pd.DataFrame(rows)

    return result.sort_values(
        "mean_separation",
        ascending=False,
    )


# ============================================================
# Box plots
# ============================================================

def save_feature_plots(
    df: pd.DataFrame,
):

    class_order = [1, 2, 3]

    class_names = [
        "Uniform",
        "Side-Dominant",
        "Irregular",
    ]

    for feature in IMPORTANT_FEATURES:

        data = []

        for class_id in class_order:

            values = (
                df.loc[
                    df["pdl_pattern_score"] == class_id,
                    feature
                ]
                .dropna()
                .astype(float)
                .to_numpy()
            )

            data.append(values)

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.boxplot(
            data,
            labels=class_names,
        )

        ax.set_title(
            feature
        )

        ax.set_ylabel(
            feature
        )

        ax.grid(
            True,
            axis="y",
        )

        fig.tight_layout()

        output_path = (
            PLOT_DIR
            / f"{feature}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
        )

        plt.close(fig)


# ============================================================
# Main
# ============================================================

def run():

    print()
    print(
        "Starting PDL feature evaluation..."
    )

    print(
        "FEATURE_CSV:",
        FEATURE_CSV
    )

    print(
        "LABEL_CSV:",
        LABEL_CSV
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not FEATURE_CSV.exists():

        print(
            "ERROR: Feature CSV not found."
        )

        return

    if not LABEL_CSV.exists():

        print(
            "ERROR: Label CSV not found."
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    features_df = pd.read_csv(
        FEATURE_CSV
    )

    labels_df = pd.read_csv(
        LABEL_CSV
    )

    validate_columns(
        features_df,
        ["image_name"] + IMPORTANT_FEATURES,
        "Feature dataset",
    )

    validate_columns(
        labels_df,
        [
            "image_name",
            "pdl_pattern_score",
            "pdl_pattern_label",
        ],
        "Dentist label dataset",
    )

    # --------------------------------------------------------
    # Standardize IDs
    # --------------------------------------------------------

    features_df["image_name"] = (
        features_df["image_name"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    labels_df["image_name"] = (
        labels_df["image_name"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged_df = features_df.merge(
        labels_df,
        on="image_name",
        how="inner",
    )

    print()
    print(
        "Feature rows:",
        len(features_df)
    )

    print(
        "Label rows:",
        len(labels_df)
    )

    print(
        "Merged rows:",
        len(merged_df)
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        merged_df[
            "pdl_pattern_score"
        ].value_counts().sort_index()
    )

    merged_df.to_csv(
        MERGED_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Class summaries
    # --------------------------------------------------------

    summary_df = build_class_summary(
        merged_df
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Feature ranking
    # --------------------------------------------------------

    ranking_df = build_separability_ranking(
        merged_df
    )

    ranking_df.to_csv(
        SEPARABILITY_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_feature_plots(
        merged_df
    )

    # --------------------------------------------------------
    # Terminal results
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Feature evaluation completed."
    )

    print()
    print(
        "Top features:"
    )

    print(
        ranking_df[
            [
                "feature",
                "mean_separation"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        "Merged dataset:"
    )

    print(
        MERGED_CSV
    )

    print()
    print(
        "Class summary:"
    )

    print(
        SUMMARY_CSV
    )

    print()
    print(
        "Separability ranking:"
    )

    print(
        SEPARABILITY_CSV
    )

    print()
    print(
        "Plots:"
    )

    print(
        PLOT_DIR
    )


if __name__ == "__main__":
    run()