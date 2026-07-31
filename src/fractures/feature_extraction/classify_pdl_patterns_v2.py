from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# Input
# ============================================================

INPUT_CSV = Path(
    "data/fractures/processed/"
    "pdl_width_feature_extraction_v2/"
    "evaluation/"
    "pdl_features_with_dentist_labels.csv"
)


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = Path(
    "data/fractures/processed/"
    "pdl_width_feature_extraction_v2/"
    "rule_classifier_final"
)

PREDICTION_CSV = (
    OUTPUT_DIR
    / "pdl_rule_predictions_final.csv"
)

METRICS_CSV = (
    OUTPUT_DIR
    / "pdl_rule_metrics_final.csv"
)

CONFUSION_MATRIX_CSV = (
    OUTPUT_DIR
    / "pdl_rule_confusion_matrix_final.csv"
)

CONFUSION_MATRIX_PNG = (
    OUTPUT_DIR
    / "pdl_rule_confusion_matrix_final.png"
)


# ============================================================
# Classes
# ============================================================

CLASS_NAMES = {
    1: "Uniform widening",
    2: "Side-dominant widening",
    3: "Irregular localized widening",
}


# ============================================================
# Utility functions
# ============================================================

def clamp01(value: float) -> float:
    """
    Restrict a value to the range [0, 1].
    """

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def normalize_high(
    value: float,
    threshold: float,
    upper: float,
) -> float:
    """
    Larger feature values produce stronger evidence.

    threshold -> score 0
    upper     -> score 1
    """

    if upper <= threshold:
        return 0.0

    return clamp01(
        (value - threshold)
        / (upper - threshold)
    )


def normalize_low(
    value: float,
    threshold: float,
    lower: float = 0.0,
) -> float:
    """
    Smaller feature values produce stronger evidence.

    threshold -> score 0
    lower     -> score 1
    """

    if threshold <= lower:
        return 0.0

    return clamp01(
        (threshold - value)
        / (threshold - lower)
    )


# ============================================================
# Uniform widening score
# ============================================================

def calculate_uniform_score(
    row: pd.Series,
) -> tuple[float, dict]:
    """
    Uniform widening evidence:

    - left and right profiles behave similarly
    - low average asymmetry
    - low asymmetry variation
    - no strong persistent side dominance
    """

    correlation = float(
        row["left_right_correlation"]
    )

    mean_asymmetry = float(
        row["mean_asymmetry_normalized"]
    )

    asymmetry_variance = float(
        row["asymmetry_variance_normalized"]
    )

    side_dominance = float(
        row["side_dominant_length_ratio"]
    )

    # --------------------------------------------------------
    # High correlation between left and right PDL profiles
    # --------------------------------------------------------

    correlation_score = normalize_high(
        correlation,
        threshold=0.55,
        upper=0.90,
    )

    # --------------------------------------------------------
    # Low mean asymmetry
    # --------------------------------------------------------

    low_asymmetry_score = normalize_low(
        mean_asymmetry,
        threshold=0.095,
        lower=0.03,
    )

    # --------------------------------------------------------
    # Low asymmetry variance
    # --------------------------------------------------------

    low_asymmetry_variance_score = normalize_low(
        asymmetry_variance,
        threshold=0.009,
        lower=0.002,
    )

    # --------------------------------------------------------
    # Weak persistent side dominance
    # --------------------------------------------------------

    balanced_score = normalize_low(
        side_dominance,
        threshold=0.65,
        lower=0.20,
    )

    # --------------------------------------------------------
    # Final Uniform score
    # --------------------------------------------------------

    score = (
        0.35 * correlation_score
        + 0.25 * low_asymmetry_score
        + 0.25 * low_asymmetry_variance_score
        + 0.15 * balanced_score
    )

    details = {
        "uniform_correlation_score":
            correlation_score,

        "uniform_low_asymmetry_score":
            low_asymmetry_score,

        "uniform_low_asymmetry_variance_score":
            low_asymmetry_variance_score,

        "uniform_balanced_score":
            balanced_score,
    }

    return (
        float(score),
        details,
    )


# ============================================================
# Side-dominant widening score
# ============================================================

def calculate_side_score(
    row: pd.Series,
) -> tuple[float, dict]:
    """
    Side-dominant widening evidence:

    - one side dominates over a meaningful root region
    - side dominance is consistent
    - left-right difference is noticeable
    - overall width variation remains relatively controlled
    """

    side_consistency = float(
        row["side_consistency"]
    )

    side_dominance = float(
        row["side_dominant_length_ratio"]
    )

    lr_difference = float(
        row[
            "left_right_mean_difference_normalized"
        ]
    )

    width_variance = float(
        row[
            "variance_overall_width_normalized"
        ]
    )

    # --------------------------------------------------------
    # Persistent side dominance
    # --------------------------------------------------------

    dominance_score = normalize_high(
        side_dominance,
        threshold=0.45,
        upper=0.85,
    )

    # --------------------------------------------------------
    # Same side remains dominant
    # --------------------------------------------------------

    consistency_score = normalize_high(
        side_consistency,
        threshold=0.65,
        upper=0.95,
    )

    # --------------------------------------------------------
    # Mean left-right difference
    # --------------------------------------------------------

    difference_score = normalize_high(
        lr_difference,
        threshold=0.025,
        upper=0.12,
    )

    # --------------------------------------------------------
    # Side-dominant should not simply be highly irregular
    # --------------------------------------------------------

    controlled_variance_score = normalize_low(
        width_variance,
        threshold=0.022,
        lower=0.004,
    )

    # --------------------------------------------------------
    # Final Side-Dominant score
    # --------------------------------------------------------

    score = (
        0.35 * dominance_score
        + 0.35 * consistency_score
        + 0.20 * difference_score
        + 0.10 * controlled_variance_score
    )

    details = {
        "side_dominance_score":
            dominance_score,

        "side_consistency_score":
            consistency_score,

        "side_lr_difference_score":
            difference_score,

        "side_controlled_variance_score":
            controlled_variance_score,
    }

    return (
        float(score),
        details,
    )


# ============================================================
# Irregular localized widening score
# ============================================================

def calculate_irregular_score(
    row: pd.Series,
) -> tuple[float, dict]:
    """
    Irregular widening evidence:

    - high mean asymmetry
    - high variation in asymmetry
    - lower agreement between left/right profiles
    - increased width variance
    - less stable side behaviour
    """

    mean_asymmetry = float(
        row["mean_asymmetry_normalized"]
    )

    asymmetry_variance = float(
        row["asymmetry_variance_normalized"]
    )

    correlation = float(
        row["left_right_correlation"]
    )

    width_variance = float(
        row[
            "variance_overall_width_normalized"
        ]
    )

    side_consistency = float(
        row["side_consistency"]
    )

    # --------------------------------------------------------
    # Elevated asymmetry
    # --------------------------------------------------------

    asymmetry_score = normalize_high(
        mean_asymmetry,
        threshold=0.055,
        upper=0.14,
    )

    # --------------------------------------------------------
    # Elevated asymmetry variation
    # --------------------------------------------------------

    asymmetry_variance_score = normalize_high(
        asymmetry_variance,
        threshold=0.0035,
        upper=0.015,
    )

    # --------------------------------------------------------
    # Lower left-right correlation
    # --------------------------------------------------------

    low_correlation_score = normalize_low(
        correlation,
        threshold=0.75,
        lower=0.20,
    )

    # --------------------------------------------------------
    # Higher width variation
    # --------------------------------------------------------

    width_variance_score = normalize_high(
        width_variance,
        threshold=0.006,
        upper=0.030,
    )

    # --------------------------------------------------------
    # Less consistent side behaviour
    # --------------------------------------------------------

    inconsistency_score = normalize_low(
        side_consistency,
        threshold=0.85,
        lower=0.50,
    )

    # --------------------------------------------------------
    # Final Irregular score
    # --------------------------------------------------------

    score = (
        0.30 * asymmetry_score
        + 0.30 * asymmetry_variance_score
        + 0.10 * low_correlation_score
        + 0.25 * width_variance_score
        + 0.05 * inconsistency_score
    )

    details = {
        "irregular_asymmetry_score":
            asymmetry_score,

        "irregular_asymmetry_variance_score":
            asymmetry_variance_score,

        "irregular_low_correlation_score":
            low_correlation_score,

        "irregular_width_variance_score":
            width_variance_score,

        "irregular_inconsistency_score":
            inconsistency_score,
    }

    return (
        float(score),
        details,
    )


# ============================================================
# Final classifier
# ============================================================

def classify_row(
    row: pd.Series,
) -> dict:
    """
    Calculate the evidence for all three patterns and choose
    the pattern with the highest score.
    """

    uniform_score, uniform_details = (
        calculate_uniform_score(
            row
        )
    )

    side_score, side_details = (
        calculate_side_score(
            row
        )
    )

    irregular_score, irregular_details = (
        calculate_irregular_score(
            row
        )
    )

    scores = {
        1: uniform_score,
        2: side_score,
        3: irregular_score,
    }

    predicted_class = max(
        scores,
        key=scores.get,
    )

    ordered_scores = sorted(
        scores.values(),
        reverse=True,
    )

    confidence_margin = float(
        ordered_scores[0]
        - ordered_scores[1]
    )

    result = {
        "uniform_score":
            uniform_score,

        "side_score":
            side_score,

        "irregular_score":
            irregular_score,

        "predicted_pdl_pattern_score":
            predicted_class,

        "predicted_pdl_pattern_label":
            CLASS_NAMES[
                predicted_class
            ],

        "prediction_confidence_margin":
            confidence_margin,
    }

    result.update(
        uniform_details
    )

    result.update(
        side_details
    )

    result.update(
        irregular_details
    )

    return result


# ============================================================
# Confusion matrix plot
# ============================================================

def save_confusion_matrix_plot(
    matrix: np.ndarray,
) -> None:

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix
    )

    labels = [
        "Uniform",
        "Side",
        "Irregular",
    ]

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_yticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Dentist Label"
    )

    ax.set_title(
        "Final Rule-Based PDL Pattern Classification"
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            ax.text(
                j,
                i,
                str(
                    matrix[i, j]
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PNG,
        dpi=150,
    )

    plt.close(
        fig
    )


# ============================================================
# Main runner
# ============================================================

def run() -> None:

    print()
    print(
        "Starting final calibrated rule-based "
        "PDL pattern classification..."
    )

    print(
        "INPUT_CSV:",
        INPUT_CSV
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        print()
        print(
            "ERROR: Input dataset not found:"
        )

        print(
            INPUT_CSV
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = [
        "image_name",
        "pdl_pattern_score",
        "left_right_correlation",
        "mean_asymmetry_normalized",
        "asymmetry_variance_normalized",
        "left_right_mean_difference_normalized",
        "side_dominant_length_ratio",
        "side_consistency",
        "variance_overall_width_normalized",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print(
            "ERROR: Missing required columns:"
        )

        for column in missing_columns:
            print(
                "  -",
                column
            )

        return

    # --------------------------------------------------------
    # Remove incomplete rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    ).copy()

    df[
        "pdl_pattern_score"
    ] = (
        df[
            "pdl_pattern_score"
        ]
        .astype(int)
    )

    print()
    print(
        "Samples:",
        len(df)
    )

    print()
    print(
        "Dentist class distribution:"
    )

    print(
        df[
            "pdl_pattern_score"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # Classification
    # ========================================================

    prediction_rows = []

    for _, row in df.iterrows():

        prediction_rows.append(
            classify_row(
                row
            )
        )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    result_df = pd.concat(
        [
            df.reset_index(
                drop=True
            ),

            prediction_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    result_df[
        "prediction_correct"
    ] = (
        result_df[
            "pdl_pattern_score"
        ]
        ==
        result_df[
            "predicted_pdl_pattern_score"
        ]
    )

    # ========================================================
    # Evaluation
    # ========================================================

    y_true = (
        result_df[
            "pdl_pattern_score"
        ]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        result_df[
            "predicted_pdl_pattern_score"
        ]
        .astype(int)
        .to_numpy()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    (
        class_precision,
        class_recall,
        class_f1,
        class_support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[
            1,
            2,
            3,
        ],
        zero_division=0,
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            1,
            2,
            3,
        ],
    )

    matrix_df = pd.DataFrame(
        matrix,

        index=[
            "Actual Uniform",
            "Actual Side",
            "Actual Irregular",
        ],

        columns=[
            "Predicted Uniform",
            "Predicted Side",
            "Predicted Irregular",
        ],
    )

    # ========================================================
    # Save prediction CSV
    # ========================================================

    result_df.to_csv(
        PREDICTION_CSV,
        index=False,
    )

    matrix_df.to_csv(
        CONFUSION_MATRIX_CSV
    )

    save_confusion_matrix_plot(
        matrix
    )

    # ========================================================
    # Metrics CSV
    # ========================================================

    metric_rows = [
        {
            "metric_scope": "overall",
            "class_id": "",
            "class_name": "Overall",
            "support": len(df),
            "accuracy": accuracy,
            "precision": "",
            "recall": "",
            "f1": "",
        },

        {
            "metric_scope": "macro",
            "class_id": "",
            "class_name": "Macro average",
            "support": len(df),
            "accuracy": "",
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },

        {
            "metric_scope": "weighted",
            "class_id": "",
            "class_name": "Weighted average",
            "support": len(df),
            "accuracy": "",
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
    ]

    for index, class_id in enumerate(
        [
            1,
            2,
            3,
        ]
    ):

        metric_rows.append(
            {
                "metric_scope": "class",

                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "support":
                    int(
                        class_support[
                            index
                        ]
                    ),

                "accuracy":
                    "",

                "precision":
                    float(
                        class_precision[
                            index
                        ]
                    ),

                "recall":
                    float(
                        class_recall[
                            index
                        ]
                    ),

                "f1":
                    float(
                        class_f1[
                            index
                        ]
                    ),
            }
        )

    pd.DataFrame(
        metric_rows
    ).to_csv(
        METRICS_CSV,
        index=False,
    )

    # ========================================================
    # Classification report
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=[
            1,
            2,
            3,
        ],
        target_names=[
            "Uniform",
            "Side-Dominant",
            "Irregular",
        ],
        zero_division=0,
    )

    # ========================================================
    # Terminal output
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "Final calibrated classification completed."
    )

    print()

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Macro Precision:   {macro_precision:.4f}"
    )

    print(
        f"Macro Recall:      {macro_recall:.4f}"
    )

    print(
        f"Macro F1:          {macro_f1:.4f}"
    )

    print(
        f"Weighted F1:       {weighted_f1:.4f}"
    )

    print()
    print(
        "Classification Report:"
    )

    print(
        report
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        matrix_df
    )

    print()
    print(
        "Predicted class distribution:"
    )

    print(
        result_df[
            "predicted_pdl_pattern_score"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # Misclassified cases
    # ========================================================

    wrong_df = result_df[
        ~result_df[
            "prediction_correct"
        ]
    ]

    print()
    print(
        "Misclassified samples:",
        len(
            wrong_df
        )
    )

    if len(
        wrong_df
    ) > 0:

        print()

        print(
            wrong_df[
                [
                    "image_name",
                    "pdl_pattern_score",
                    "predicted_pdl_pattern_score",
                    "uniform_score",
                    "side_score",
                    "irregular_score",
                    "prediction_confidence_margin",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # Lowest confidence
    # ========================================================

    print()
    print(
        "Lowest-confidence predictions:"
    )

    lowest_confidence = (
        result_df
        .sort_values(
            "prediction_confidence_margin",
            ascending=True,
        )
        .head(10)
    )

    print(
        lowest_confidence[
            [
                "image_name",
                "pdl_pattern_score",
                "predicted_pdl_pattern_score",
                "uniform_score",
                "side_score",
                "irregular_score",
                "prediction_confidence_margin",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Output paths
    # ========================================================

    print()
    print(
        "Predictions:"
    )

    print(
        PREDICTION_CSV
    )

    print()
    print(
        "Metrics:"
    )

    print(
        METRICS_CSV
    )

    print()
    print(
        "Confusion matrix CSV:"
    )

    print(
        CONFUSION_MATRIX_CSV
    )

    print()
    print(
        "Confusion matrix image:"
    )

    print(
        CONFUSION_MATRIX_PNG
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run()