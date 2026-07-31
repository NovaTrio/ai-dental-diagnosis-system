from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.fractures.risk_assessment.fracture_risk_assessment import (
    assess_fracture_risk,
)


# ============================================================
# Input
# ============================================================

INPUT_CSV = Path(
    "data/fractures/processed/"
    "pdl_width_feature_extraction_v2/"
    "rule_classifier_final/"
    "pdl_rule_predictions_final.csv"
)


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = Path(
    "data/fractures/processed/"
    "fracture_risk_assessment"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "fracture_risk_predictions.csv"
)


# ============================================================
# Main
# ============================================================

def run() -> None:

    print()
    print("Starting fracture risk assessment...")
    print("INPUT_CSV:", INPUT_CSV)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        print()
        print("ERROR: Input CSV not found:")
        print(INPUT_CSV)

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load classifier predictions
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = [
        "image_name",
        "predicted_pdl_pattern_score",
        "predicted_pdl_pattern_label",
        "prediction_confidence_margin",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print()
        print("ERROR: Missing required columns:")

        for column in missing:
            print(" -", column)

        return

    # ========================================================
    # Create CLEAN output dataset
    # ========================================================

    output_rows = []

    for _, row in df.iterrows():

        pdl_score = int(
            row[
                "predicted_pdl_pattern_score"
            ]
        )

        risk = assess_fracture_risk(
            pdl_score
        )

        output_rows.append(
            {
                "image_name":
                    row["image_name"],

                "predicted_pdl_pattern_score":
                    pdl_score,

                "predicted_pdl_pattern_label":
                    row[
                        "predicted_pdl_pattern_label"
                    ],

                "prediction_confidence_margin":
                    float(
                        row[
                            "prediction_confidence_margin"
                        ]
                    ),

                "fracture_risk_score":
                    risk.fracture_risk_score,

                "fracture_risk_label":
                    risk.fracture_risk_label,

                "fracture_risk_explanation":
                    risk.explanation,
            }
        )

    # --------------------------------------------------------
    # Create compact DataFrame
    # --------------------------------------------------------

    output_df = pd.DataFrame(
        output_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)

    print(
        "Fracture risk assessment completed."
    )

    print()
    print(
        "Samples:",
        len(output_df)
    )

    print(
        "Output columns:",
        len(output_df.columns)
    )

    print()
    print(
        "Fracture risk distribution:"
    )

    print(
        output_df[
            "fracture_risk_label"
        ]
        .value_counts()
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_CSV
    )


if __name__ == "__main__":
    run()