import os
import json
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error
)


# ============================================================
# Paths
# ============================================================

INPUT_CSV_PATH = "data/fractures/processed/pdl_pattern_features_v2_with_dentist_labels.csv"

OUTPUT_DIR = "data/fractures/evaluation/pdl_feature_extraction"

THRESHOLD_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "calibrated_pdl_rule_thresholds.json"
)

PREDICTION_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "calibrated_pdl_pattern_predictions.csv"
)

METRICS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "calibrated_pdl_pattern_metrics.json"
)

CONFUSION_MATRIX_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "calibrated_pdl_pattern_confusion_matrix.csv"
)

MISCLASSIFIED_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "calibrated_pdl_pattern_misclassified_cases.csv"
)


# ============================================================
# Label meanings
# ============================================================

PDL_PATTERN_LABELS = {
    1: "Uniform widening",
    2: "Side-dominant widening",
    3: "Irregular localized widening"
}

FRACTURE_PROBABILITY_FROM_PATTERN = {
    1: 1,  # Low
    2: 2,  # Moderate
    3: 3   # High
}

FRACTURE_PROBABILITY_LABELS = {
    0: "Very low",
    1: "Low",
    2: "Moderate",
    3: "High"
}

RCT_SUCCESS_FROM_PATTERN = {
    1: 1.0,  # High
    2: 0.5,  # Moderate
    3: 0.0   # Low
}

RCT_SUCCESS_LABELS = {
    0.0: "Low",
    0.5: "Moderate",
    1.0: "High"
}


# ============================================================
# Utility functions
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_label_column(df):
    """
    Try to find the dentist PDL pattern label column automatically.
    """

    candidate_cols = [
        "dentist_pdl_pattern_score",
        "dentist_pdl_pattern",
        "pdl_pattern_score_dentist",
        "pdl_pattern_dentist",
        "manual_pdl_pattern_score",
        "manual_pdl_pattern",
        "pdl_pattern_score",
        "PDL Pattern",
        "pdl_pattern"
    ]

    for col in candidate_cols:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find dentist PDL pattern label column. "
        "Expected one of: " + ", ".join(candidate_cols)
    )


def safe_numeric(df, col, default=0.0):
    """
    Return numeric column if available, otherwise a default-valued Series.
    """

    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    return pd.Series(default, index=df.index)


def safe_divide(a, b, eps=1e-6):
    return a / (b + eps)


def minmax(series):
    """
    Min-max normalize into 0-1.
    If constant, return zeros.
    """

    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mn = series.min()
    mx = series.max()

    if abs(mx - mn) < 1e-9:
        return pd.Series(0.0, index=series.index)

    return (series - mn) / (mx - mn)


def weighted_score(df, weighted_columns):
    """
    weighted_columns:
        list of tuples: (column_name, weight)

    Only uses columns that exist.
    """

    total_weight = 0.0
    score = pd.Series(0.0, index=df.index)

    for col, weight in weighted_columns:
        if col not in df.columns:
            continue

        score += minmax(df[col]) * weight
        total_weight += weight

    if total_weight <= 0:
        return pd.Series(0.0, index=df.index)

    return score / total_weight


def pattern_to_label(pattern_score):
    return PDL_PATTERN_LABELS.get(int(pattern_score), "Unknown")


def fracture_score_to_label(score):
    return FRACTURE_PROBABILITY_LABELS.get(int(score), "Unknown")


def rct_score_to_label(score):
    return RCT_SUCCESS_LABELS.get(float(score), "Unknown")


# ============================================================
# Feature engineering
# ============================================================

def add_calibration_features(df):
    """
    Adds stronger pattern-separation features using the existing feature CSV.

    These are still classical image-processing / tabular features.
    No pretrained model is used.
    """

    mean_left = safe_numeric(df, "mean_left_width_px")
    mean_right = safe_numeric(df, "mean_right_width_px")
    mean_total = safe_numeric(df, "mean_total_width_px")
    median_total = safe_numeric(df, "median_total_width_px")
    p95_total = safe_numeric(df, "p95_total_width_px")
    max_total = safe_numeric(df, "max_total_width_px")

    width_cv = safe_numeric(df, "width_cv")
    width_std = safe_numeric(df, "width_std_px")
    irregularity_index = safe_numeric(df, "irregularity_index")
    asymmetry_ratio = safe_numeric(df, "asymmetry_ratio")
    mean_asym = safe_numeric(df, "mean_asymmetry_px")
    max_asym = safe_numeric(df, "max_asymmetry_px")

    localized_widening = safe_numeric(df, "localized_widening_percent")
    localized_asymmetry = safe_numeric(df, "localized_asymmetry_percent")

    side_difference = safe_numeric(df, "side_difference_px")
    side_dominance_ratio = safe_numeric(df, "side_dominance_ratio")
    side_consistency = safe_numeric(df, "side_consistency")

    p95_to_median = safe_numeric(df, "p95_to_median_width_ratio")
    max_to_mean = safe_numeric(df, "max_to_mean_width_ratio")

    # --------------------------------------------------------
    # Basic derived geometry features
    # --------------------------------------------------------

    df["calib_abs_side_difference_px"] = np.abs(mean_left - mean_right)

    df["calib_side_balance_error"] = safe_divide(
        np.abs(mean_left - mean_right),
        mean_total
    )

    df["calib_side_width_ratio"] = safe_divide(
        np.maximum(mean_left, mean_right),
        np.minimum(mean_left, mean_right)
    )

    df["calib_localized_peak_strength"] = safe_divide(
        p95_total - median_total,
        median_total
    )

    df["calib_global_peak_strength"] = safe_divide(
        max_total - mean_total,
        mean_total
    )

    df["calib_asymmetry_strength"] = safe_divide(
        mean_asym,
        mean_total
    )

    df["calib_max_asymmetry_strength"] = safe_divide(
        max_asym,
        mean_total
    )

    # --------------------------------------------------------
    # Pattern-specific scores
    # --------------------------------------------------------

    # Irregular localized widening:
    # high variation, high localized widening, high spikes, high asymmetry
    df["calib_irregular_score"] = weighted_score(
        df.assign(
            _width_cv=width_cv,
            _width_std=width_std,
            _irregularity_index=irregularity_index,
            _localized_widening=localized_widening,
            _localized_asymmetry=localized_asymmetry,
            _p95_to_median=p95_to_median,
            _max_to_mean=max_to_mean,
            _asymmetry_ratio=asymmetry_ratio,
            _localized_peak_strength=df["calib_localized_peak_strength"],
            _global_peak_strength=df["calib_global_peak_strength"],
            _max_asymmetry_strength=df["calib_max_asymmetry_strength"],
        ),
        [
            ("_width_cv", 0.14),
            ("_width_std", 0.08),
            ("_irregularity_index", 0.16),
            ("_localized_widening", 0.15),
            ("_localized_asymmetry", 0.15),
            ("_p95_to_median", 0.10),
            ("_max_to_mean", 0.08),
            ("_asymmetry_ratio", 0.08),
            ("_localized_peak_strength", 0.10),
            ("_global_peak_strength", 0.08),
            ("_max_asymmetry_strength", 0.10),
        ]
    )

    # Side-dominant widening:
    # one side consistently wider, but not necessarily locally spiky
    df["calib_side_dominance_score"] = weighted_score(
        df.assign(
            _side_difference=side_difference,
            _abs_side_difference=df["calib_abs_side_difference_px"],
            _side_balance_error=df["calib_side_balance_error"],
            _side_width_ratio=df["calib_side_width_ratio"],
            _side_dominance_ratio=side_dominance_ratio,
            _side_consistency=side_consistency,
            _asymmetry_ratio=asymmetry_ratio,
            _asymmetry_strength=df["calib_asymmetry_strength"],
        ),
        [
            ("_side_difference", 0.12),
            ("_abs_side_difference", 0.12),
            ("_side_balance_error", 0.16),
            ("_side_width_ratio", 0.10),
            ("_side_dominance_ratio", 0.18),
            ("_side_consistency", 0.20),
            ("_asymmetry_ratio", 0.06),
            ("_asymmetry_strength", 0.06),
        ]
    )

    # Uniform widening:
    # low variation, low side dominance, low localized spikes
    irregular_norm = minmax(df["calib_irregular_score"])
    side_norm = minmax(df["calib_side_dominance_score"])
    width_cv_norm = minmax(width_cv)
    localized_norm = minmax(localized_widening)
    asym_norm = minmax(asymmetry_ratio)

    df["calib_uniformity_score"] = (
        1.0
        - (
            0.30 * irregular_norm
            + 0.25 * side_norm
            + 0.20 * width_cv_norm
            + 0.15 * localized_norm
            + 0.10 * asym_norm
        )
    )

    df["calib_uniformity_score"] = df["calib_uniformity_score"].clip(0, 1)

    return df


# ============================================================
# Rule classifier
# ============================================================

def predict_with_thresholds(df, thresholds):
    """
    Rule order:

    1. Irregular localized widening
    2. Side-dominant widening
    3. Uniform widening

    This follows the clinical meaning:
    - localized abnormal spikes should dominate over simple side dominance
    - stable one-side widening should be side-dominant
    - remaining low-irregularity cases become uniform
    """

    irregular_thr = thresholds["irregular_score_threshold"]
    side_thr = thresholds["side_dominance_score_threshold"]
    side_max_irregular = thresholds["side_max_irregular_score"]
    min_side_consistency = thresholds["min_side_consistency"]
    irregular_margin = thresholds["irregular_over_side_margin"]

    irregular_score = df["calib_irregular_score"]
    side_score = df["calib_side_dominance_score"]

    if "side_consistency" in df.columns:
        side_consistency = pd.to_numeric(
            df["side_consistency"],
            errors="coerce"
        ).fillna(0.0)
    else:
        side_consistency = side_score

    predictions = []

    for i in range(len(df)):
        irr = float(irregular_score.iloc[i])
        side = float(side_score.iloc[i])
        cons = float(side_consistency.iloc[i])

        # Class 3: Irregular localized widening
        if (
            irr >= irregular_thr
            and irr >= side * irregular_margin
        ):
            predictions.append(3)

        # Class 2: Side-dominant widening
        elif (
            side >= side_thr
            and irr <= side_max_irregular
            and cons >= min_side_consistency
        ):
            predictions.append(2)

        # Class 1: Uniform widening
        else:
            predictions.append(1)

    return np.array(predictions, dtype=int)


# ============================================================
# Evaluation
# ============================================================

def evaluate_predictions(y_true, y_pred):
    labels = [1, 2, 3]

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    kappa = cohen_kappa_score(y_true, y_pred)

    try:
        quadratic_kappa = cohen_kappa_score(
            y_true,
            y_pred,
            weights="quadratic"
        )
    except Exception:
        quadratic_kappa = None

    ordinal_mae = mean_absolute_error(y_true, y_pred)
    within_one_accuracy = np.mean(np.abs(y_true - y_pred) <= 1)

    return {
        "accuracy_exact_match": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(kappa),
        "quadratic_weighted_kappa": None if quadratic_kappa is None else float(quadratic_kappa),
        "ordinal_mae": float(ordinal_mae),
        "within_one_score_accuracy": float(within_one_accuracy),
    }


def objective_score(metrics):
    """
    Main optimization target.

    Macro-F1 is prioritized because class balance matters.
    Cohen kappa is used as a secondary agreement score.
    Accuracy is used only as a smaller supporting term.
    """

    return (
        metrics["macro_f1"] * 0.60
        + metrics["cohen_kappa"] * 0.25
        + metrics["accuracy_exact_match"] * 0.15
    )


# ============================================================
# Grid search
# ============================================================

def grid_search_thresholds(df, y_true):
    best = None

    irregular_thresholds = np.round(np.arange(0.25, 0.86, 0.05), 2)
    side_thresholds = np.round(np.arange(0.25, 0.86, 0.05), 2)
    side_max_irregular_values = np.round(np.arange(0.35, 0.96, 0.05), 2)
    min_side_consistency_values = np.round(np.arange(0.10, 0.86, 0.05), 2)
    irregular_margins = np.round(np.arange(0.80, 1.31, 0.05), 2)

    total = (
        len(irregular_thresholds)
        * len(side_thresholds)
        * len(side_max_irregular_values)
        * len(min_side_consistency_values)
        * len(irregular_margins)
    )

    print("Starting threshold calibration...")
    print("Total combinations:", total)

    checked = 0

    for (
        irregular_thr,
        side_thr,
        side_max_irregular,
        min_side_consistency,
        irregular_margin
    ) in itertools.product(
        irregular_thresholds,
        side_thresholds,
        side_max_irregular_values,
        min_side_consistency_values,
        irregular_margins
    ):

        thresholds = {
            "irregular_score_threshold": float(irregular_thr),
            "side_dominance_score_threshold": float(side_thr),
            "side_max_irregular_score": float(side_max_irregular),
            "min_side_consistency": float(min_side_consistency),
            "irregular_over_side_margin": float(irregular_margin),
        }

        y_pred = predict_with_thresholds(df, thresholds)
        metrics = evaluate_predictions(y_true, y_pred)
        score = objective_score(metrics)

        if best is None or score > best["objective_score"]:
            best = {
                "thresholds": thresholds,
                "metrics": metrics,
                "objective_score": float(score),
                "predictions": y_pred
            }

        checked += 1

        if checked % 10000 == 0:
            print(f"Checked {checked}/{total} combinations...")

    return best


# ============================================================
# Main
# ============================================================

def run():
    ensure_output_dir()

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV_PATH}\n"
            "Run pdl_pattern_feature_extraction.py first."
        )

    df = pd.read_csv(INPUT_CSV_PATH)

    print("Loaded:", INPUT_CSV_PATH)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    label_col = find_label_column(df)
    print("Dentist label column:", label_col)

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()

    df[label_col] = df[label_col].astype(int)

    # Keep only valid labels
    df = df[df[label_col].isin([1, 2, 3])].copy()

    if len(df) == 0:
        raise ValueError("No valid dentist PDL pattern labels found.")

    y_true = df[label_col].values.astype(int)

    print("\nDentist label distribution:")
    print(df[label_col].value_counts().sort_index())

    df = add_calibration_features(df)

    best = grid_search_thresholds(df, y_true)

    thresholds = best["thresholds"]
    y_pred = best["predictions"]
    metrics = best["metrics"]

    df["calibrated_pdl_pattern_score"] = y_pred
    df["calibrated_pdl_pattern_label"] = [
        pattern_to_label(x) for x in y_pred
    ]

    df["calibrated_fracture_probability_score"] = [
        FRACTURE_PROBABILITY_FROM_PATTERN[int(x)] for x in y_pred
    ]
    df["calibrated_fracture_probability_label"] = [
        fracture_score_to_label(x)
        for x in df["calibrated_fracture_probability_score"]
    ]

    df["calibrated_rct_success_score"] = [
        RCT_SUCCESS_FROM_PATTERN[int(x)] for x in y_pred
    ]
    df["calibrated_rct_success_label"] = [
        rct_score_to_label(x)
        for x in df["calibrated_rct_success_score"]
    ]

    df["dentist_pdl_pattern_score_used"] = y_true
    df["is_correct_calibrated_pattern"] = (
        df["calibrated_pdl_pattern_score"] == df["dentist_pdl_pattern_score_used"]
    )

    # Save thresholds
    threshold_output = {
        "selected_thresholds": thresholds,
        "objective_score": best["objective_score"],
        "optimized_for": "0.60 macro_f1 + 0.25 cohen_kappa + 0.15 accuracy",
        "label_column_used": label_col,
        "pattern_labels": PDL_PATTERN_LABELS,
    }

    with open(THRESHOLD_JSON_PATH, "w") as f:
        json.dump(threshold_output, f, indent=4)

    # Save metrics
    metrics_output = {
        "metrics": metrics,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[1, 2, 3],
            target_names=[
                "Uniform widening",
                "Side-dominant widening",
                "Irregular localized widening"
            ],
            zero_division=0,
            output_dict=True
        ),
        "selected_thresholds": thresholds,
        "num_samples": int(len(df)),
        "num_correct": int(np.sum(y_true == y_pred)),
        "num_incorrect": int(np.sum(y_true != y_pred)),
    }

    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_output, f, indent=4)

    # Save prediction CSV
    df.to_csv(PREDICTION_CSV_PATH, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])

    cm_df = pd.DataFrame(
        cm,
        index=[
            "dentist_uniform_1",
            "dentist_side_dominant_2",
            "dentist_irregular_3"
        ],
        columns=[
            "pred_uniform_1",
            "pred_side_dominant_2",
            "pred_irregular_3"
        ]
    )

    cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH)

    # Misclassified cases
    misclassified_df = df[
        df["dentist_pdl_pattern_score_used"]
        != df["calibrated_pdl_pattern_score"]
    ].copy()

    misclassified_df.to_csv(MISCLASSIFIED_CSV_PATH, index=False)

    # Console summary
    print("\n============================================================")
    print("BEST CALIBRATED THRESHOLDS")
    print("============================================================")
    print(json.dumps(thresholds, indent=4))

    print("\n============================================================")
    print("CALIBRATED METRICS")
    print("============================================================")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n============================================================")
    print("CONFUSION MATRIX")
    print("============================================================")
    print(cm_df)

    print("\n============================================================")
    print("MISCLASSIFIED CASES")
    print("============================================================")
    print(f"{len(misclassified_df)} out of {len(df)}")

    if "image_name" in misclassified_df.columns:
        print(misclassified_df[
            [
                "image_name",
                "dentist_pdl_pattern_score_used",
                "calibrated_pdl_pattern_score",
                "calib_irregular_score",
                "calib_side_dominance_score",
                "calib_uniformity_score"
            ]
        ])
    else:
        print(misclassified_df[
            [
                "dentist_pdl_pattern_score_used",
                "calibrated_pdl_pattern_score",
                "calib_irregular_score",
                "calib_side_dominance_score",
                "calib_uniformity_score"
            ]
        ])

    print("\nSaved:")
    print("Thresholds:", THRESHOLD_JSON_PATH)
    print("Predictions:", PREDICTION_CSV_PATH)
    print("Metrics:", METRICS_JSON_PATH)
    print("Confusion matrix:", CONFUSION_MATRIX_CSV_PATH)
    print("Misclassified cases:", MISCLASSIFIED_CSV_PATH)


if __name__ == "__main__":
    run()