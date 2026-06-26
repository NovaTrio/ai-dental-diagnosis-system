import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error
)


# ============================================================
# Paths
# ============================================================

FEATURE_CSV_PATH = "data/fractures/processed/pdl_pattern_features.csv"

DENTIST_LABEL_CSV_PATH = "data/fractures/labels/fracture_dataset_numeric.csv"

MERGED_CSV_PATH = "data/fractures/processed/pdl_pattern_features_with_dentist_labels.csv"

OUTPUT_DIR = "data/fractures/evaluation/pdl_feature_extraction"

METRICS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "pdl_feature_extraction_metrics.json"
)

SUMMARY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pdl_feature_extraction_summary.csv"
)

MISCLASSIFIED_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pdl_pattern_misclassified_cases.csv"
)

GROUP_STATS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pdl_pattern_group_statistics.csv"
)

FEATURE_CORRELATION_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pdl_feature_correlations_with_dentist_pattern.csv"
)

CONFUSION_MATRIX_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "pdl_pattern_confusion_matrix.csv"
)


# ============================================================
# Utility functions
# ============================================================

def normalize_file_name(value):
    """
    Normalize file name for safe merging.
    """
    if pd.isna(value):
        return value

    value = str(value).strip()
    value = os.path.basename(value)

    return value


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_or_create_merged_dataset():
    """
    Loads merged feature + dentist label CSV if available.
    Otherwise merges:
        pdl_pattern_features.csv
        fracture_dataset_numeric.csv
    """

    if os.path.exists(MERGED_CSV_PATH):
        print("Loading merged CSV:", MERGED_CSV_PATH)
        df = pd.read_csv(MERGED_CSV_PATH)
        return df

    print("Merged CSV not found. Creating merge...")

    if not os.path.exists(FEATURE_CSV_PATH):
        raise FileNotFoundError(f"Feature CSV not found: {FEATURE_CSV_PATH}")

    if not os.path.exists(DENTIST_LABEL_CSV_PATH):
        raise FileNotFoundError(f"Dentist label CSV not found: {DENTIST_LABEL_CSV_PATH}")

    features_df = pd.read_csv(FEATURE_CSV_PATH)
    labels_df = pd.read_csv(DENTIST_LABEL_CSV_PATH)

    if "image_name" not in features_df.columns:
        if "file_name" in features_df.columns:
            features_df["image_name"] = features_df["file_name"]
        else:
            raise ValueError("Feature CSV must contain image_name or file_name.")

    if "image_name" not in labels_df.columns:
        if "file_name" in labels_df.columns:
            labels_df["image_name"] = labels_df["file_name"]
        else:
            raise ValueError("Dentist label CSV must contain image_name or file_name.")

    features_df["image_name"] = features_df["image_name"].apply(normalize_file_name)
    labels_df["image_name"] = labels_df["image_name"].apply(normalize_file_name)

    rename_map = {}

    if "pdl_pattern_score" in labels_df.columns:
        rename_map["pdl_pattern_score"] = "dentist_pdl_pattern_score"

    if "fracture_probability_score" in labels_df.columns:
        rename_map["fracture_probability_score"] = "dentist_fracture_probability_score"

    if "rct_success_score" in labels_df.columns:
        rename_map["rct_success_score"] = "dentist_rct_success_score"

    labels_df = labels_df.rename(columns=rename_map)

    merged = pd.merge(
        features_df,
        labels_df,
        on="image_name",
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError("No matching image_name values between feature CSV and dentist label CSV.")

    os.makedirs(os.path.dirname(MERGED_CSV_PATH), exist_ok=True)
    merged.to_csv(MERGED_CSV_PATH, index=False)

    print("Saved merged CSV:", MERGED_CSV_PATH)

    return merged


def clean_numeric_column(df, column):
    if column not in df.columns:
        return None

    values = df[column].apply(safe_float)

    return values


# ============================================================
# Evaluation metrics
# ============================================================

def evaluate_discrete_task(
    df,
    true_col,
    pred_col,
    task_name,
    valid_labels,
    label_names=None
):
    """
    Evaluates discrete / ordinal outputs.

    Used for:
        PDL pattern score
        Fracture probability score
        RCT success score
    """

    if true_col not in df.columns:
        print(f"SKIPPED {task_name}: missing true column {true_col}")
        return None, None

    if pred_col not in df.columns:
        print(f"SKIPPED {task_name}: missing predicted column {pred_col}")
        return None, None

    temp = df.copy()

    temp[true_col] = clean_numeric_column(temp, true_col)
    temp[pred_col] = clean_numeric_column(temp, pred_col)

    temp = temp.dropna(subset=[true_col, pred_col])

    if len(temp) == 0:
        print(f"SKIPPED {task_name}: no valid rows.")
        return None, None

    y_true = temp[true_col].astype(float)
    y_pred = temp[pred_col].astype(float)

    exact_match = float((y_true == y_pred).mean())

    mae = float(mean_absolute_error(y_true, y_pred))

    within_one = float((np.abs(y_true - y_pred) <= 1.0).mean())

    # For sklearn classification metrics, convert to string labels.
    y_true_str = y_true.astype(str)
    y_pred_str = y_pred.astype(str)

    accuracy = float(accuracy_score(y_true_str, y_pred_str))

    precision_macro = float(
        precision_score(
            y_true_str,
            y_pred_str,
            average="macro",
            zero_division=0
        )
    )

    recall_macro = float(
        recall_score(
            y_true_str,
            y_pred_str,
            average="macro",
            zero_division=0
        )
    )

    f1_macro = float(
        f1_score(
            y_true_str,
            y_pred_str,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted = float(
        precision_score(
            y_true_str,
            y_pred_str,
            average="weighted",
            zero_division=0
        )
    )

    recall_weighted = float(
        recall_score(
            y_true_str,
            y_pred_str,
            average="weighted",
            zero_division=0
        )
    )

    f1_weighted = float(
        f1_score(
            y_true_str,
            y_pred_str,
            average="weighted",
            zero_division=0
        )
    )

    try:
        kappa = float(cohen_kappa_score(y_true_str, y_pred_str))
    except Exception:
        kappa = None

    try:
        quadratic_kappa = float(
            cohen_kappa_score(
                y_true_str,
                y_pred_str,
                weights="quadratic"
            )
        )
    except Exception:
        quadratic_kappa = None

    report = classification_report(
        y_true_str,
        y_pred_str,
        zero_division=0,
        output_dict=True
    )

    labels_as_str = [str(float(x)) for x in valid_labels]

    cm = confusion_matrix(
        y_true_str,
        y_pred_str,
        labels=labels_as_str
    )

    if label_names is None:
        label_names = labels_as_str

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in label_names],
        columns=[f"pred_{name}" for name in label_names]
    )

    metrics = {
        "task": task_name,
        "sample_count": int(len(temp)),
        "accuracy_exact_match": exact_match,
        "mae_ordinal_error": mae,
        "within_one_score_accuracy": within_one,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "cohen_kappa": kappa,
        "quadratic_weighted_kappa": quadratic_kappa,
        "classification_report": report
    }

    return metrics, cm_df


# ============================================================
# Feature behaviour analysis
# ============================================================

def compute_group_statistics(df):
    """
    Checks whether extracted features behave correctly for dentist labels.

    Example:
        Irregular localized widening should have high:
            asymmetry_ratio
            localized_asymmetry_percent
            irregularity_index
    """

    if "dentist_pdl_pattern_score" not in df.columns:
        print("Cannot compute group statistics: dentist_pdl_pattern_score missing.")
        return None

    important_features = [
        "mean_left_width_px",
        "mean_right_width_px",
        "mean_total_width_px",
        "max_total_width_px",
        "p90_total_width_px",
        "p95_total_width_px",
        "width_std_px",
        "width_cv",
        "mean_asymmetry_px",
        "max_asymmetry_px",
        "asymmetry_ratio",
        "side_difference_px",
        "side_dominance_ratio",
        "side_consistency",
        "localized_widening_percent",
        "localized_asymmetry_percent",
        "max_to_mean_width_ratio",
        "p95_to_median_width_ratio",
        "coronal_mean_width_px",
        "middle_mean_width_px",
        "apical_mean_width_px",
        "apical_to_coronal_width_ratio",
        "left_continuity_percent",
        "right_continuity_percent",
        "total_continuity_percent",
        "total_longest_run_ratio",
        "irregularity_index",
        "pdl_root_area_ratio"
    ]

    available_features = [
        col for col in important_features
        if col in df.columns
    ]

    if len(available_features) == 0:
        print("No expected feature columns found for group statistics.")
        return None

    temp = df.copy()
    temp["dentist_pdl_pattern_score"] = clean_numeric_column(
        temp,
        "dentist_pdl_pattern_score"
    )

    temp = temp.dropna(subset=["dentist_pdl_pattern_score"])

    rows = []

    for pattern_score in sorted(temp["dentist_pdl_pattern_score"].unique()):
        group = temp[temp["dentist_pdl_pattern_score"] == pattern_score]

        if pattern_score == 1:
            pattern_label = "UNIFORM_WIDENING"
        elif pattern_score == 2:
            pattern_label = "SIDE_DOMINANT_WIDENING"
        elif pattern_score == 3:
            pattern_label = "IRREGULAR_LOCALIZED_WIDENING"
        else:
            pattern_label = "UNKNOWN"

        row = {
            "dentist_pdl_pattern_score": pattern_score,
            "dentist_pdl_pattern_label": pattern_label,
            "sample_count": len(group)
        }

        for feature in available_features:
            values = pd.to_numeric(group[feature], errors="coerce")
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_std"] = float(values.std())
            row[f"{feature}_median"] = float(values.median())
            row[f"{feature}_min"] = float(values.min())
            row[f"{feature}_max"] = float(values.max())

        rows.append(row)

    group_stats = pd.DataFrame(rows)

    return group_stats


def compute_feature_correlations(df):
    """
    Computes Spearman correlation between numeric extracted features
    and dentist PDL pattern score.

    High positive correlation means the feature increases as the pattern
    moves from:
        1 Uniform
        2 Side-dominant
        3 Irregular localized
    """

    if "dentist_pdl_pattern_score" not in df.columns:
        print("Cannot compute correlations: dentist_pdl_pattern_score missing.")
        return None

    temp = df.copy()

    temp["dentist_pdl_pattern_score"] = clean_numeric_column(
        temp,
        "dentist_pdl_pattern_score"
    )

    numeric_df = temp.select_dtypes(include=[np.number]).copy()

    exclude_keywords = [
        "dentist",
        "rule",
        "score",
        "image_width",
        "image_height",
        "upscale_factor"
    ]

    feature_columns = []

    for col in numeric_df.columns:
        lower = col.lower()

        if col == "dentist_pdl_pattern_score":
            continue

        if any(keyword in lower for keyword in exclude_keywords):
            continue

        feature_columns.append(col)

    rows = []

    for feature in feature_columns:
        valid = temp[[feature, "dentist_pdl_pattern_score"]].copy()
        valid[feature] = pd.to_numeric(valid[feature], errors="coerce")
        valid["dentist_pdl_pattern_score"] = pd.to_numeric(
            valid["dentist_pdl_pattern_score"],
            errors="coerce"
        )

        valid = valid.dropna()

        if len(valid) < 3:
            continue

        spearman_corr = valid[feature].corr(
            valid["dentist_pdl_pattern_score"],
            method="spearman"
        )

        pearson_corr = valid[feature].corr(
            valid["dentist_pdl_pattern_score"],
            method="pearson"
        )

        rows.append({
            "feature": feature,
            "spearman_correlation_with_pdl_pattern": float(spearman_corr),
            "pearson_correlation_with_pdl_pattern": float(pearson_corr),
            "absolute_spearman": float(abs(spearman_corr)),
            "sample_count": int(len(valid))
        })

    corr_df = pd.DataFrame(rows)

    if len(corr_df) > 0:
        corr_df = corr_df.sort_values(
            by="absolute_spearman",
            ascending=False
        )

    return corr_df


def extract_misclassified_cases(df):
    """
    Lists cases where rule-based PDL pattern does not match dentist label.
    """

    required_cols = [
        "image_name",
        "dentist_pdl_pattern_score",
        "pdl_pattern_score_rule"
    ]

    for col in required_cols:
        if col not in df.columns:
            print("Cannot extract misclassified cases. Missing:", col)
            return None

    temp = df.copy()

    temp["dentist_pdl_pattern_score"] = clean_numeric_column(
        temp,
        "dentist_pdl_pattern_score"
    )

    temp["pdl_pattern_score_rule"] = clean_numeric_column(
        temp,
        "pdl_pattern_score_rule"
    )

    temp = temp.dropna(
        subset=[
            "dentist_pdl_pattern_score",
            "pdl_pattern_score_rule"
        ]
    )

    misclassified = temp[
        temp["dentist_pdl_pattern_score"].astype(float)
        != temp["pdl_pattern_score_rule"].astype(float)
    ].copy()

    useful_cols = [
        "image_name",
        "dentist_pdl_pattern_score",
        "pdl_pattern_score_rule",
        "pdl_pattern_label_rule",
        "mean_left_width_px",
        "mean_right_width_px",
        "mean_total_width_px",
        "asymmetry_ratio",
        "side_dominance_ratio",
        "side_consistency",
        "localized_asymmetry_percent",
        "localized_widening_percent",
        "width_cv",
        "irregularity_index",
        "fracture_probability_score_rule",
        "dentist_fracture_probability_score",
        "rct_success_score_rule",
        "dentist_rct_success_score"
    ]

    useful_cols = [
        col for col in useful_cols
        if col in misclassified.columns
    ]

    misclassified = misclassified[useful_cols]

    return misclassified


# ============================================================
# Main evaluation
# ============================================================

def run():
    print("Starting PDL feature extraction evaluation...")

    ensure_output_dir()

    df = load_or_create_merged_dataset()

    print("Total merged samples:", len(df))

    if "image_name" in df.columns:
        df["image_name"] = df["image_name"].apply(normalize_file_name)

    all_metrics = {}

    summary_rows = []

    # ------------------------------------------------------------
    # 1. Evaluate PDL pattern classification
    # ------------------------------------------------------------

    pdl_metrics, pdl_cm = evaluate_discrete_task(
        df=df,
        true_col="dentist_pdl_pattern_score",
        pred_col="pdl_pattern_score_rule",
        task_name="PDL Pattern Classification",
        valid_labels=[1, 2, 3],
        label_names=[
            "Uniform",
            "SideDominant",
            "IrregularLocalized"
        ]
    )

    if pdl_metrics is not None:
        all_metrics["pdl_pattern"] = pdl_metrics

        summary_rows.append({
            "task": "PDL Pattern Classification",
            "sample_count": pdl_metrics["sample_count"],
            "accuracy_exact_match": pdl_metrics["accuracy_exact_match"],
            "mae_ordinal_error": pdl_metrics["mae_ordinal_error"],
            "within_one_score_accuracy": pdl_metrics["within_one_score_accuracy"],
            "precision_macro": pdl_metrics["precision_macro"],
            "recall_macro": pdl_metrics["recall_macro"],
            "f1_macro": pdl_metrics["f1_macro"],
            "cohen_kappa": pdl_metrics["cohen_kappa"],
            "quadratic_weighted_kappa": pdl_metrics["quadratic_weighted_kappa"]
        })

        pdl_cm.to_csv(CONFUSION_MATRIX_CSV_PATH)
        print("Saved PDL pattern confusion matrix:", CONFUSION_MATRIX_CSV_PATH)

    # ------------------------------------------------------------
    # 2. Evaluate fracture probability rule score
    # ------------------------------------------------------------

    fracture_metrics, fracture_cm = evaluate_discrete_task(
        df=df,
        true_col="dentist_fracture_probability_score",
        pred_col="fracture_probability_score_rule",
        task_name="Fracture Probability Score",
        valid_labels=[0, 1, 2, 3],
        label_names=[
            "VeryLow",
            "Low",
            "Moderate",
            "High"
        ]
    )

    if fracture_metrics is not None:
        all_metrics["fracture_probability"] = fracture_metrics

        summary_rows.append({
            "task": "Fracture Probability Score",
            "sample_count": fracture_metrics["sample_count"],
            "accuracy_exact_match": fracture_metrics["accuracy_exact_match"],
            "mae_ordinal_error": fracture_metrics["mae_ordinal_error"],
            "within_one_score_accuracy": fracture_metrics["within_one_score_accuracy"],
            "precision_macro": fracture_metrics["precision_macro"],
            "recall_macro": fracture_metrics["recall_macro"],
            "f1_macro": fracture_metrics["f1_macro"],
            "cohen_kappa": fracture_metrics["cohen_kappa"],
            "quadratic_weighted_kappa": fracture_metrics["quadratic_weighted_kappa"]
        })

        fracture_cm_path = os.path.join(
            OUTPUT_DIR,
            "fracture_probability_confusion_matrix.csv"
        )

        fracture_cm.to_csv(fracture_cm_path)
        print("Saved fracture probability confusion matrix:", fracture_cm_path)

    # ------------------------------------------------------------
    # 3. Evaluate RCT success rule score
    # ------------------------------------------------------------

    rct_metrics, rct_cm = evaluate_discrete_task(
        df=df,
        true_col="dentist_rct_success_score",
        pred_col="rct_success_score_rule",
        task_name="RCT Success Score",
        valid_labels=[0.0, 0.5, 1.0],
        label_names=[
            "Low",
            "Moderate",
            "High"
        ]
    )

    if rct_metrics is not None:
        all_metrics["rct_success"] = rct_metrics

        summary_rows.append({
            "task": "RCT Success Score",
            "sample_count": rct_metrics["sample_count"],
            "accuracy_exact_match": rct_metrics["accuracy_exact_match"],
            "mae_ordinal_error": rct_metrics["mae_ordinal_error"],
            "within_one_score_accuracy": rct_metrics["within_one_score_accuracy"],
            "precision_macro": rct_metrics["precision_macro"],
            "recall_macro": rct_metrics["recall_macro"],
            "f1_macro": rct_metrics["f1_macro"],
            "cohen_kappa": rct_metrics["cohen_kappa"],
            "quadratic_weighted_kappa": rct_metrics["quadratic_weighted_kappa"]
        })

        rct_cm_path = os.path.join(
            OUTPUT_DIR,
            "rct_success_confusion_matrix.csv"
        )

        rct_cm.to_csv(rct_cm_path)
        print("Saved RCT success confusion matrix:", rct_cm_path)

    # ------------------------------------------------------------
    # 4. Group statistics
    # ------------------------------------------------------------

    group_stats = compute_group_statistics(df)

    if group_stats is not None:
        group_stats.to_csv(GROUP_STATS_CSV_PATH, index=False)
        print("Saved group statistics:", GROUP_STATS_CSV_PATH)

    # ------------------------------------------------------------
    # 5. Feature correlations
    # ------------------------------------------------------------

    corr_df = compute_feature_correlations(df)

    if corr_df is not None:
        corr_df.to_csv(FEATURE_CORRELATION_CSV_PATH, index=False)
        print("Saved feature correlations:", FEATURE_CORRELATION_CSV_PATH)

    # ------------------------------------------------------------
    # 6. Misclassified cases
    # ------------------------------------------------------------

    misclassified = extract_misclassified_cases(df)

    if misclassified is not None:
        misclassified.to_csv(MISCLASSIFIED_CSV_PATH, index=False)
        print("Saved misclassified cases:", MISCLASSIFIED_CSV_PATH)

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print("\nEvaluation summary:")
    print(summary_df)

    print("\nSaved metrics:", METRICS_JSON_PATH)
    print("Saved summary:", SUMMARY_CSV_PATH)

    if corr_df is not None and len(corr_df) > 0:
        print("\nTop feature correlations with dentist PDL pattern:")
        print(corr_df.head(10))

    if misclassified is not None:
        print("\nMisclassified PDL pattern cases:", len(misclassified))


if __name__ == "__main__":
    run()