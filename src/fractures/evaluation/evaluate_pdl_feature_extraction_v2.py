import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# Paths
# ============================================================

INPUT_CSV_PATH = (
    "data/fractures/processed/"
    "pdl_pattern_features_v3_with_dentist_labels.csv"
)

OUTPUT_DIR = (
    "data/fractures/processed/"
    "pdl_feature_extraction_v3_evaluation"
)

FEATURE_SANITY_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_sanity_report.csv"
)

FEATURE_SUMMARY_BY_CLASS_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_summary_by_dentist_class.csv"
)

FEATURE_SEPARABILITY_RANKING_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_separability_ranking.csv"
)

CLINICAL_FEATURE_CHECK_PATH = os.path.join(
    OUTPUT_DIR,
    "clinical_feature_direction_check.csv"
)

PCA_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "pca_feature_space_by_dentist_label.png"
)

SUMMARY_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_extraction_evaluation_summary.json"
)


# ============================================================
# Labels
# ============================================================

CLASS_LABELS = [1, 2, 3]

PDL_LABEL_NAMES = {
    1: "Uniform widening",
    2: "Side-dominant widening",
    3: "Irregular localized widening",
}


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_target_column(df):
    candidates = [
        "pdl_pattern_score",
        "dentist_pdl_pattern_score",
        "dentist_pattern_score",
        "pattern_score",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find dentist PDL pattern label column. "
        "Expected one of: " + ", ".join(candidates)
    )


def is_prediction_column(col_name):
    name = col_name.lower()

    prediction_keywords = [
        "pred",
        "prediction",
        "predicted",
        "rule",
        "classified",
        "classification",
        "model_output",
        "output_label",
    ]

    return any(keyword in name for keyword in prediction_keywords)


def get_numeric_feature_columns(df, target_col):
    non_feature_columns = {
        "image_name",
        "image_name_dentist",
        "_merge_stem",
        "feature_error",
        target_col,
        "pdl_pattern_score",
        "dentist_pdl_pattern_score",
        "fracture_probability_score",
        "rct_success_score",
        "dentist_fracture_probability_score",
        "dentist_rct_success_score",
    }

    feature_cols = []

    for col in df.columns:
        if col in non_feature_columns:
            continue

        if is_prediction_column(col):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    return feature_cols


def clean_dataset(df, target_col, feature_cols):
    df = df.copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df[df[target_col].notna()]
    df = df[df[target_col].isin(CLASS_LABELS)]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# Sanity check
# ============================================================

def create_feature_sanity_report(df, feature_cols):
    rows = []

    for col in feature_cols:
        values = df[col]

        missing_count = int(values.isna().sum())
        missing_percent = float(missing_count / len(df) * 100)

        valid_values = values.dropna()

        if len(valid_values) > 0:
            min_value = float(valid_values.min())
            max_value = float(valid_values.max())
            mean_value = float(valid_values.mean())
            std_value = float(valid_values.std())
            unique_count = int(valid_values.nunique())
            zero_variance = bool(valid_values.nunique() <= 1)
        else:
            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan
            std_value = np.nan
            unique_count = 0
            zero_variance = True

        rows.append(
            {
                "feature": col,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "unique_count": unique_count,
                "zero_variance": zero_variance,
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
                "std": std_value,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Feature summary by dentist class
# ============================================================

def create_feature_summary_by_class(df, target_col, feature_cols):
    rows = []

    for feature in feature_cols:
        for label in CLASS_LABELS:
            class_df = df[df[target_col] == label]
            values = class_df[feature].dropna()

            if len(values) == 0:
                row = {
                    "feature": feature,
                    "dentist_label": label,
                    "dentist_label_name": PDL_LABEL_NAMES[label],
                    "count": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "max": np.nan,
                }
            else:
                row = {
                    "feature": feature,
                    "dentist_label": label,
                    "dentist_label_name": PDL_LABEL_NAMES[label],
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Feature separability ranking
# ============================================================

def safe_kruskal_wallis(groups):
    """
    Kruskal-Wallis test without requiring scipy at import time.

    Returns:
        H statistic, p-value

    If scipy is unavailable or test fails, returns NaN.
    """

    try:
        from scipy.stats import kruskal

        clean_groups = [g[~np.isnan(g)] for g in groups]
        clean_groups = [g for g in clean_groups if len(g) > 0]

        if len(clean_groups) < 2:
            return np.nan, np.nan

        h, p = kruskal(*clean_groups)
        return float(h), float(p)

    except Exception:
        return np.nan, np.nan


def epsilon_squared_from_kruskal(h, n, k):
    """
    Effect size for Kruskal-Wallis.

    Higher value means stronger class separation.
    """

    if np.isnan(h):
        return np.nan

    if n <= k:
        return np.nan

    eps = (h - k + 1) / (n - k)
    eps = max(0.0, min(1.0, eps))

    return float(eps)


def create_feature_separability_ranking(df, target_col, feature_cols):
    y = df[target_col].astype(int).values

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation for mutual information only
    X_imputed = X.copy()
    for col in feature_cols:
        median_value = X_imputed[col].median()
        X_imputed[col] = X_imputed[col].fillna(median_value)

    try:
        mi_scores = mutual_info_classif(
            X_imputed.values,
            y,
            discrete_features=False,
            random_state=42,
        )
    except Exception:
        mi_scores = np.zeros(len(feature_cols))

    rows = []

    for i, feature in enumerate(feature_cols):
        groups = []

        means = {}

        for label in CLASS_LABELS:
            values = df[df[target_col] == label][feature].astype(float).values
            groups.append(values)

            values_no_nan = values[~np.isnan(values)]

            if len(values_no_nan) > 0:
                means[label] = float(np.mean(values_no_nan))
            else:
                means[label] = np.nan

        h, p = safe_kruskal_wallis(groups)
        eps = epsilon_squared_from_kruskal(
            h=h,
            n=len(df),
            k=len(CLASS_LABELS),
        )

        valid_means = {
            label: value
            for label, value in means.items()
            if not np.isnan(value)
        }

        if valid_means:
            max_mean_class = max(valid_means, key=valid_means.get)
            min_mean_class = min(valid_means, key=valid_means.get)
        else:
            max_mean_class = None
            min_mean_class = None

        rows.append(
            {
                "feature": feature,
                "mutual_information": float(mi_scores[i]),
                "kruskal_h": h,
                "kruskal_p": p,
                "kruskal_epsilon_squared": eps,
                "mean_class_1_uniform": means[1],
                "mean_class_2_side_dominant": means[2],
                "mean_class_3_irregular": means[3],
                "highest_mean_class": max_mean_class,
                "highest_mean_class_name": (
                    PDL_LABEL_NAMES[max_mean_class]
                    if max_mean_class in PDL_LABEL_NAMES
                    else None
                ),
                "lowest_mean_class": min_mean_class,
                "lowest_mean_class_name": (
                    PDL_LABEL_NAMES[min_mean_class]
                    if min_mean_class in PDL_LABEL_NAMES
                    else None
                ),
            }
        )

    ranking_df = pd.DataFrame(rows)

    ranking_df = ranking_df.sort_values(
        by=[
            "kruskal_epsilon_squared",
            "mutual_information",
        ],
        ascending=[False, False],
    )

    return ranking_df


# ============================================================
# Clinical direction check
# ============================================================

def create_clinical_feature_direction_check(summary_df):
    """
    Checks whether key engineered features behave as clinically expected.

    Expected:
        uniformity_score highest for class 1
        side_dominance_score highest for class 2
        localized_spike_score highest for class 3
        localized_peak_count highest for class 3
        total_local_spike_percent highest for class 3
    """

    expected_rules = [
        {
            "feature": "uniformity_score",
            "expected_highest_class": 1,
            "reason": "Uniform cases should have the highest uniformity score.",
        },
        {
            "feature": "side_dominance_score",
            "expected_highest_class": 2,
            "reason": "Side-dominant cases should have the strongest side-dominance score.",
        },
        {
            "feature": "side_dominant_length_ratio",
            "expected_highest_class": 2,
            "reason": "Side-dominant cases should have longer one-sided dominance.",
        },
        {
            "feature": "localized_spike_score",
            "expected_highest_class": 3,
            "reason": "Irregular localized widening should have the highest localized spike score.",
        },
        {
            "feature": "localized_peak_count",
            "expected_highest_class": 3,
            "reason": "Irregular cases should contain more localized width peaks.",
        },
        {
            "feature": "total_local_spike_percent",
            "expected_highest_class": 3,
            "reason": "Irregular cases should have more local spike rows.",
        },
        {
            "feature": "irregularity_index",
            "expected_highest_class": 3,
            "reason": "Irregular cases should have the highest irregularity index.",
        },
    ]

    rows = []

    for rule in expected_rules:
        feature = rule["feature"]

        feature_summary = summary_df[summary_df["feature"] == feature]

        if feature_summary.empty:
            rows.append(
                {
                    "feature": feature,
                    "available": False,
                    "expected_highest_class": rule["expected_highest_class"],
                    "actual_highest_class": None,
                    "direction_correct": False,
                    "class_1_mean": np.nan,
                    "class_2_mean": np.nan,
                    "class_3_mean": np.nan,
                    "reason": rule["reason"],
                }
            )
            continue

        means = {}

        for label in CLASS_LABELS:
            row = feature_summary[feature_summary["dentist_label"] == label]

            if len(row) == 0:
                means[label] = np.nan
            else:
                means[label] = float(row.iloc[0]["mean"])

        valid_means = {
            label: value
            for label, value in means.items()
            if not np.isnan(value)
        }

        if valid_means:
            actual_highest_class = max(valid_means, key=valid_means.get)
        else:
            actual_highest_class = None

        direction_correct = (
            actual_highest_class == rule["expected_highest_class"]
        )

        rows.append(
            {
                "feature": feature,
                "available": True,
                "expected_highest_class": rule["expected_highest_class"],
                "expected_highest_class_name": PDL_LABEL_NAMES[
                    rule["expected_highest_class"]
                ],
                "actual_highest_class": actual_highest_class,
                "actual_highest_class_name": (
                    PDL_LABEL_NAMES[actual_highest_class]
                    if actual_highest_class in PDL_LABEL_NAMES
                    else None
                ),
                "direction_correct": direction_correct,
                "class_1_mean": means[1],
                "class_2_mean": means[2],
                "class_3_mean": means[3],
                "reason": rule["reason"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PCA feature-space plot
# ============================================================

def save_pca_plot(df, target_col, feature_cols, output_path):
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        X[col] = X[col].fillna(X[col].median())

    y = df[target_col].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    for label in CLASS_LABELS:
        mask = y == label
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"{label}: {PDL_LABEL_NAMES[label]}",
            s=60,
            alpha=0.8,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("V2 PDL Feature Space by Dentist Label")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return {
        "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "total_2d_variance_ratio": float(
            pca.explained_variance_ratio_[0]
            + pca.explained_variance_ratio_[1]
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("\nStarting V2 PDL feature extraction evaluation...")
    print("Input CSV:", INPUT_CSV_PATH)
    print("Output directory:", OUTPUT_DIR)

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

    print("\nLoaded dataset:")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    target_col = find_target_column(df)
    print("\nDentist label column:", target_col)

    feature_cols = get_numeric_feature_columns(df, target_col)

    print("\nNumeric feature count:", len(feature_cols))

    df = clean_dataset(df, target_col, feature_cols)

    print("\nCleaned dataset:")
    print("Samples:", len(df))

    print("\nClass distribution:")
    for label in CLASS_LABELS:
        count = int((df[target_col] == label).sum())
        print(f"{label} ({PDL_LABEL_NAMES[label]}): {count}")

    # ------------------------------------------------------------
    # 1. Sanity report
    # ------------------------------------------------------------

    sanity_df = create_feature_sanity_report(df, feature_cols)
    sanity_df.to_csv(FEATURE_SANITY_REPORT_PATH, index=False)

    print("\nSaved feature sanity report:")
    print(FEATURE_SANITY_REPORT_PATH)

    zero_variance_features = sanity_df[sanity_df["zero_variance"] == True][
        "feature"
    ].tolist()

    high_missing_features = sanity_df[
        sanity_df["missing_percent"] > 20
    ]["feature"].tolist()

    # ------------------------------------------------------------
    # 2. Summary by class
    # ------------------------------------------------------------

    summary_df = create_feature_summary_by_class(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    summary_df.to_csv(FEATURE_SUMMARY_BY_CLASS_PATH, index=False)

    print("\nSaved feature summary by dentist class:")
    print(FEATURE_SUMMARY_BY_CLASS_PATH)

    # ------------------------------------------------------------
    # 3. Feature separability ranking
    # ------------------------------------------------------------

    ranking_df = create_feature_separability_ranking(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    ranking_df.to_csv(FEATURE_SEPARABILITY_RANKING_PATH, index=False)

    print("\nSaved feature separability ranking:")
    print(FEATURE_SEPARABILITY_RANKING_PATH)

    # ------------------------------------------------------------
    # 4. Clinical feature direction check
    # ------------------------------------------------------------

    clinical_check_df = create_clinical_feature_direction_check(summary_df)
    clinical_check_df.to_csv(CLINICAL_FEATURE_CHECK_PATH, index=False)

    print("\nSaved clinical feature direction check:")
    print(CLINICAL_FEATURE_CHECK_PATH)

    # ------------------------------------------------------------
    # 5. PCA visualization
    # ------------------------------------------------------------

    pca_info = save_pca_plot(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        output_path=PCA_PLOT_PATH,
    )

    print("\nSaved PCA feature-space plot:")
    print(PCA_PLOT_PATH)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    top_features = ranking_df.head(15)[
        [
            "feature",
            "kruskal_epsilon_squared",
            "mutual_information",
            "kruskal_p",
            "highest_mean_class",
            "highest_mean_class_name",
        ]
    ].to_dict(orient="records")

    clinical_pass_count = int(
        clinical_check_df["direction_correct"].sum()
    )

    clinical_total_count = int(len(clinical_check_df))

    evaluation_summary = {
        "input_csv": INPUT_CSV_PATH,
        "sample_count": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "target_column": target_col,
        "class_distribution": {
            str(label): int((df[target_col] == label).sum())
            for label in CLASS_LABELS
        },
        "zero_variance_features": zero_variance_features,
        "high_missing_features_over_20_percent": high_missing_features,
        "top_15_separating_features": top_features,
        "clinical_direction_checks_passed": clinical_pass_count,
        "clinical_direction_checks_total": clinical_total_count,
        "pca_info": pca_info,
        "output_files": {
            "feature_sanity_report": FEATURE_SANITY_REPORT_PATH,
            "feature_summary_by_class": FEATURE_SUMMARY_BY_CLASS_PATH,
            "feature_separability_ranking": FEATURE_SEPARABILITY_RANKING_PATH,
            "clinical_feature_check": CLINICAL_FEATURE_CHECK_PATH,
            "pca_plot": PCA_PLOT_PATH,
        },
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=4)

    print("\nSaved summary JSON:")
    print(SUMMARY_JSON_PATH)

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION EVALUATION SUMMARY")
    print("=" * 70)

    print("\nSamples:", len(df))
    print("Features:", len(feature_cols))

    print("\nZero-variance features:")
    if zero_variance_features:
        for feature in zero_variance_features:
            print(" -", feature)
    else:
        print("None")

    print("\nHigh-missing features > 20%:")
    if high_missing_features:
        for feature in high_missing_features:
            print(" -", feature)
    else:
        print("None")

    print("\nTop 10 separating features:")
    display_cols = [
        "feature",
        "kruskal_epsilon_squared",
        "mutual_information",
        "kruskal_p",
        "highest_mean_class_name",
    ]

    print(ranking_df[display_cols].head(10).to_string(index=False))

    print("\nClinical direction check:")
    print(
        clinical_check_df[
            [
                "feature",
                "expected_highest_class_name",
                "actual_highest_class_name",
                "direction_correct",
            ]
        ].to_string(index=False)
    )

    print("\nClinical direction checks passed:")
    print(f"{clinical_pass_count} / {clinical_total_count}")

    print("\nInterpretation guide:")
    print(
        "1. Good feature extraction should have few/no missing values.\n"
        "2. Important clinical features should show different means across dentist labels.\n"
        "3. uniformity_score should be highest for class 1.\n"
        "4. side_dominance_score should be highest for class 2.\n"
        "5. localized_spike_score and localized_peak_count should be highest for class 3.\n"
        "6. If these directions are not correct, the feature extraction logic needs improvement."
    )


if __name__ == "__main__":
    main()