import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)


warnings.filterwarnings("ignore")


# ============================================================
# Paths
# ============================================================

INPUT_CSV_PATH = (
    "data/fractures/processed/"
    "pdl_pattern_features_v2_with_dentist_labels.csv"
)

OUTPUT_DIR = "data/fractures/processed/pdl_pattern_tabular_model_results"

METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "tabular_model_metrics.csv")
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "tabular_model_metrics.json")
BEST_MODEL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "best_model_summary.json")

RULE_BASELINE = {
    "name": "V2 calibrated rule-based classifier",
    "accuracy": 0.6765,
    "macro_f1": 0.589,
    "misclassified_count": 11,
    "total_count": 34,
}


# ============================================================
# Label names
# ============================================================

PDL_LABEL_NAMES = {
    1: "Uniform widening",
    2: "Side-dominant widening",
    3: "Irregular localized widening",
}

CLASS_LABELS = [1, 2, 3]


# ============================================================
# Utility functions
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_target_column(df):
    """
    Finds the dentist PDL pattern label column.

    Current expected column:
        pdl_pattern_score

    Older possible names are included for safety.
    """

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
        "Could not find target label column. Expected one of: "
        + ", ".join(candidates)
    )


def is_prediction_column(col_name):
    """
    Removes any previous predicted-label columns so the model does not cheat.
    """

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


def get_feature_columns(df, target_col):
    """
    Select numeric extracted feature columns only.

    Removes:
    - image identity columns
    - dentist target columns
    - fracture/RCT target columns
    - merge/helper/error columns
    - any previous prediction columns
    - non-numeric columns
    """

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
        raise ValueError("No valid numeric feature columns found.")

    return feature_cols


def clean_dataset(df, target_col, feature_cols):
    """
    Keeps only rows with valid target labels and selected features.
    """

    df = df.copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df[df[target_col].notna()]
    df = df[df[target_col].isin(CLASS_LABELS)]

    y = df[target_col].astype(int).values
    X = df[feature_cols].copy()

    image_names = (
        df["image_name"].astype(str).values
        if "image_name" in df.columns
        else np.array([f"sample_{i}" for i in range(len(df))])
    )

    return X, y, image_names, df


def ordinal_mae(y_true, y_pred):
    """
    Ordinal mean absolute error.

    Since labels are ordered:
        1 = uniform
        2 = side-dominant
        3 = irregular

    Predicting 1 instead of 3 is worse than predicting 2 instead of 3.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))


def build_models():
    """
    Classical supervised tabular models.

    These are newly trained on your own extracted features.
    No pretrained model is used.
    """

    models = {}

    models["Decision Tree"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=3,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    models["SVM RBF"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    models["Logistic Regression"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="auto",
                    random_state=42,
                ),
            ),
        ]
    )

    models["KNN"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=3,
                    weights="distance",
                    metric="minkowski",
                ),
            ),
        ]
    )

    models["Naive Bayes"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]
    )

    return models


def leave_one_out_predict(model, X, y):
    """
    Leave-One-Out Cross Validation.

    For 34 images:
        train on 33
        test on 1
        repeat 34 times
    """

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    test_indices = []

    X_values = X.values

    for train_idx, test_idx in loo.split(X_values):
        X_train = X_values[train_idx]
        X_test = X_values[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)

        y_true_all.append(int(y_test[0]))
        y_pred_all.append(int(y_pred[0]))
        test_indices.append(int(test_idx[0]))

    return np.array(y_true_all), np.array(y_pred_all), np.array(test_indices)


def compute_metrics(y_true, y_pred):
    """
    Computes classification metrics.
    """

    acc = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )

    macro_recall = recall_score(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )

    macro_f1 = f1_score(
        y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )

    weighted_f1 = f1_score(
        y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0
    )

    kappa = cohen_kappa_score(y_true, y_pred, labels=CLASS_LABELS)

    quadratic_kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        weights="quadratic",
    )

    mae = ordinal_mae(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    misclassified_count = int(np.sum(y_true != y_pred))

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(kappa),
        "quadratic_weighted_kappa": float(quadratic_kappa),
        "ordinal_mae": float(mae),
        "misclassified_count": misclassified_count,
        "total_count": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
    }


def save_predictions(
    model_name,
    y_true,
    y_pred,
    test_indices,
    image_names,
):
    """
    Saves all LOOCV predictions and misclassified cases.
    """

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    pred_rows = []

    for i, idx in enumerate(test_indices):
        actual = int(y_true[i])
        predicted = int(y_pred[i])

        pred_rows.append(
            {
                "loo_test_index": int(idx),
                "image_name": image_names[idx],
                "actual_label": actual,
                "actual_label_name": PDL_LABEL_NAMES.get(actual, "Unknown"),
                "predicted_label": predicted,
                "predicted_label_name": PDL_LABEL_NAMES.get(predicted, "Unknown"),
                "is_correct": actual == predicted,
                "absolute_label_error": abs(actual - predicted),
            }
        )

    pred_df = pd.DataFrame(pred_rows)

    predictions_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_loocv_predictions.csv",
    )

    misclassified_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_misclassified_cases.csv",
    )

    pred_df.to_csv(predictions_path, index=False)

    misclassified_df = pred_df[pred_df["is_correct"] == False].copy()
    misclassified_df.to_csv(misclassified_path, index=False)

    return predictions_path, misclassified_path


def save_confusion_matrix(model_name, cm):
    """
    Saves confusion matrix as CSV.
    """

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "actual_uniform_1",
            "actual_side_dominant_2",
            "actual_irregular_3",
        ],
        columns=[
            "pred_uniform_1",
            "pred_side_dominant_2",
            "pred_irregular_3",
        ],
    )

    path = os.path.join(OUTPUT_DIR, f"{safe_name}_confusion_matrix.csv")
    cm_df.to_csv(path)

    return path


def print_model_report(model_name, metrics, y_true, y_pred):
    """
    Prints a readable report for one model.
    """

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print(f"Accuracy:                  {metrics['accuracy']:.4f}")
    print(f"Macro precision:           {metrics['macro_precision']:.4f}")
    print(f"Macro recall:              {metrics['macro_recall']:.4f}")
    print(f"Macro-F1:                  {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:               {metrics['weighted_f1']:.4f}")
    print(f"Cohen kappa:               {metrics['cohen_kappa']:.4f}")
    print(f"Quadratic weighted kappa:  {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"Ordinal MAE:               {metrics['ordinal_mae']:.4f}")
    print(
        f"Misclassified:             "
        f"{metrics['misclassified_count']} / {metrics['total_count']}"
    )

    print("\nConfusion matrix:")
    print(
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=[
                "actual_uniform_1",
                "actual_side_dominant_2",
                "actual_irregular_3",
            ],
            columns=[
                "pred_uniform_1",
                "pred_side_dominant_2",
                "pred_irregular_3",
            ],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=[
                "Uniform widening",
                "Side-dominant widening",
                "Irregular localized widening",
            ],
            zero_division=0,
        )
    )


def compare_with_rule_baseline(best_metrics):
    """
    Compares best supervised model with current V2 calibrated rule baseline.
    """

    supervised_misclassified = best_metrics["misclassified_count"]
    rule_misclassified = RULE_BASELINE["misclassified_count"]

    if supervised_misclassified < rule_misclassified:
        decision = "Use supervised tabular model as the final classifier."
    elif supervised_misclassified == rule_misclassified:
        decision = (
            "Supervised model ties with the V2 calibrated rule. "
            "Prefer the rule-based method if explainability is more important, "
            "or use the supervised model if it has better macro-F1/kappa."
        )
    else:
        decision = (
            "Keep the V2 calibrated rule-based classifier as the final "
            "explainable method."
        )

    return decision


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("\nStarting supervised PDL pattern tabular-model training...")
    print("Input CSV:", INPUT_CSV_PATH)
    print("Output directory:", OUTPUT_DIR)

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

    print("\nLoaded dataset:")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    target_col = find_target_column(df)
    print("\nTarget column:", target_col)

    feature_cols = get_feature_columns(df, target_col)

    print("\nSelected numeric feature columns:")
    for col in feature_cols:
        print(" -", col)

    X, y, image_names, cleaned_df = clean_dataset(df, target_col, feature_cols)

    print("\nCleaned dataset:")
    print("Samples:", len(y))
    print("Features:", len(feature_cols))

    print("\nClass distribution:")
    class_counts = pd.Series(y).value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"{label} ({PDL_LABEL_NAMES[label]}): {count}")

    if len(y) < 3:
        raise ValueError("Not enough valid samples for Leave-One-Out CV.")

    models = build_models()

    all_metrics = []
    all_metrics_json = {}

    best_model_name = None
    best_metrics = None

    for model_name, model in models.items():
        print("\nRunning Leave-One-Out CV for:", model_name)

        y_true, y_pred, test_indices = leave_one_out_predict(model, X, y)

        metrics = compute_metrics(y_true, y_pred)

        predictions_path, misclassified_path = save_predictions(
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            test_indices=test_indices,
            image_names=image_names,
        )

        cm_path = save_confusion_matrix(
            model_name=model_name,
            cm=np.array(metrics["confusion_matrix"]),
        )

        metrics["model_name"] = model_name
        metrics["predictions_csv"] = predictions_path
        metrics["misclassified_csv"] = misclassified_path
        metrics["confusion_matrix_csv"] = cm_path

        all_metrics.append(metrics)
        all_metrics_json[model_name] = metrics

        print_model_report(model_name, metrics, y_true, y_pred)

        if best_metrics is None:
            best_model_name = model_name
            best_metrics = metrics
        else:
            current_key = (
                metrics["misclassified_count"],
                -metrics["macro_f1"],
                -metrics["quadratic_weighted_kappa"],
            )

            best_key = (
                best_metrics["misclassified_count"],
                -best_metrics["macro_f1"],
                -best_metrics["quadratic_weighted_kappa"],
            )

            if current_key < best_key:
                best_model_name = model_name
                best_metrics = metrics

    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics)

    ordered_cols = [
        "model_name",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
        "quadratic_weighted_kappa",
        "ordinal_mae",
        "misclassified_count",
        "total_count",
        "predictions_csv",
        "misclassified_csv",
        "confusion_matrix_csv",
    ]

    metrics_df = metrics_df[ordered_cols]
    metrics_df = metrics_df.sort_values(
        by=[
            "misclassified_count",
            "macro_f1",
            "quadratic_weighted_kappa",
        ],
        ascending=[True, False, False],
    )

    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics_json, f, indent=4)

    final_decision = compare_with_rule_baseline(best_metrics)

    best_summary = {
        "best_supervised_model": best_model_name,
        "best_supervised_metrics": best_metrics,
        "rule_baseline": RULE_BASELINE,
        "final_decision": final_decision,
        "feature_columns_used": feature_cols,
        "target_column": target_col,
        "sample_count": int(len(y)),
        "feature_count": int(len(feature_cols)),
    }

    with open(BEST_MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=4)

    print("\n" + "=" * 70)
    print("FINAL SUPERVISED MODEL COMPARISON")
    print("=" * 70)

    print("\nCurrent best rule-based baseline:")
    print(
        f"Misclassified: {RULE_BASELINE['misclassified_count']} / "
        f"{RULE_BASELINE['total_count']}"
    )
    print(f"Accuracy:      {RULE_BASELINE['accuracy']:.4f}")
    print(f"Macro-F1:      {RULE_BASELINE['macro_f1']:.4f}")

    print("\nBest supervised tabular model:")
    print("Model:", best_model_name)
    print(
        f"Misclassified: {best_metrics['misclassified_count']} / "
        f"{best_metrics['total_count']}"
    )
    print(f"Accuracy:      {best_metrics['accuracy']:.4f}")
    print(f"Macro-F1:      {best_metrics['macro_f1']:.4f}")
    print(f"Kappa:         {best_metrics['cohen_kappa']:.4f}")
    print(
        f"Quadratic weighted kappa: "
        f"{best_metrics['quadratic_weighted_kappa']:.4f}"
    )
    print(f"Ordinal MAE:   {best_metrics['ordinal_mae']:.4f}")

    print("\nDecision:")
    print(final_decision)

    print("\nSaved outputs:")
    print("Metrics CSV:", METRICS_CSV_PATH)
    print("Metrics JSON:", METRICS_JSON_PATH)
    print("Best model summary:", BEST_MODEL_SUMMARY_PATH)


if __name__ == "__main__":
    main()