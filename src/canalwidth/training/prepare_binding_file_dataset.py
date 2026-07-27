"""Prepare leakage-safe features for continuous first-binding-file regression."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.canalwidth.training.prepare_wl_dataset import measurement_key


DEFAULT_WIDTHS = Path("data/canalwidth/widths/canal_widths.csv")
DEFAULT_WL_LOOCV = Path(
    "src/canalwidth/models/working_length/loocv_predictions.csv"
)
DEFAULT_LABELS = Path("data/working_length/labels/wl_labels.csv")


def prepare_binding_file_dataset(
    widths_csv: str | Path = DEFAULT_WIDTHS,
    wl_predictions_csv: str | Path = DEFAULT_WL_LOOCV,
    labels_csv: str | Path = DEFAULT_LABELS,
) -> pd.DataFrame:
    widths = pd.read_csv(widths_csv)
    wl_predictions = pd.read_csv(wl_predictions_csv)
    labels = pd.read_csv(labels_csv)
    for table, required, name in (
        (
            widths,
            {"image_name", "median_canal_width_px", "width_valid"},
            "width",
        ),
        (
            wl_predictions,
            {"image_name", "predicted_wl_mm"},
            "working-length prediction",
        ),
        (
            labels,
            {"image_name", "root_id", "first_binding_file_size_k"},
            "label",
        ),
    ):
        if missing := required - set(table.columns):
            raise ValueError(f"Missing {name} column(s): {', '.join(sorted(missing))}")

    valid_width = widths["width_valid"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    widths = widths.loc[
        valid_width, ["image_name", "median_canal_width_px"]
    ].copy()
    width_keys = widths["image_name"].map(measurement_key)
    widths["image_key"] = width_keys.map(lambda key: key[0])
    widths["root_id"] = width_keys.map(lambda key: key[1])
    widths["apical_diameter_px"] = pd.to_numeric(
        widths.pop("median_canal_width_px"), errors="coerce"
    )

    wl_predictions = wl_predictions[
        ["image_name", "predicted_wl_mm"]
    ].copy()
    prediction_keys = wl_predictions["image_name"].map(measurement_key)
    wl_predictions["image_key"] = prediction_keys.map(lambda key: key[0])
    wl_predictions["root_id"] = prediction_keys.map(lambda key: key[1])
    wl_predictions["predicted_wl_mm"] = pd.to_numeric(
        wl_predictions["predicted_wl_mm"], errors="coerce"
    )

    labels = labels.copy()
    labels["image_key"] = labels["image_name"].map(
        lambda name: Path(str(name)).stem
    )
    labels["root_id"] = labels["root_id"].fillna("").astype(str).str.strip()
    labels["root_id"] = labels["root_id"].mask(labels["root_id"].eq(""), "R1")
    labels["first_binding_file_size_k"] = pd.to_numeric(
        labels["first_binding_file_size_k"], errors="coerce"
    )
    duplicate_labels = labels.duplicated(["image_key", "root_id"], keep=False)
    if duplicate_labels.any():
        ambiguous = labels.loc[
            duplicate_labels, ["image_key", "root_id"]
        ].drop_duplicates()
        warnings.warn(
            "Excluding ambiguous duplicate binding-file key(s): "
            + ", ".join(
                f"{row.image_key}/{row.root_id}"
                for row in ambiguous.itertuples(index=False)
            ),
            stacklevel=2,
        )
        labels = labels.loc[~duplicate_labels].copy()

    joined = widths.merge(
        wl_predictions[["image_key", "root_id", "predicted_wl_mm"]],
        on=["image_key", "root_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        labels[
            ["image_key", "root_id", "first_binding_file_size_k"]
        ],
        on=["image_key", "root_id"],
        how="inner",
        validate="one_to_one",
    )
    dataset = joined[
        [
            "image_name",
            "root_id",
            "predicted_wl_mm",
            "apical_diameter_px",
            "first_binding_file_size_k",
        ]
    ].copy()
    numeric = dataset[
        ["predicted_wl_mm", "apical_diameter_px", "first_binding_file_size_k"]
    ]
    valid = np.isfinite(numeric).all(axis=1) & numeric.gt(0).all(axis=1)
    dataset = dataset.loc[valid].reset_index(drop=True)
    if dataset.empty:
        raise ValueError("No valid binding-file training samples")
    return dataset
