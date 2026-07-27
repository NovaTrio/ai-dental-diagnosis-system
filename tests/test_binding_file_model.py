from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.canalwidth.models.train_binding_file_model import (
    nearest_iso_size,
    train_binding_file_model,
)


class BindingFileModelTests(unittest.TestCase):
    def test_nearest_iso_examples(self) -> None:
        np.testing.assert_array_equal(
            nearest_iso_size([18, 22, 27]),
            [20, 20, 25],
        )

    def test_two_feature_regression(self) -> None:
        wl = np.asarray([18, 19, 20, 21, 22, 23], dtype=float)
        width = np.asarray([2, 3, 4, 5, 6, 7], dtype=float)
        target = 2 * wl + 3 * width
        dataset = pd.DataFrame(
            {
                "image_name": [f"case_{i}.png" for i in range(len(wl))],
                "root_id": ["R1"] * len(wl),
                "predicted_wl_mm": wl,
                "apical_diameter_px": width,
                "first_binding_file_size_k": target,
            }
        )
        model, _, metrics = train_binding_file_model(dataset)
        np.testing.assert_allclose(
            model.predict(
                dataset[["predicted_wl_mm", "apical_diameter_px"]].to_numpy()
            ),
            target,
        )
        self.assertLess(float(metrics["continuous_mae_k"]), 1e-10)


if __name__ == "__main__":
    unittest.main()
