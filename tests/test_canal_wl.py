from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.canalwidth.models.train_wl_model import train_canal_wl_model
from src.canalwidth.training.prepare_wl_dataset import measurement_key


class CanalWorkingLengthTests(unittest.TestCase):
    def test_measurement_key_maps_root_suffix(self) -> None:
        self.assertEqual(measurement_key("case_tooth.png"), ("case", "R1"))
        self.assertEqual(measurement_key("case_tooth_2.png"), ("case", "R2"))

    def test_linear_model_recovers_exact_relationship(self) -> None:
        lengths = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
        dataset = pd.DataFrame(
            {
                "image_name": [f"case_{i}.png" for i in range(5)],
                "root_id": ["R1"] * 5,
                "midline_length_px": lengths,
                "actual_wl_mm": 10.0 + 0.1 * lengths,
            }
        )
        model, predictions, metrics = train_canal_wl_model(dataset)
        self.assertAlmostEqual(float(model.intercept_), 10.0)
        self.assertAlmostEqual(float(model.coef_[0]), 0.1)
        self.assertLess(float(metrics["mae_mm"]), 1e-10)
        self.assertTrue(np.allclose(predictions["predicted_wl_mm"], dataset["actual_wl_mm"]))


if __name__ == "__main__":
    unittest.main()
