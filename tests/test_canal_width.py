from __future__ import annotations

import unittest

import numpy as np

from src.canalwidth.features.canal_width import (
    measure_apical_canal_widths,
    point_and_tangent_at_fraction,
)


class CanalWidthTests(unittest.TestCase):
    def test_arc_fraction_and_tangent_for_vertical_path(self) -> None:
        path = [(25.0, float(y)) for y in range(10, 91)]
        point, tangent = point_and_tangent_at_fraction(path, 0.95)
        np.testing.assert_allclose(point, [25.0, 86.0], atol=1e-6)
        np.testing.assert_allclose(tangent, [0.0, 1.0], atol=1e-6)

    def test_vertical_canal_has_expected_perpendicular_width(self) -> None:
        mask = np.zeros((120, 60), dtype=np.uint8)
        mask[5:110, 20:30] = 255
        path = [(24.5, float(y)) for y in range(5, 110)]
        result = measure_apical_canal_widths(mask, path)
        self.assertTrue(result["valid"])
        self.assertEqual(result["valid_measurement_count"], 5)
        self.assertAlmostEqual(float(result["median_width_px"]), 10.0, delta=0.5)

    def test_median_reduces_single_wide_outlier(self) -> None:
        mask = np.zeros((120, 80), dtype=np.uint8)
        mask[5:115, 30:40] = 255
        # Widen only the 97% neighbourhood.
        mask[110:112, 20:50] = 255
        path = [(34.5, float(y)) for y in range(5, 115)]
        result = measure_apical_canal_widths(mask, path)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(float(result["median_width_px"]), 10.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
