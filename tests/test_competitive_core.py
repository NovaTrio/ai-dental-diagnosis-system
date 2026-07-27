from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.working_length.segmentation.hybrid_core import (
    competitive_core_reconstruction,
)


class CompetitiveCoreTests(unittest.TestCase):
    def test_separates_two_regions_joined_by_thin_bridge(self) -> None:
        candidate = np.zeros((40, 60), dtype=np.uint8)
        candidate[5:35, 8:25] = 255
        candidate[5:35, 35:52] = 255
        candidate[18:21, 24:36] = 255

        all_cores = np.zeros_like(candidate)
        all_cores[12:28, 13:20] = 255
        all_cores[12:28, 40:47] = 255
        selected_core = np.zeros_like(candidate)
        selected_core[12:28, 13:20] = 255

        territory, reconstruction = competitive_core_reconstruction(
            selected_core,
            all_cores,
            candidate,
        )

        self.assertGreater(cv2.countNonZero(reconstruction[:, 8:25]), 0)
        self.assertEqual(cv2.countNonZero(reconstruction[:, 35:52]), 0)
        self.assertEqual(cv2.countNonZero(territory[:, 50:]), 0)

    def test_without_competitor_preserves_complete_candidate(self) -> None:
        candidate = np.zeros((30, 30), dtype=np.uint8)
        candidate[3:27, 8:22] = 255
        selected_core = np.zeros_like(candidate)
        selected_core[10:20, 12:18] = 255

        territory, reconstruction = competitive_core_reconstruction(
            selected_core,
            selected_core,
            candidate,
        )

        np.testing.assert_array_equal(territory, candidate)
        np.testing.assert_array_equal(reconstruction, candidate)


if __name__ == "__main__":
    unittest.main()
