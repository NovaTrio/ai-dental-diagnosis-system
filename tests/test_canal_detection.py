from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.canalwidth.segmentation.canal_detection import (
    create_tooth_interior_mask,
    detect_canal,
    select_central_canal_component,
)


class CanalDetectionTests(unittest.TestCase):
    def test_black_background_is_excluded(self) -> None:
        image = np.zeros((80, 50), dtype=np.uint8)
        image[5:75, 10:40] = 180
        mask = create_tooth_interior_mask(image)
        self.assertEqual(cv2.countNonZero(mask[:, :9]), 0)
        self.assertGreater(cv2.countNonZero(mask[:, 12:38]), 0)

    def test_component_selection_prefers_central_vertical_candidate(self) -> None:
        tooth = np.zeros((100, 60), dtype=np.uint8)
        tooth[5:95, 8:52] = 255
        candidates = np.zeros_like(tooth)
        candidates[12:90, 28:32] = 255
        candidates[30:60, 10:13] = 255
        selected, reports = select_central_canal_component(candidates, tooth)
        self.assertGreater(cv2.countNonZero(selected[:, 28:32]), 0)
        self.assertEqual(cv2.countNonZero(selected[:, 10:13]), 0)
        self.assertTrue(any(report["selected"] for report in reports))

    def test_complete_pipeline_detects_synthetic_dark_canal(self) -> None:
        image = np.zeros((160, 80), dtype=np.uint8)
        image[10:150, 15:65] = 190
        image[25:140, 37:43] = 45
        result = detect_canal(image, blackhat_kernel_size=15)
        self.assertTrue(result["detected"])
        self.assertGreater(cv2.countNonZero(result["canal_mask"]), 0)


if __name__ == "__main__":
    unittest.main()
