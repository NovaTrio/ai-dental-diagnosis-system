from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.canalwidth.features.canal_midline import (
    detect_bright_canal_material,
    extract_canal_midline,
    longest_skeleton_path,
)


class CanalMidlineTests(unittest.TestCase):
    def test_longest_path_removes_short_branch(self) -> None:
        skeleton = np.zeros((40, 30), dtype=np.uint8)
        skeleton[3:37, 15] = 255
        skeleton[18, 15:23] = 255
        path = longest_skeleton_path(skeleton)
        self.assertLessEqual(max(x for x, _ in path), 16)
        self.assertEqual(path[0][1], 3)
        self.assertEqual(path[-1][1], 36)

    def test_bright_vertical_filling_is_detected(self) -> None:
        image = np.zeros((100, 50), dtype=np.uint8)
        image[5:95, 8:42] = 130
        image[15:90, 23:27] = 245
        tooth = (image > 0).astype(np.uint8) * 255
        bright = detect_bright_canal_material(
            image,
            tooth,
            np.zeros_like(image),
        )
        self.assertGreater(cv2.countNonZero(bright[15:90, 23:27]), 0)

    def test_midline_follows_canal_not_tooth_silhouette(self) -> None:
        image = np.zeros((100, 60), dtype=np.uint8)
        image[5:95, 8:52] = 170
        canal = np.zeros_like(image)
        canal[10:90, 18:24] = 255
        result = extract_canal_midline(image, canal, smoothing=0.5)
        xs = [x for x, _ in result["midline_path"]]
        self.assertTrue(result["detected"])
        self.assertLess(float(np.mean(xs)), 27.0)


if __name__ == "__main__":
    unittest.main()
