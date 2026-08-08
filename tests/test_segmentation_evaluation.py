from __future__ import annotations

import unittest

import numpy as np

from src.working_length.evaluation.evaluate_segmentation import (
    create_agreement_overlay,
)


class SegmentationEvaluationTests(unittest.TestCase):
    def test_agreement_overlay_colors_tp_fp_and_fn(self) -> None:
        prediction = np.asarray([[255, 255], [0, 0]], dtype=np.uint8)
        ground_truth = np.asarray([[255, 0], [255, 0]], dtype=np.uint8)

        overlay = create_agreement_overlay(prediction, ground_truth)

        np.testing.assert_array_equal(overlay[0, 0], [0, 255, 0])
        np.testing.assert_array_equal(overlay[0, 1], [0, 0, 255])
        np.testing.assert_array_equal(overlay[1, 0], [255, 0, 0])
        np.testing.assert_array_equal(overlay[1, 1], [0, 0, 0])

    def test_agreement_overlay_rejects_different_shapes(self) -> None:
        with self.assertRaises(ValueError):
            create_agreement_overlay(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((3, 2), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
