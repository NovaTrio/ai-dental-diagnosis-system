from __future__ import annotations

import unittest

from src.canalwidth.recommend_gp import (
    PROTOCOLS,
    format_gp,
    recommend_gp,
    select_iso_size,
)


class GPRecommendationTests(unittest.TestCase):
    def test_iso_policies_for_continuous_size_22(self) -> None:
        self.assertEqual(select_iso_size(22, "nearest"), 20)
        self.assertEqual(select_iso_size(22, "up"), 25)
        self.assertEqual(select_iso_size(22, "down"), 20)

    def test_requested_protocol_examples(self) -> None:
        self.assertEqual(
            recommend_gp(22, PROTOCOLS["nearest-06"]),
            (20, "20/.06"),
        )
        self.assertEqual(
            recommend_gp(22, PROTOCOLS["upsize-04"]),
            (25, "25/.04"),
        )

    def test_gp_label_has_two_digit_taper(self) -> None:
        self.assertEqual(format_gp(6, 0.04), "06/.04")
        self.assertEqual(format_gp(25, 0.06), "25/.06")


if __name__ == "__main__":
    unittest.main()
