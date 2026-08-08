from __future__ import annotations

import unittest

from api.services.rct_risk_service import (
    assess_lesion_risk,
    assess_overall_rct_risk,
)


class RCTRiskTests(unittest.TestCase):
    def test_lesion_risk_uses_five_mm_threshold(self) -> None:
        self.assertEqual(
            assess_lesion_risk({"lesion_detected": True, "lesion_diameter_mm": 4.99}),
            "Low",
        )
        self.assertEqual(
            assess_lesion_risk({"lesion_detected": True, "lesion_diameter_mm": 5.0}),
            "High",
        )
        self.assertEqual(assess_lesion_risk({"lesion_detected": False}), "Low")

    def test_highest_risk_wins_for_every_supported_combination(self) -> None:
        expected = {
            ("Low", "Low"): "Low",
            ("Moderate", "Low"): "Moderate",
            ("High", "Low"): "High",
            ("Low", "High"): "High",
            ("Moderate", "High"): "High",
            ("High", "High"): "High",
        }
        for risks, overall in expected.items():
            with self.subTest(risks=risks):
                result = assess_overall_rct_risk(*risks)
                self.assertEqual(result["overall_risk"], overall)
                self.assertTrue(result["treatment_outlook"])

    def test_unknown_risk_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assess_overall_rct_risk("Unknown", "Low")


if __name__ == "__main__":
    unittest.main()
