"""Rule-based combination of fracture and lesion findings for RCT screening."""

from __future__ import annotations

from typing import Any, Literal


RiskLevel = Literal["Low", "Moderate", "High"]

LESION_HIGH_RISK_DIAMETER_MM = 5.0
RISK_ORDER: dict[RiskLevel, int] = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
}
TREATMENT_OUTLOOK: dict[RiskLevel, str] = {
    "Low": "RCT success is likely to be high.",
    "Moderate": (
        "RCT may be successful, but careful clinical evaluation and follow-up "
        "are recommended."
    ),
    "High": (
        "RCT success is likely to be very low. Consider alternative treatment "
        "after specialist evaluation."
    ),
}


def assess_lesion_risk(lesion: dict[str, Any]) -> RiskLevel:
    """Map lesion detection and major-axis diameter to the project risk rule."""
    if not lesion.get("lesion_detected", False):
        return "Low"

    diameter = lesion.get("lesion_diameter_mm")
    if diameter is None:
        raise ValueError("A detected lesion must have a measured diameter")
    return "High" if float(diameter) >= LESION_HIGH_RISK_DIAMETER_MM else "Low"


def assess_overall_rct_risk(
    fracture_risk: str,
    lesion_risk: str,
) -> dict[str, str]:
    """Apply the conservative highest-risk-wins combination matrix."""
    if fracture_risk not in RISK_ORDER:
        raise ValueError(f"Unknown fracture risk: {fracture_risk}")
    if lesion_risk not in RISK_ORDER:
        raise ValueError(f"Unknown lesion risk: {lesion_risk}")

    typed_fracture_risk: RiskLevel = fracture_risk  # type: ignore[assignment]
    typed_lesion_risk: RiskLevel = lesion_risk  # type: ignore[assignment]
    overall_risk = max(
        (typed_fracture_risk, typed_lesion_risk),
        key=RISK_ORDER.__getitem__,
    )
    return {
        "overall_risk": overall_risk,
        "treatment_outlook": TREATMENT_OUTLOOK[overall_risk],
    }
