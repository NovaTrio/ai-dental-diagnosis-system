from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FractureRiskResult:
    pdl_pattern_score: int
    pdl_pattern_label: str

    fracture_risk_score: int
    fracture_risk_label: str

    explanation: str


PDL_PATTERN_LABELS = {
    1: "Uniform widening",
    2: "Side-dominant widening",
    3: "Irregular localized widening",
}


def assess_fracture_risk(
    pdl_pattern_score: int,
) -> FractureRiskResult:
    """
    Convert the predicted PDL widening pattern into
    a rule-based hidden root fracture risk level.

    Mapping:
        1 -> Low
        2 -> Moderate
        3 -> High
    """

    if pdl_pattern_score not in PDL_PATTERN_LABELS:
        raise ValueError(
            f"Invalid PDL pattern score: {pdl_pattern_score}. "
            "Expected 1, 2, or 3."
        )

    pdl_pattern_label = PDL_PATTERN_LABELS[
        pdl_pattern_score
    ]

    # --------------------------------------------------------
    # Uniform widening
    # --------------------------------------------------------

    if pdl_pattern_score == 1:

        fracture_risk_score = 0

        fracture_risk_label = "Low"

        explanation = (
            "The PDL width profile is relatively uniform, "
            "with no strong persistent unilateral widening or "
            "marked irregular localized variation. This pattern "
            "is therefore mapped to a low hidden root fracture risk."
        )

    # --------------------------------------------------------
    # Side-dominant widening
    # --------------------------------------------------------

    elif pdl_pattern_score == 2:

        fracture_risk_score = 1

        fracture_risk_label = "Moderate"

        explanation = (
            "The PDL width profile shows persistent widening "
            "predominantly on one side of the root. This pattern "
            "is mapped to a moderate hidden root fracture risk."
        )

    # --------------------------------------------------------
    # Irregular localized widening
    # --------------------------------------------------------

    else:

        fracture_risk_score = 2

        fracture_risk_label = "High"

        explanation = (
            "The PDL width profile shows irregular localized "
            "widening and increased asymmetry or variation. "
            "This pattern is mapped to a high hidden root "
            "fracture risk."
        )

    return FractureRiskResult(
        pdl_pattern_score=pdl_pattern_score,
        pdl_pattern_label=pdl_pattern_label,
        fracture_risk_score=fracture_risk_score,
        fracture_risk_label=fracture_risk_label,
        explanation=explanation,
    )