"""Map continuous apical preparation predictions to master GP recommendations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.canalwidth.models.train_binding_file_model import ISO_FILE_SIZES


DEFAULT_BINDING_PREDICTIONS = Path(
    "data/canalwidth/predictions/binding_files.csv"
)
DEFAULT_OUTPUT = Path("data/canalwidth/predictions/gp_recommendations.csv")


@dataclass(frozen=True)
class PreparationProtocol:
    name: str
    taper: float
    size_policy: str
    description: str


PROTOCOLS = {
    "nearest-04": PreparationProtocol(
        "nearest-04",
        0.04,
        "nearest",
        "Nearest ISO apical size with 4% taper",
    ),
    "upsize-04": PreparationProtocol(
        "upsize-04",
        0.04,
        "up",
        "Next ISO apical size with 4% taper",
    ),
    "nearest-06": PreparationProtocol(
        "nearest-06",
        0.06,
        "nearest",
        "Nearest ISO apical size with 6% taper",
    ),
    "upsize-06": PreparationProtocol(
        "upsize-06",
        0.06,
        "up",
        "Next ISO apical size with 6% taper",
    ),
}


def select_iso_size(value: float, policy: str = "nearest") -> int:
    """Select an ISO size using a declared preparation policy."""
    if not np.isfinite(value):
        raise ValueError("continuous file size must be finite")
    sizes = ISO_FILE_SIZES
    if policy == "nearest":
        return int(sizes[np.abs(sizes - value).argmin()])
    if policy == "up":
        candidates = sizes[sizes >= value]
        return int(candidates[0] if candidates.size else sizes[-1])
    if policy == "down":
        candidates = sizes[sizes <= value]
        return int(candidates[-1] if candidates.size else sizes[0])
    raise ValueError("policy must be 'nearest', 'up', or 'down'")


def format_gp(size: int, taper: float) -> str:
    if not np.isfinite(taper) or taper <= 0:
        raise ValueError("taper must be a positive finite value")
    return f"{size:02d}/.{int(round(taper * 100)):02d}"


def recommend_gp(
    continuous_file_size: float,
    protocol: PreparationProtocol,
) -> tuple[int, str]:
    size = select_iso_size(continuous_file_size, protocol.size_policy)
    return size, format_gp(size, protocol.taper)


def generate_recommendations(
    binding_predictions_csv: Path,
    protocol_name: str,
) -> pd.DataFrame:
    if protocol_name not in PROTOCOLS:
        raise ValueError(f"Unknown preparation protocol: {protocol_name}")
    if not binding_predictions_csv.is_file():
        raise FileNotFoundError(
            f"Binding-file predictions not found: {binding_predictions_csv}"
        )
    table = pd.read_csv(binding_predictions_csv)
    required = {
        "image_name",
        "predicted_wl_mm",
        "apical_diameter_px",
        "continuous_file_size_k",
    }
    if missing := required - set(table.columns):
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")

    protocol = PROTOCOLS[protocol_name]
    sizes = []
    labels = []
    for value in pd.to_numeric(
        table["continuous_file_size_k"], errors="coerce"
    ):
        size, label = recommend_gp(float(value), protocol)
        sizes.append(size)
        labels.append(label)
    result = table[
        [
            "image_name",
            "predicted_wl_mm",
            "apical_diameter_px",
            "continuous_file_size_k",
        ]
    ].copy()
    result["preparation_protocol"] = protocol.name
    result["size_policy"] = protocol.size_policy
    result["recommended_master_gp_size"] = sizes
    result["recommended_gp_taper"] = protocol.taper
    result["master_gp_recommendation"] = labels
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binding-predictions",
        type=Path,
        default=DEFAULT_BINDING_PREDICTIONS,
    )
    parser.add_argument(
        "--protocol",
        choices=sorted(PROTOCOLS),
        default="nearest-06",
        help=(
            "Explicit size/taper mapping protocol. For example, a continuous "
            "size of 22 maps to 20/.06 with nearest-06 and 25/.04 with upsize-04."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    recommendations = generate_recommendations(
        args.binding_predictions,
        args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(args.output, index=False)
    print(
        f"Saved {len(recommendations)} GP recommendation(s) "
        f"using protocol '{args.protocol}'"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
