from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass
class PDLFeatureConfig:
    """
    Parameters used for PDL width measurement and feature extraction.
    """

    # --------------------------------------------------------
    # Width scanning
    # --------------------------------------------------------

    # Maximum outward search distance relative to local root width.
    #
    # Example:
    # root width = 60 px
    # ratio      = 0.50
    # scan       = 30 px
    max_scan_root_ratio: float = 0.50

    # Prevent extremely small scan regions close to apex.
    min_scan_px: int = 5

    # Optional safety cap.
    max_scan_cap_px: int = 80

    # Small segmentation gaps allowed inside detected PDL.
    gap_tolerance: int = 2

    # --------------------------------------------------------
    # Smoothing
    # --------------------------------------------------------

    median_kernel_size: int = 3

    # --------------------------------------------------------
    # Side dominance
    # --------------------------------------------------------

    # Difference between normalized left/right widths required
    # before one side is considered dominant.
    #
    # 0.03 means 3% of local root width.
    dominance_margin_ratio: float = 0.03

    # --------------------------------------------------------
    # Localized widening
    # --------------------------------------------------------

    local_window_radius: int = 3

    # Current point must be at least this many times
    # larger than its local neighborhood baseline.
    local_ratio_threshold: float = 1.35

    # Also require absolute normalized difference.
    #
    # 0.03 = 3% of root width.
    local_absolute_margin_ratio: float = 0.03

    # --------------------------------------------------------
    # Localized asymmetry
    # --------------------------------------------------------

    asymmetry_ratio_threshold: float = 1.30
    asymmetry_absolute_margin_ratio: float = 0.025

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    min_valid_rows: int = 10


# ============================================================
# Output feature structure
# ============================================================

@dataclass
class PDLFeatures:

    image_name: str

    valid_row_count: int

    # ========================================================
    # Root geometry
    # ========================================================

    mean_root_width_px: float
    median_root_width_px: float

    # ========================================================
    # RAW PDL FEATURES
    # ========================================================

    mean_left_width_px: float
    mean_right_width_px: float
    mean_overall_width_px: float

    median_left_width_px: float
    median_right_width_px: float
    median_overall_width_px: float

    std_left_width_px: float
    std_right_width_px: float
    std_overall_width_px: float

    variance_left_width_px: float
    variance_right_width_px: float
    variance_overall_width_px: float

    min_left_width_px: float
    min_right_width_px: float
    min_overall_width_px: float

    max_left_width_px: float
    max_right_width_px: float
    max_overall_width_px: float

    p75_overall_width_px: float
    p90_overall_width_px: float
    p95_overall_width_px: float

    overall_width_cv: float

    # ========================================================
    # RAW ASYMMETRY
    # ========================================================

    mean_asymmetry_px: float
    median_asymmetry_px: float
    max_asymmetry_px: float

    asymmetry_variance_px: float
    asymmetry_std_px: float

    left_right_mean_difference_px: float

    # ========================================================
    # NORMALIZED WIDTH FEATURES
    # ========================================================

    mean_left_width_normalized: float
    mean_right_width_normalized: float
    mean_overall_width_normalized: float

    median_left_width_normalized: float
    median_right_width_normalized: float
    median_overall_width_normalized: float

    std_left_width_normalized: float
    std_right_width_normalized: float
    std_overall_width_normalized: float

    variance_left_width_normalized: float
    variance_right_width_normalized: float
    variance_overall_width_normalized: float

    max_left_width_normalized: float
    max_right_width_normalized: float
    max_overall_width_normalized: float

    p75_overall_width_normalized: float
    p90_overall_width_normalized: float
    p95_overall_width_normalized: float

    # ========================================================
    # NORMALIZED ASYMMETRY
    # ========================================================

    mean_asymmetry_normalized: float
    median_asymmetry_normalized: float
    max_asymmetry_normalized: float

    asymmetry_variance_normalized: float
    asymmetry_std_normalized: float

    left_right_mean_difference_normalized: float

    asymmetry_ratio: float

    # ========================================================
    # RELATIONSHIP BETWEEN SIDES
    # ========================================================

    left_right_correlation: float

    left_dominant_ratio: float
    right_dominant_ratio: float
    balanced_ratio: float

    dominant_side: str

    side_dominant_length_ratio: float
    side_consistency: float
    side_switch_count: int

    # ========================================================
    # LOCALIZED WIDTH CHANGES
    # ========================================================

    width_spike_count: int
    width_spike_ratio: float
    width_max_spike_ratio: float

    asymmetry_spike_count: int
    asymmetry_spike_ratio: float
    asymmetry_max_spike_ratio: float

    # ========================================================
    # ROOT THIRDS
    # ========================================================

    coronal_mean_width_normalized: float
    middle_mean_width_normalized: float
    apical_mean_width_normalized: float

    coronal_mean_asymmetry_normalized: float
    middle_mean_asymmetry_normalized: float
    apical_mean_asymmetry_normalized: float


# ============================================================
# Mask utilities
# ============================================================

def load_binary_mask(mask_path) -> np.ndarray:
    """
    Load an image mask and convert it to 0/1 binary form.
    """

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not load mask: {mask_path}"
        )

    return (mask > 0).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep the largest connected foreground component.

    Useful for removing isolated noise from the root mask.
    """

    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return binary

    largest_label = 1 + np.argmax(
        stats[
            1:,
            cv2.CC_STAT_AREA,
        ]
    )

    output = np.zeros_like(binary)

    output[
        labels == largest_label
    ] = 1

    return output


# ============================================================
# PDL outward scanning
# ============================================================

def scan_pdl_width(
    pdl_row: np.ndarray,
    start_x: int,
    direction: int,
    max_scan_px: int,
    gap_tolerance: int,
) -> float:
    """
    Scan outward from one root boundary and measure attached PDL.

    direction:
        -1 = scan left
         1 = scan right
    """

    image_width = len(pdl_row)

    x = start_x

    detected_width = 0
    gap = 0

    found_pdl = False

    for _ in range(max_scan_px):

        if x < 0 or x >= image_width:
            break

        if pdl_row[x] > 0:

            found_pdl = True

            # Include previously tolerated small gap.
            detected_width += gap
            gap = 0

            detected_width += 1

        else:

            if found_pdl:

                gap += 1

                if gap > gap_tolerance:
                    break

        x += direction

    return float(detected_width)


# ============================================================
# Row-wise width measurement
# ============================================================

def measure_width_profiles(
    root_mask: np.ndarray,
    pdl_mask: np.ndarray,
    config: PDLFeatureConfig,
):
    """
    For each root row:

        1. Find left root boundary
        2. Find right root boundary
        3. Calculate local root width
        4. Determine resolution-relative scan distance
        5. Measure left PDL
        6. Measure right PDL
    """

    root = keep_largest_component(
        root_mask
    )

    pdl = (
        pdl_mask > 0
    ).astype(np.uint8)

    if root.shape != pdl.shape:

        raise ValueError(
            f"Root mask shape {root.shape} "
            f"does not match PDL mask shape {pdl.shape}"
        )

    y_positions = []

    left_widths = []
    right_widths = []

    root_widths = []

    height, _ = root.shape

    for y in range(height):

        root_x = np.flatnonzero(
            root[y] > 0
        )

        if root_x.size == 0:
            continue

        left_root_x = int(
            root_x.min()
        )

        right_root_x = int(
            root_x.max()
        )

        root_width = (
            right_root_x
            - left_root_x
            + 1
        )

        if root_width <= 0:
            continue

        # ----------------------------------------------------
        # Resolution-relative scan distance
        # ----------------------------------------------------

        row_max_scan = int(
            round(
                root_width
                * config.max_scan_root_ratio
            )
        )

        row_max_scan = max(
            config.min_scan_px,
            row_max_scan,
        )

        row_max_scan = min(
            config.max_scan_cap_px,
            row_max_scan,
        )

        # ----------------------------------------------------
        # Left PDL
        # ----------------------------------------------------

        left_width = scan_pdl_width(
            pdl_row=pdl[y],
            start_x=left_root_x - 1,
            direction=-1,
            max_scan_px=row_max_scan,
            gap_tolerance=config.gap_tolerance,
        )

        # ----------------------------------------------------
        # Right PDL
        # ----------------------------------------------------

        right_width = scan_pdl_width(
            pdl_row=pdl[y],
            start_x=right_root_x + 1,
            direction=1,
            max_scan_px=row_max_scan,
            gap_tolerance=config.gap_tolerance,
        )

        # Skip rows where there is no PDL on either side.
        if (
            left_width <= 0
            and right_width <= 0
        ):
            continue

        y_positions.append(y)

        left_widths.append(
            left_width
        )

        right_widths.append(
            right_width
        )

        root_widths.append(
            root_width
        )

    return (
        np.asarray(
            y_positions,
            dtype=np.int32,
        ),
        np.asarray(
            left_widths,
            dtype=np.float32,
        ),
        np.asarray(
            right_widths,
            dtype=np.float32,
        ),
        np.asarray(
            root_widths,
            dtype=np.float32,
        ),
    )


# ============================================================
# Profile smoothing
# ============================================================

def median_smooth_profile(
    values: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Apply mild median smoothing.

    Keeps real localized widening while reducing isolated
    one-row segmentation noise.
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    n = len(values)

    if (
        n < 3
        or kernel_size <= 1
    ):
        return values.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    if kernel_size > n:

        kernel_size = n

        if kernel_size % 2 == 0:
            kernel_size -= 1

    if kernel_size < 3:
        return values.copy()

    output = cv2.medianBlur(
        values.reshape(-1, 1),
        kernel_size,
    )

    return (
        output
        .reshape(-1)
        .astype(np.float32)
    )


# ============================================================
# Safe statistics
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:

    if abs(denominator) < 1e-8:
        return default

    value = numerator / denominator

    if not np.isfinite(value):
        return default

    return float(value)


def safe_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    if (
        len(a) < 2
        or len(b) < 2
    ):
        return 0.0

    if (
        np.std(a) < 1e-8
        or np.std(b) < 1e-8
    ):
        return 0.0

    correlation = np.corrcoef(
        a,
        b,
    )[0, 1]

    if not np.isfinite(
        correlation
    ):
        return 0.0

    return float(
        correlation
    )


# ============================================================
# Side dominance
# ============================================================

def calculate_side_dominance(
    left_normalized: np.ndarray,
    right_normalized: np.ndarray,
    margin_ratio: float,
):
    """
    Classify every row as:

        -1 = left dominant
         0 = balanced
         1 = right dominant

    Uses normalized widths rather than raw pixels.
    """

    difference = (
        right_normalized
        - left_normalized
    )

    side_profile = np.zeros(
        len(difference),
        dtype=np.int8,
    )

    side_profile[
        difference > margin_ratio
    ] = 1

    side_profile[
        difference < -margin_ratio
    ] = -1

    n = len(
        side_profile
    )

    if n == 0:

        return {
            "left_dominant_ratio": 0.0,
            "right_dominant_ratio": 0.0,
            "balanced_ratio": 0.0,
            "dominant_side": "NONE",
            "side_dominant_length_ratio": 0.0,
            "side_consistency": 0.0,
            "side_switch_count": 0,
            "side_profile": side_profile,
        }

    left_ratio = float(
        np.mean(
            side_profile == -1
        )
    )

    right_ratio = float(
        np.mean(
            side_profile == 1
        )
    )

    balanced_ratio = float(
        np.mean(
            side_profile == 0
        )
    )

    if left_ratio > right_ratio:

        dominant_side = "LEFT"
        dominant_ratio = left_ratio

    elif right_ratio > left_ratio:

        dominant_side = "RIGHT"
        dominant_ratio = right_ratio

    else:

        dominant_side = "BALANCED"
        dominant_ratio = max(
            left_ratio,
            right_ratio,
        )

    non_balanced = side_profile[
        side_profile != 0
    ]

    if len(
        non_balanced
    ) == 0:

        switch_count = 0
        consistency = 0.0

    else:

        switch_count = int(
            np.sum(
                non_balanced[1:]
                != non_balanced[:-1]
            )
        )

        left_count = int(
            np.sum(
                non_balanced == -1
            )
        )

        right_count = int(
            np.sum(
                non_balanced == 1
            )
        )

        consistency = safe_divide(
            max(
                left_count,
                right_count,
            ),
            len(
                non_balanced
            ),
        )

    return {
        "left_dominant_ratio":
            left_ratio,

        "right_dominant_ratio":
            right_ratio,

        "balanced_ratio":
            balanced_ratio,

        "dominant_side":
            dominant_side,

        "side_dominant_length_ratio":
            float(dominant_ratio),

        "side_consistency":
            float(consistency),

        "side_switch_count":
            switch_count,

        "side_profile":
            side_profile,
    }


# ============================================================
# Local spike detection
# ============================================================

def detect_local_spikes(
    profile: np.ndarray,
    window_radius: int,
    ratio_threshold: float,
    absolute_margin: float,
):
    """
    Detect localized widening relative to neighboring rows.

    Example:

        0.08
        0.09
        0.08
        0.19   <- spike
        0.09
        0.08
    """

    values = np.asarray(
        profile,
        dtype=np.float32,
    )

    n = len(
        values
    )

    flags = np.zeros(
        n,
        dtype=bool,
    )

    ratios = np.ones(
        n,
        dtype=np.float32,
    )

    for i in range(n):

        start = max(
            0,
            i - window_radius,
        )

        end = min(
            n,
            i + window_radius + 1,
        )

        indexes = np.arange(
            start,
            end,
        )

        neighborhood = values[
            start:end
        ]

        # Remove center point.
        neighborhood = neighborhood[
            indexes != i
        ]

        if len(
            neighborhood
        ) == 0:
            continue

        positive_neighbors = neighborhood[
            neighborhood > 0
        ]

        if len(
            positive_neighbors
        ) == 0:
            continue

        baseline = float(
            np.median(
                positive_neighbors
            )
        )

        if baseline <= 1e-8:
            continue

        current = float(
            values[i]
        )

        ratio = (
            current
            / baseline
        )

        difference = (
            current
            - baseline
        )

        ratios[i] = ratio

        if (
            ratio
            >= ratio_threshold
            and difference
            >= absolute_margin
        ):
            flags[i] = True

    count = int(
        np.sum(
            flags
        )
    )

    spike_ratio = safe_divide(
        count,
        n,
    )

    if n > 0:

        max_ratio = float(
            np.max(
                ratios
            )
        )

    else:

        max_ratio = 0.0

    return {
        "flags": flags,
        "count": count,
        "ratio": spike_ratio,
        "max_ratio": max_ratio,
    }


# ============================================================
# Root thirds
# ============================================================

def split_into_root_thirds(
    values: np.ndarray,
):
    """
    Split profile according to relative root position:

        first third  = coronal
        second third = middle
        third third  = apical
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    sections = np.array_split(
        values,
        3,
    )

    means = []

    for section in sections:

        if len(
            section
        ) == 0:

            means.append(
                0.0
            )

        else:

            means.append(
                float(
                    np.mean(
                        section
                    )
                )
            )

    while len(means) < 3:

        means.append(
            0.0
        )

    return (
        means[0],
        means[1],
        means[2],
    )


# ============================================================
# Main extraction
# ============================================================

def extract_pdl_features(
    root_mask: np.ndarray,
    pdl_mask: np.ndarray,
    image_name: str,
    config: Optional[
        PDLFeatureConfig
    ] = None,
):
    """
    Extract raw and normalized PDL features for one tooth.
    """

    if config is None:
        config = PDLFeatureConfig()

    # --------------------------------------------------------
    # 1. Raw row-wise measurements
    # --------------------------------------------------------

    (
        y_positions,
        left_raw,
        right_raw,
        root_widths,
    ) = measure_width_profiles(
        root_mask=root_mask,
        pdl_mask=pdl_mask,
        config=config,
    )

    if (
        len(y_positions)
        < config.min_valid_rows
    ):

        raise ValueError(
            f"{image_name}: only "
            f"{len(y_positions)} valid rows found; "
            f"minimum = {config.min_valid_rows}"
        )

    # --------------------------------------------------------
    # 2. Mild smoothing
    # --------------------------------------------------------

    left = median_smooth_profile(
        left_raw,
        config.median_kernel_size,
    )

    right = median_smooth_profile(
        right_raw,
        config.median_kernel_size,
    )

    root_widths_smoothed = (
        median_smooth_profile(
            root_widths,
            config.median_kernel_size,
        )
    )

    # Ensure denominator is never zero.
    safe_root_widths = np.maximum(
        root_widths_smoothed,
        1.0,
    )

    # --------------------------------------------------------
    # 3. Raw bilateral width
    # --------------------------------------------------------

    overall = (
        left + right
    ) / 2.0

    raw_asymmetry = np.abs(
        left - right
    )

    # --------------------------------------------------------
    # 4. Resolution-normalized width
    # --------------------------------------------------------

    left_norm = (
        left
        / safe_root_widths
    )

    right_norm = (
        right
        / safe_root_widths
    )

    overall_norm = (
        left_norm
        + right_norm
    ) / 2.0

    asymmetry_norm = np.abs(
        left_norm
        - right_norm
    )

    # --------------------------------------------------------
    # 5. Normalized root position
    # --------------------------------------------------------

    if len(
        y_positions
    ) > 1:

        normalized_y = np.linspace(
            0.0,
            1.0,
            len(y_positions),
            dtype=np.float32,
        )

    else:

        normalized_y = np.zeros(
            len(y_positions),
            dtype=np.float32,
        )

    # ========================================================
    # ROOT GEOMETRY
    # ========================================================

    mean_root_width = float(
        np.mean(
            root_widths
        )
    )

    median_root_width = float(
        np.median(
            root_widths
        )
    )

    # ========================================================
    # RAW WIDTH STATISTICS
    # ========================================================

    mean_left = float(
        np.mean(left)
    )

    mean_right = float(
        np.mean(right)
    )

    mean_overall = float(
        np.mean(overall)
    )

    median_left = float(
        np.median(left)
    )

    median_right = float(
        np.median(right)
    )

    median_overall = float(
        np.median(overall)
    )

    std_left = float(
        np.std(left)
    )

    std_right = float(
        np.std(right)
    )

    std_overall = float(
        np.std(overall)
    )

    variance_left = float(
        np.var(left)
    )

    variance_right = float(
        np.var(right)
    )

    variance_overall = float(
        np.var(overall)
    )

    min_left = float(
        np.min(left)
    )

    min_right = float(
        np.min(right)
    )

    min_overall = float(
        np.min(overall)
    )

    max_left = float(
        np.max(left)
    )

    max_right = float(
        np.max(right)
    )

    max_overall = float(
        np.max(overall)
    )

    p75 = float(
        np.percentile(
            overall,
            75,
        )
    )

    p90 = float(
        np.percentile(
            overall,
            90,
        )
    )

    p95 = float(
        np.percentile(
            overall,
            95,
        )
    )

    raw_cv = safe_divide(
        std_overall,
        mean_overall,
    )

    # ========================================================
    # RAW ASYMMETRY
    # ========================================================

    mean_asymmetry_px = float(
        np.mean(
            raw_asymmetry
        )
    )

    median_asymmetry_px = float(
        np.median(
            raw_asymmetry
        )
    )

    max_asymmetry_px = float(
        np.max(
            raw_asymmetry
        )
    )

    asymmetry_variance_px = float(
        np.var(
            raw_asymmetry
        )
    )

    asymmetry_std_px = float(
        np.std(
            raw_asymmetry
        )
    )

    raw_mean_difference = abs(
        mean_left
        - mean_right
    )

    # ========================================================
    # NORMALIZED WIDTH STATISTICS
    # ========================================================

    mean_left_norm = float(
        np.mean(
            left_norm
        )
    )

    mean_right_norm = float(
        np.mean(
            right_norm
        )
    )

    mean_overall_norm = float(
        np.mean(
            overall_norm
        )
    )

    median_left_norm = float(
        np.median(
            left_norm
        )
    )

    median_right_norm = float(
        np.median(
            right_norm
        )
    )

    median_overall_norm = float(
        np.median(
            overall_norm
        )
    )

    std_left_norm = float(
        np.std(
            left_norm
        )
    )

    std_right_norm = float(
        np.std(
            right_norm
        )
    )

    std_overall_norm = float(
        np.std(
            overall_norm
        )
    )

    variance_left_norm = float(
        np.var(
            left_norm
        )
    )

    variance_right_norm = float(
        np.var(
            right_norm
        )
    )

    variance_overall_norm = float(
        np.var(
            overall_norm
        )
    )

    max_left_norm = float(
        np.max(
            left_norm
        )
    )

    max_right_norm = float(
        np.max(
            right_norm
        )
    )

    max_overall_norm = float(
        np.max(
            overall_norm
        )
    )

    p75_norm = float(
        np.percentile(
            overall_norm,
            75,
        )
    )

    p90_norm = float(
        np.percentile(
            overall_norm,
            90,
        )
    )

    p95_norm = float(
        np.percentile(
            overall_norm,
            95,
        )
    )

    # ========================================================
    # NORMALIZED ASYMMETRY
    # ========================================================

    mean_asymmetry_norm = float(
        np.mean(
            asymmetry_norm
        )
    )

    median_asymmetry_norm = float(
        np.median(
            asymmetry_norm
        )
    )

    max_asymmetry_norm = float(
        np.max(
            asymmetry_norm
        )
    )

    asymmetry_variance_norm = float(
        np.var(
            asymmetry_norm
        )
    )

    asymmetry_std_norm = float(
        np.std(
            asymmetry_norm
        )
    )

    normalized_mean_difference = abs(
        mean_left_norm
        - mean_right_norm
    )

    asymmetry_ratio = safe_divide(
        mean_asymmetry_norm,
        mean_overall_norm,
    )

    # ========================================================
    # LEFT-RIGHT CORRELATION
    # ========================================================

    correlation = safe_correlation(
        left_norm,
        right_norm,
    )

    # ========================================================
    # SIDE DOMINANCE
    # ========================================================

    dominance = calculate_side_dominance(
        left_normalized=left_norm,
        right_normalized=right_norm,
        margin_ratio=(
            config.dominance_margin_ratio
        ),
    )

    # ========================================================
    # WIDTH SPIKES
    # ========================================================

    width_spikes = detect_local_spikes(
        profile=overall_norm,
        window_radius=(
            config.local_window_radius
        ),
        ratio_threshold=(
            config.local_ratio_threshold
        ),
        absolute_margin=(
            config.local_absolute_margin_ratio
        ),
    )

    # ========================================================
    # ASYMMETRY SPIKES
    # ========================================================

    asymmetry_spikes = detect_local_spikes(
        profile=asymmetry_norm,
        window_radius=(
            config.local_window_radius
        ),
        ratio_threshold=(
            config.asymmetry_ratio_threshold
        ),
        absolute_margin=(
            config.asymmetry_absolute_margin_ratio
        ),
    )

    # ========================================================
    # ROOT THIRDS
    # ========================================================

    (
        coronal_width,
        middle_width,
        apical_width,
    ) = split_into_root_thirds(
        overall_norm
    )

    (
        coronal_asymmetry,
        middle_asymmetry,
        apical_asymmetry,
    ) = split_into_root_thirds(
        asymmetry_norm
    )

    # ========================================================
    # Final features
    # ========================================================

    features = PDLFeatures(

        image_name=image_name,

        valid_row_count=len(
            y_positions
        ),

        mean_root_width_px=
            mean_root_width,

        median_root_width_px=
            median_root_width,

        # RAW WIDTHS
        mean_left_width_px=
            mean_left,

        mean_right_width_px=
            mean_right,

        mean_overall_width_px=
            mean_overall,

        median_left_width_px=
            median_left,

        median_right_width_px=
            median_right,

        median_overall_width_px=
            median_overall,

        std_left_width_px=
            std_left,

        std_right_width_px=
            std_right,

        std_overall_width_px=
            std_overall,

        variance_left_width_px=
            variance_left,

        variance_right_width_px=
            variance_right,

        variance_overall_width_px=
            variance_overall,

        min_left_width_px=
            min_left,

        min_right_width_px=
            min_right,

        min_overall_width_px=
            min_overall,

        max_left_width_px=
            max_left,

        max_right_width_px=
            max_right,

        max_overall_width_px=
            max_overall,

        p75_overall_width_px=
            p75,

        p90_overall_width_px=
            p90,

        p95_overall_width_px=
            p95,

        overall_width_cv=
            raw_cv,

        # RAW ASYMMETRY
        mean_asymmetry_px=
            mean_asymmetry_px,

        median_asymmetry_px=
            median_asymmetry_px,

        max_asymmetry_px=
            max_asymmetry_px,

        asymmetry_variance_px=
            asymmetry_variance_px,

        asymmetry_std_px=
            asymmetry_std_px,

        left_right_mean_difference_px=
            raw_mean_difference,

        # NORMALIZED WIDTH
        mean_left_width_normalized=
            mean_left_norm,

        mean_right_width_normalized=
            mean_right_norm,

        mean_overall_width_normalized=
            mean_overall_norm,

        median_left_width_normalized=
            median_left_norm,

        median_right_width_normalized=
            median_right_norm,

        median_overall_width_normalized=
            median_overall_norm,

        std_left_width_normalized=
            std_left_norm,

        std_right_width_normalized=
            std_right_norm,

        std_overall_width_normalized=
            std_overall_norm,

        variance_left_width_normalized=
            variance_left_norm,

        variance_right_width_normalized=
            variance_right_norm,

        variance_overall_width_normalized=
            variance_overall_norm,

        max_left_width_normalized=
            max_left_norm,

        max_right_width_normalized=
            max_right_norm,

        max_overall_width_normalized=
            max_overall_norm,

        p75_overall_width_normalized=
            p75_norm,

        p90_overall_width_normalized=
            p90_norm,

        p95_overall_width_normalized=
            p95_norm,

        # NORMALIZED ASYMMETRY
        mean_asymmetry_normalized=
            mean_asymmetry_norm,

        median_asymmetry_normalized=
            median_asymmetry_norm,

        max_asymmetry_normalized=
            max_asymmetry_norm,

        asymmetry_variance_normalized=
            asymmetry_variance_norm,

        asymmetry_std_normalized=
            asymmetry_std_norm,

        left_right_mean_difference_normalized=
            normalized_mean_difference,

        asymmetry_ratio=
            asymmetry_ratio,

        # LEFT-RIGHT RELATIONSHIP
        left_right_correlation=
            correlation,

        left_dominant_ratio=
            dominance[
                "left_dominant_ratio"
            ],

        right_dominant_ratio=
            dominance[
                "right_dominant_ratio"
            ],

        balanced_ratio=
            dominance[
                "balanced_ratio"
            ],

        dominant_side=
            dominance[
                "dominant_side"
            ],

        side_dominant_length_ratio=
            dominance[
                "side_dominant_length_ratio"
            ],

        side_consistency=
            dominance[
                "side_consistency"
            ],

        side_switch_count=
            dominance[
                "side_switch_count"
            ],

        # LOCAL WIDTH SPIKES
        width_spike_count=
            width_spikes[
                "count"
            ],

        width_spike_ratio=
            width_spikes[
                "ratio"
            ],

        width_max_spike_ratio=
            width_spikes[
                "max_ratio"
            ],

        # ASYMMETRY SPIKES
        asymmetry_spike_count=
            asymmetry_spikes[
                "count"
            ],

        asymmetry_spike_ratio=
            asymmetry_spikes[
                "ratio"
            ],

        asymmetry_max_spike_ratio=
            asymmetry_spikes[
                "max_ratio"
            ],

        # ROOT THIRDS
        coronal_mean_width_normalized=
            coronal_width,

        middle_mean_width_normalized=
            middle_width,

        apical_mean_width_normalized=
            apical_width,

        coronal_mean_asymmetry_normalized=
            coronal_asymmetry,

        middle_mean_asymmetry_normalized=
            middle_asymmetry,

        apical_mean_asymmetry_normalized=
            apical_asymmetry,
    )

    # ========================================================
    # Detailed profiles for debugging
    # ========================================================

    profiles = {

        "y_positions":
            y_positions,

        "normalized_y":
            normalized_y,

        "root_width_px":
            root_widths,

        "root_width_smoothed_px":
            root_widths_smoothed,

        "left_raw_px":
            left_raw,

        "right_raw_px":
            right_raw,

        "left_smoothed_px":
            left,

        "right_smoothed_px":
            right,

        "overall_width_px":
            overall,

        "asymmetry_px":
            raw_asymmetry,

        "left_normalized":
            left_norm,

        "right_normalized":
            right_norm,

        "overall_normalized":
            overall_norm,

        "asymmetry_normalized":
            asymmetry_norm,

        "side_profile":
            dominance[
                "side_profile"
            ],

        "width_spike_flags":
            width_spikes[
                "flags"
            ],

        "asymmetry_spike_flags":
            asymmetry_spikes[
                "flags"
            ],
    }

    return (
        features,
        profiles,
    )


# ============================================================
# Convert feature object to dictionary
# ============================================================

def features_to_dict(
    features: PDLFeatures,
) -> dict:

    return asdict(
        features
    )