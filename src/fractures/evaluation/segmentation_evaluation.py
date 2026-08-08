import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# Basic utilities
# ============================================================

def ensure_uint8_gray(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    return img


def read_binary_mask(path, target_shape=None):
    if path is None or not os.path.exists(path):
        return None

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = ensure_uint8_gray(mask)

    if mask is None:
        return None

    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask = ((mask > 0).astype(np.uint8) * 255)

    return mask


def safe_imwrite(path, image):
    if path is None:
        return

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    cv2.imwrite(path, image)


def safe_div(a, b, default=0.0):
    if b == 0:
        return default

    return float(a) / float(b)


def find_mask_path(mask_dir, original_file_name, prefixes):
    """
    Search mask files using different prefixes and extensions.

    Original:
        frac01.jpg

    Possible masks:
        manual_pdl_frac01.png
        manual_pdl_frac01.jpg
        gt_pdl_frac01.png
        pdl_frac01.png
        frac01.png
    """

    name, original_ext = os.path.splitext(original_file_name)

    valid_ext = [
        ".png",
        original_ext,
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    ]

    candidates = []

    for ext in valid_ext:
        candidates.append(f"{name}{ext}")

    for prefix in prefixes:
        for ext in valid_ext:
            candidates.append(f"{prefix}{name}{ext}")

    seen = set()

    for candidate in candidates:
        if candidate in seen:
            continue

        seen.add(candidate)

        path = os.path.join(mask_dir, candidate)

        if os.path.exists(path):
            return path

    return None


# ============================================================
# Overlap metrics
# ============================================================

def compute_overlap_metrics(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0

    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & (~gt))
    fn = np.count_nonzero((~pred) & gt)
    tn = np.count_nonzero((~pred) & (~gt))

    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)

    f1 = safe_div(2 * precision * recall, precision + recall)

    false_positive_rate = safe_div(fp, fp + tn)
    false_negative_rate = safe_div(fn, fn + tp)

    pred_area = np.count_nonzero(pred)
    gt_area = np.count_nonzero(gt)

    area_difference = pred_area - gt_area
    absolute_area_error = abs(area_difference)
    relative_area_error = safe_div(absolute_area_error, gt_area)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),

        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1,

        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,

        "pred_area_px": int(pred_area),
        "gt_area_px": int(gt_area),
        "area_difference_px": int(area_difference),
        "absolute_area_error_px": int(absolute_area_error),
        "relative_area_error": relative_area_error
    }


# ============================================================
# Boundary metrics
# ============================================================

def get_boundary(mask):
    """
    Extract 1-pixel boundary from a binary mask.
    """

    mask_u8 = ((mask > 0).astype(np.uint8) * 255)

    if np.count_nonzero(mask_u8) == 0:
        return np.zeros_like(mask_u8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = cv2.subtract(mask_u8, eroded)

    return boundary


def boundary_distances(source_boundary, target_boundary):
    """
    For every boundary pixel in source_boundary, compute distance to
    nearest boundary pixel in target_boundary.
    """

    source = source_boundary > 0
    target = target_boundary > 0

    if np.count_nonzero(source) == 0 or np.count_nonzero(target) == 0:
        return np.array([], dtype=np.float32)

    # distanceTransform measures distance to nearest zero pixel.
    # Therefore target boundary should be zero, background should be 255.
    target_inv = np.where(target, 0, 255).astype(np.uint8)

    dist_map = cv2.distanceTransform(
        target_inv,
        cv2.DIST_L2,
        5
    )

    distances = dist_map[source]

    return distances.astype(np.float32)


def compute_boundary_metrics(pred_mask, gt_mask):
    pred_boundary = get_boundary(pred_mask)
    gt_boundary = get_boundary(gt_mask)

    d_pred_to_gt = boundary_distances(
        source_boundary=pred_boundary,
        target_boundary=gt_boundary
    )

    d_gt_to_pred = boundary_distances(
        source_boundary=gt_boundary,
        target_boundary=pred_boundary
    )

    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return {
            "hausdorff_distance_px": 0.0,
            "hd95_px": 0.0,
            "average_surface_distance_px": 0.0,
            "mean_pred_to_gt_surface_distance_px": 0.0,
            "mean_gt_to_pred_surface_distance_px": 0.0
        }

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    hausdorff = float(np.max(all_distances))
    hd95 = float(np.percentile(all_distances, 95))

    mean_pred_to_gt = float(np.mean(d_pred_to_gt))
    mean_gt_to_pred = float(np.mean(d_gt_to_pred))

    average_surface_distance = float(
        (mean_pred_to_gt + mean_gt_to_pred) / 2.0
    )

    return {
        "hausdorff_distance_px": hausdorff,
        "hd95_px": hd95,
        "average_surface_distance_px": average_surface_distance,
        "mean_pred_to_gt_surface_distance_px": mean_pred_to_gt,
        "mean_gt_to_pred_surface_distance_px": mean_gt_to_pred
    }


# ============================================================
# Shape metrics
# ============================================================

def compute_shape_metrics(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0

    pred_area = np.count_nonzero(pred)
    gt_area = np.count_nonzero(gt)

    pred_components = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8),
        8
    )

    gt_components = cv2.connectedComponentsWithStats(
        gt.astype(np.uint8),
        8
    )

    pred_num_labels, pred_labels, pred_stats, _ = pred_components
    gt_num_labels, gt_labels, gt_stats, _ = gt_components

    pred_component_count = max(0, pred_num_labels - 1)
    gt_component_count = max(0, gt_num_labels - 1)

    pred_largest_area = 0
    gt_largest_area = 0

    if pred_num_labels > 1:
        pred_largest_area = int(
            np.max(pred_stats[1:, cv2.CC_STAT_AREA])
        )

    if gt_num_labels > 1:
        gt_largest_area = int(
            np.max(gt_stats[1:, cv2.CC_STAT_AREA])
        )

    return {
        "pred_component_count": int(pred_component_count),
        "gt_component_count": int(gt_component_count),
        "pred_largest_component_area_px": pred_largest_area,
        "gt_largest_component_area_px": gt_largest_area,
        "pred_largest_component_ratio": safe_div(pred_largest_area, pred_area),
        "gt_largest_component_ratio": safe_div(gt_largest_area, gt_area)
    }


# ============================================================
# Debug visualization
# ============================================================

def create_segmentation_debug_image(
    image,
    pred_mask,
    gt_mask,
    title_text=None
):
    """
    Colors:
        Green  = ground truth only
        Red    = prediction only
        Yellow = overlap
    """

    if image is None:
        image = np.zeros_like(pred_mask)

    image = ensure_uint8_gray(image)

    if image.shape != pred_mask.shape:
        image = cv2.resize(
            image,
            (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    pred = pred_mask > 0
    gt = gt_mask > 0

    overlap = pred & gt
    pred_only = pred & (~gt)
    gt_only = gt & (~pred)

    overlay = base.copy()

    # Ground truth only: green
    overlay[gt_only] = (
        0.45 * overlay[gt_only]
        + 0.55 * np.array([0, 255, 0])
    ).astype(np.uint8)

    # Prediction only: red
    overlay[pred_only] = (
        0.45 * overlay[pred_only]
        + 0.55 * np.array([0, 0, 255])
    ).astype(np.uint8)

    # Overlap: yellow
    overlay[overlap] = (
        0.45 * overlay[overlap]
        + 0.55 * np.array([0, 255, 255])
    ).astype(np.uint8)

    pred_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

    pred_boundary = get_boundary(pred_mask)
    gt_boundary = get_boundary(gt_mask)

    boundary_view = base.copy()

    boundary_view[gt_boundary > 0] = (0, 255, 0)
    boundary_view[pred_boundary > 0] = (0, 0, 255)

    if title_text is not None:
        cv2.putText(
            overlay,
            title_text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    debug = np.hstack(
        [
            base,
            gt_bgr,
            pred_bgr,
            overlay,
            boundary_view
        ]
    )

    return debug


# ============================================================
# Single mask-pair evaluation
# ============================================================

def evaluate_mask_pair(
    image_path,
    pred_mask_path,
    gt_mask_path,
    region_name,
    debug_path=None
):
    image = None

    if image_path is not None and os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = ensure_uint8_gray(image)

    gt_mask = read_binary_mask(gt_mask_path)

    if gt_mask is None:
        print("ERROR: Could not read GT mask:", gt_mask_path)
        return None

    pred_mask = read_binary_mask(
        pred_mask_path,
        target_shape=gt_mask.shape
    )

    if pred_mask is None:
        print("ERROR: Could not read predicted mask:", pred_mask_path)
        return None

    overlap_metrics = compute_overlap_metrics(
        pred_mask=pred_mask,
        gt_mask=gt_mask
    )

    boundary_metrics = compute_boundary_metrics(
        pred_mask=pred_mask,
        gt_mask=gt_mask
    )

    shape_metrics = compute_shape_metrics(
        pred_mask=pred_mask,
        gt_mask=gt_mask
    )

    metrics = {}

    metrics["file_name"] = os.path.basename(image_path) if image_path else os.path.basename(pred_mask_path)
    metrics["region"] = region_name

    metrics.update(overlap_metrics)
    metrics.update(boundary_metrics)
    metrics.update(shape_metrics)

    if debug_path:
        debug = create_segmentation_debug_image(
            image=image,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            title_text=f"{region_name} | Dice={metrics['dice']:.3f} IoU={metrics['iou']:.3f}"
        )

        safe_imwrite(debug_path, debug)

    return metrics


# ============================================================
# Batch evaluation
# ============================================================

def summarize_metrics(results_df):
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns

    summary_rows = []

    for region in sorted(results_df["region"].unique()):
        region_df = results_df[results_df["region"] == region]

        row = {
            "region": region,
            "sample_count": len(region_df)
        }

        for col in numeric_cols:
            row[f"{col}_mean"] = float(region_df[col].mean())
            row[f"{col}_std"] = float(region_df[col].std())
            row[f"{col}_min"] = float(region_df[col].min())
            row[f"{col}_max"] = float(region_df[col].max())

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def run_segmentation_evaluation(
    image_dir,

    pred_pdl_mask_dir,
    gt_pdl_mask_dir,

    pred_root_mask_dir=None,
    gt_root_mask_dir=None,

    output_csv_path=None,
    summary_csv_path=None,
    debug_dir=None,

    evaluate_pdl=True,
    evaluate_root=True
):
    os.makedirs(debug_dir, exist_ok=True) if debug_dir else None

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    image_files = [
        f for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in valid_ext)
    ]

    image_files.sort()

    all_results = []

    for file_name in image_files:
        image_path = os.path.join(image_dir, file_name)
        name, ext = os.path.splitext(file_name)

        # --------------------------------------------------------
        # PDL evaluation
        # --------------------------------------------------------

        if evaluate_pdl:
            pred_pdl_path = find_mask_path(
                pred_pdl_mask_dir,
                file_name,
                prefixes=[
                    "dark_pdl_",
                    "pdl_",
                    "pred_pdl_"
                ]
            )

            gt_pdl_path = find_mask_path(
                gt_pdl_mask_dir,
                file_name,
                prefixes=[
                    "manual_pdl_",
                    "gt_pdl_",
                    "pdl_"
                ]
            )

            if pred_pdl_path is None:
                print(f"SKIPPED PDL: predicted mask missing for {file_name}")
            elif gt_pdl_path is None:
                print(f"SKIPPED PDL: ground-truth mask missing for {file_name}")
            else:
                debug_path = None

                if debug_dir:
                    debug_path = os.path.join(
                        debug_dir,
                        f"debug_pdl_eval_{name}{ext}"
                    )

                metrics = evaluate_mask_pair(
                    image_path=image_path,
                    pred_mask_path=pred_pdl_path,
                    gt_mask_path=gt_pdl_path,
                    region_name="PDL",
                    debug_path=debug_path
                )

                if metrics is not None:
                    all_results.append(metrics)

        # --------------------------------------------------------
        # Root evaluation
        # --------------------------------------------------------

        if evaluate_root and pred_root_mask_dir is not None and gt_root_mask_dir is not None:
            pred_root_path = find_mask_path(
                pred_root_mask_dir,
                file_name,
                prefixes=[
                    "polynomial_root_",
                    "root_",
                    "pred_root_"
                ]
            )

            gt_root_path = find_mask_path(
                gt_root_mask_dir,
                file_name,
                prefixes=[
                    "manual_root_",
                    "gt_root_",
                    "root_"
                ]
            )

            if pred_root_path is None:
                print(f"SKIPPED ROOT: predicted mask missing for {file_name}")
            elif gt_root_path is None:
                print(f"SKIPPED ROOT: ground-truth mask missing for {file_name}")
            else:
                debug_path = None

                if debug_dir:
                    debug_path = os.path.join(
                        debug_dir,
                        f"debug_root_eval_{name}{ext}"
                    )

                metrics = evaluate_mask_pair(
                    image_path=image_path,
                    pred_mask_path=pred_root_path,
                    gt_mask_path=gt_root_path,
                    region_name="ROOT",
                    debug_path=debug_path
                )

                if metrics is not None:
                    all_results.append(metrics)

    if len(all_results) == 0:
        print("No evaluation results generated.")
        return None

    results_df = pd.DataFrame(all_results)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        print("Saved detailed evaluation:", output_csv_path)

    summary_df = summarize_metrics(results_df)

    if summary_csv_path:
        os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False)
        print("Saved summary evaluation:", summary_csv_path)

    print("\nEvaluation summary:")
    print(
        summary_df[
            [
                "region",
                "sample_count",
                "dice_mean",
                "iou_mean",
                "precision_mean",
                "recall_sensitivity_mean",
                "hd95_px_mean",
                "average_surface_distance_px_mean"
            ]
        ]
    )

    return results_df, summary_df