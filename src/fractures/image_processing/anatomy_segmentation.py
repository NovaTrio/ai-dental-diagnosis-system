import cv2
import numpy as np
import os

from src.fractures.image_processing.fracture_preprocess import fracture_specific_preprocess


def get_dark_border_component(img_uint8):
    """
    Detect the large dark background/soft-tissue region connected to image borders.
    This is more reliable than gum-line edge detection.
    """

    # Dynamic threshold for dark areas
    dark_threshold = min(60, int(np.percentile(img_uint8, 20)))

    dark_mask = (img_uint8 <= dark_threshold).astype(np.uint8) * 255

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, 8)

    h, w = img_uint8.shape

    best_label = None
    best_area = 0

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        touches_border = (
            x <= 2 or
            y <= 2 or
            x + cw >= w - 2 or
            y + ch >= h - 2
        )

        if touches_border and area > best_area:
            best_area = area
            best_label = i

    if best_label is None:
        return None

    border_component = (labels == best_label).astype(np.uint8) * 255

    return border_component


def detect_background_position(border_component):
    """
    Decide whether the black background is mainly at the top or bottom.
    """

    h, w = border_component.shape

    top_score = np.sum(border_component[: int(h * 0.15), :] > 0)
    bottom_score = np.sum(border_component[int(h * 0.85):, :] > 0)

    if top_score >= bottom_score:
        return "top"
    else:
        return "bottom"


def extract_boundary_curve(border_component, position):
    """
    Extract boundary curve of the border-connected black region.
    If black region is top: use lower boundary.
    If black region is bottom: use upper boundary.
    """

    h, w = border_component.shape
    boundary = np.full(w, np.nan)

    for x in range(w):
        ys = np.where(border_component[:, x] > 0)[0]

        if len(ys) == 0:
            continue

        if position == "top":
            boundary[x] = np.max(ys)
        else:
            boundary[x] = np.min(ys)

    # Fill missing boundary values using interpolation
    valid = ~np.isnan(boundary)

    if np.sum(valid) < 5:
        return None

    xs = np.arange(w)
    boundary = np.interp(xs, xs[valid], boundary[valid])

    # Smooth boundary
    boundary = cv2.GaussianBlur(
        boundary.reshape(1, -1).astype(np.float32),
        (31, 1),
        0
    ).flatten()

    return boundary.astype(np.int32)


def crop_using_background_boundary(img_uint8, boundary, position, margin=8):
    """
    Crop useful tooth/root/bone area based on detected black background boundary.
    """

    h, w = img_uint8.shape

    if position == "top":
        crop_y = int(np.percentile(boundary, 90)) + margin
        crop_y = max(0, min(crop_y, h - 1))
        cropped = img_uint8[crop_y:, :]
        crop_box = (0, crop_y, w, h)

    else:
        crop_y = int(np.percentile(boundary, 10)) - margin
        crop_y = max(1, min(crop_y, h))
        cropped = img_uint8[:crop_y, :]
        crop_box = (0, 0, w, crop_y)

    return cropped, crop_box


def create_debug_image(img_uint8, border_component, boundary, crop_box, position):
    """
    Create debug output showing:
    - detected boundary curve
    - crop region
    - dark border component
    """

    debug = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    h, w = img_uint8.shape

    # Draw boundary curve
    for x in range(w - 1):
        y1 = int(boundary[x])
        y2 = int(boundary[x + 1])
        cv2.line(debug, (x, y1), (x + 1, y2), (0, 0, 255), 2)

    # Draw crop box
    x1, y1, x2, y2 = crop_box
    cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)

    component_vis = cv2.cvtColor(border_component, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([debug, component_vis])

    return combined


def extract_anatomical_region(image_path, save_path=None, debug_path=None):
    """
    Main function:
    Original X-ray
    → fracture preprocessing
    → detect border-connected dark background
    → detect boundary
    → crop useful anatomical region
    """

    img = fracture_specific_preprocess(image_path)

    border_component = get_dark_border_component(img)

    if border_component is None:
        print(f"Warning: No border-connected dark region found for {image_path}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

        return img

    position = detect_background_position(border_component)

    boundary = extract_boundary_curve(border_component, position)

    if boundary is None:
        print(f"Warning: Boundary curve could not be estimated for {image_path}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

        return img

    cropped, crop_box = crop_using_background_boundary(
        img_uint8=img,
        boundary=boundary,
        position=position,
        margin=8
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cropped)

    if debug_path:
        debug_img = create_debug_image(
            img_uint8=img,
            border_component=border_component,
            boundary=boundary,
            crop_box=crop_box,
            position=position
        )

        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, debug_img)

    return cropped