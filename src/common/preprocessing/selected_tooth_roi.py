import cv2
import numpy as np
import os
import json


RAW_DIR = "data/abscess/raw/images"
ANNOTATION_DIR = "data/common/annotations/tooth_clicks"

OUTPUT_DIR = "data/common/processed/common_selected_tooth_roi"
DEBUG_DIR = "data/common/processed/debug_common_selected_tooth_roi"


def smooth_1d(signal, ksize=31):
    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(
        signal.reshape(1, -1).astype(np.float32),
        (ksize, 1),
        0
    ).flatten()


def calculate_manual_rotation_angle(selected_x, selected_y,
                                    direction_x, direction_y):

    dx = direction_x - selected_x
    dy = direction_y - selected_y

    if abs(dx) < 1 and abs(dy) < 1:
        return 0.0

    angle = np.degrees(np.arctan2(-dx, dy))

    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    # Flip tooth so root side points upward
    if selected_y > direction_y:
        angle += 180

    return angle


def rotate_image_and_points(image, points, angle_degrees):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    rotated_points = []

    for x, y in points:
        px = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        py = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        rotated_points.append((int(px), int(py)))

    return rotated, rotated_points


def find_black_gap_boundaries(gray, click_x, fixed_width_ratio=0.32):
    h, w = gray.shape
    cx = int(np.clip(click_x, 0, w - 1))

    roi_width = int(fixed_width_ratio * w)

    x1 = cx - roi_width // 2
    x2 = cx + roi_width // 2

    if x1 < 0:
        x2 += abs(x1)
        x1 = 0

    if x2 > w:
        x1 -= (x2 - w)
        x2 = w

    x1 = max(0, x1)
    x2 = min(w, x2)

    # only for debug drawing
    left_gap = x1
    right_gap = x2

    gap_score = np.zeros(w, dtype=np.float32)
    gap_score[x1:x2] = 1.0

    threshold = 0

    return (x1, 0, x2, h), (left_gap, right_gap), gap_score, threshold


def create_debug_image(
    original,
    rotated,
    click_point,
    direction_point,
    rotated_click_point,
    rotated_direction_point,
    crop_box,
    gap_boundaries,
    gap_score,
    threshold,
    angle
):
    debug_original = original.copy()
    debug_rotated = rotated.copy()

    h, w = rotated.shape[:2]

    click_x, click_y = click_point
    direction_x, direction_y = direction_point

    rotated_click_x, rotated_click_y = rotated_click_point
    rotated_direction_x, rotated_direction_y = rotated_direction_point

    x1, y1, x2, y2 = crop_box
    left_gap, right_gap = gap_boundaries

    cv2.circle(debug_original, (click_x, click_y), 6, (0, 0, 255), -1)
    cv2.circle(debug_original, (direction_x, direction_y), 6, (255, 0, 0), -1)
    cv2.line(debug_original, (click_x, click_y), (direction_x, direction_y), (0, 255, 0), 2)

    cv2.putText(
        debug_original,
        f"Manual angle: {angle:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.circle(debug_rotated, (rotated_click_x, rotated_click_y), 6, (0, 0, 255), -1)
    cv2.circle(debug_rotated, (rotated_direction_x, rotated_direction_y), 6, (255, 0, 0), -1)

    cv2.line(
        debug_rotated,
        (rotated_click_x, rotated_click_y),
        (rotated_direction_x, rotated_direction_y),
        (0, 255, 0),
        2
    )

    cv2.line(debug_rotated, (left_gap, 0), (left_gap, h), (0, 0, 255), 2)
    cv2.line(debug_rotated, (right_gap, 0), (right_gap, h), (0, 0, 255), 2)

    cv2.rectangle(debug_rotated, (x1, y1), (x2, y2 - 1), (0, 255, 0), 3)

    cv2.putText(
        debug_rotated,
        f"Dark threshold: {threshold}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    projection = np.zeros((h, w, 3), dtype=np.uint8)

    score_scaled = (
        gap_score / (gap_score.max() + 1e-6) * (h - 1)
    ).astype(np.int32)

    for x in range(min(len(score_scaled) - 1, w - 1)):
        py1 = h - 1 - score_scaled[x]
        py2 = h - 1 - score_scaled[x + 1]
        cv2.line(projection, (x, py1), (x + 1, py2), (255, 255, 255), 1)

    roi = rotated[y1:y2, x1:x2]

    target_h = 320

    def resize_h(img):
        scale = target_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (new_w, target_h))

    combined = np.hstack([
        resize_h(debug_original),
        resize_h(debug_rotated),
        resize_h(projection),
        resize_h(roi)
    ])

    return combined


def extract_selected_tooth_roi(image_path, annotation_path, save_path, debug_path=None):
    original = cv2.imread(image_path)

    if original is None:
        raise ValueError(f"Image not found: {image_path}")

    with open(annotation_path, "r") as f:
        ann = json.load(f)

    click_x = int(ann["selected_x"])
    click_y = int(ann["selected_y"])

    direction_x = int(ann.get("direction_x", click_x))
    direction_y = int(ann.get("direction_y", click_y + 50))

    angle = calculate_manual_rotation_angle(
        click_x,
        click_y,
        direction_x,
        direction_y
    )

    rotated, rotated_points = rotate_image_and_points(
        original,
        [(click_x, click_y), (direction_x, direction_y)],
        angle
    )

    rotated_click_x, rotated_click_y = rotated_points[0]
    rotated_direction_x, rotated_direction_y = rotated_points[1]

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    crop_box, gap_boundaries, gap_score, threshold = find_black_gap_boundaries(
        gray,
        rotated_click_x
    )

    x1, y1, x2, y2 = crop_box
    roi = rotated[y1:y2, x1:x2]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, roi)

    if debug_path:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        debug_img = create_debug_image(
            original=original,
            rotated=rotated,
            click_point=(click_x, click_y),
            direction_point=(direction_x, direction_y),
            rotated_click_point=(rotated_click_x, rotated_click_y),
            rotated_direction_point=(rotated_direction_x, rotated_direction_y),
            crop_box=crop_box,
            gap_boundaries=gap_boundaries,
            gap_score=gap_score,
            threshold=threshold,
            angle=angle
        )

        cv2.imwrite(debug_path, debug_img)

    return roi


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    annotations = [
        f for f in os.listdir(ANNOTATION_DIR)
        if f.endswith(".json")
    ]

    for ann_file in annotations:
        annotation_path = os.path.join(ANNOTATION_DIR, ann_file)

        with open(annotation_path, "r") as f:
            ann = json.load(f)

        image_name = ann["image_name"]
        image_path = os.path.join(RAW_DIR, image_name)

        output_path = os.path.join(OUTPUT_DIR, image_name)
        debug_path = os.path.join(DEBUG_DIR, image_name)

        try:
            extract_selected_tooth_roi(
                image_path=image_path,
                annotation_path=annotation_path,
                save_path=output_path,
                debug_path=debug_path
            )

            print(f"Processed selected tooth ROI: {image_name}")

        except Exception as e:
            print(f"Failed: {image_name} | Error: {e}")


if __name__ == "__main__":
    run()