import cv2
import os
import json


RAW_DIR = "data/fractures/raw/images"
ANNOTATION_DIR = "data/fractures/annotations/tooth_marks"


clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def mark_tooth(image_path, save_json_path):
    global clicked_point
    clicked_point = None

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    display = img.copy()

    window_name = "Click the RCT tooth center - Press S to save, Q to skip"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = display.copy()

        if clicked_point is not None:
            x, y = clicked_point
            cv2.circle(temp, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                temp,
                f"Selected: ({x}, {y})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(window_name, temp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if clicked_point is None:
                print("No point selected.")
                continue

            os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

            h, w = img.shape[:2]

            data = {
                "image_name": os.path.basename(image_path),
                "original_width": w,
                "original_height": h,
                "tooth_center_x": clicked_point[0],
                "tooth_center_y": clicked_point[1]
            }

            with open(save_json_path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Saved mark: {save_json_path}")
            break

        if key == ord("q"):
            print(f"Skipped: {image_path}")
            break

    cv2.destroyAllWindows()


def run():
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png"]

    files = os.listdir(RAW_DIR)

    for file_name in files:
        if not any(file_name.lower().endswith(ext) for ext in valid_ext):
            continue

        image_path = os.path.join(RAW_DIR, file_name)
        base_name = os.path.splitext(file_name)[0]
        save_json_path = os.path.join(ANNOTATION_DIR, base_name + ".json")

        if os.path.exists(save_json_path):
            print(f"Already marked: {file_name}")
            continue

        print(f"Marking: {file_name}")
        mark_tooth(image_path, save_json_path)


if __name__ == "__main__":
    run()