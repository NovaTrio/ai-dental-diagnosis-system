import cv2
import os
import json


RAW_DIR = "data/abscess/raw/images"
ANNOTATION_DIR = "data/common/annotations/tooth_clicks"

os.makedirs(ANNOTATION_DIR, exist_ok=True)

points = []


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)} selected: ({x}, {y})")


def annotate_image(image_path):
    global points
    points = []

    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    display = image.copy()
    image_name = os.path.basename(image_path)

    cv2.namedWindow("Select Tooth Orientation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Tooth Orientation", mouse_callback)

    while True:
        temp = display.copy()

        cv2.putText(
            temp,
            "Click 1: selected tooth center | Click 2: tooth vertical direction | Press S to save | R reset | ESC skip",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        if len(points) >= 1:
            cv2.circle(temp, points[0], 6, (0, 0, 255), -1)
            cv2.putText(temp, "Selected tooth", (points[0][0] + 10, points[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if len(points) >= 2:
            cv2.circle(temp, points[1], 6, (255, 0, 0), -1)
            cv2.line(temp, points[0], points[1], (0, 255, 0), 2)
            cv2.putText(temp, "Vertical direction", (points[1][0] + 10, points[1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Select Tooth Orientation", temp)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            print(f"Skipped: {image_name}")
            break

        elif key == ord("r"):
            points = []
            print("Reset points")

        elif key == ord("s"):
            if len(points) != 2:
                print("Please select 2 points before saving.")
                continue

            annotation = {
                "image_name": image_name,
                "selected_x": int(points[0][0]),
                "selected_y": int(points[0][1]),
                "direction_x": int(points[1][0]),
                "direction_y": int(points[1][1])
            }

            save_path = os.path.join(
                ANNOTATION_DIR,
                os.path.splitext(image_name)[0] + ".json"
            )

            with open(save_path, "w") as f:
                json.dump(annotation, f, indent=4)

            print(f"Saved annotation: {save_path}")
            break

    cv2.destroyAllWindows()


def run():
    image_files = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for image_file in image_files:
        image_path = os.path.join(RAW_DIR, image_file)
        print(f"\nAnnotating: {image_file}")
        annotate_image(image_path)


if __name__ == "__main__":
    run()