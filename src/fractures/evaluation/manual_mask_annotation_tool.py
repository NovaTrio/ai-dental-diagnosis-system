import os
import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = "data/fractures/processed/anatomical_region"

# Choose which mask you want to create:
# "pdl"  or  "root"
ANNOTATION_TYPE = "root"

# Optional: start from your automatic mask and manually correct it.
# This is faster than drawing from zero.
START_FROM_AUTO_MASK = True

AUTO_PDL_MASK_DIR = "data/fractures/processed/dark_pdl_from_polynomial_root_masks"
AUTO_ROOT_MASK_DIR = "data/fractures/processed/polynomial_root_masks"

MANUAL_PDL_MASK_DIR = "data/fractures/manual/pdl_masks"
MANUAL_ROOT_MASK_DIR = "data/fractures/manual/root_masks"

DISPLAY_SCALE = 2.0
DEFAULT_BRUSH_RADIUS = 4


# ============================================================
# Utility functions
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


def save_mask(path, mask):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    mask = ((mask > 0).astype(np.uint8) * 255)

    cv2.imwrite(path, mask)

    print("Saved:", path)


def get_output_dir_and_prefix():
    if ANNOTATION_TYPE.lower() == "pdl":
        return MANUAL_PDL_MASK_DIR, "manual_pdl_"

    if ANNOTATION_TYPE.lower() == "root":
        return MANUAL_ROOT_MASK_DIR, "manual_root_"

    raise ValueError("ANNOTATION_TYPE must be 'pdl' or 'root'.")


def get_auto_mask_path(file_name):
    name, _ = os.path.splitext(file_name)

    if ANNOTATION_TYPE.lower() == "pdl":
        auto_dir = AUTO_PDL_MASK_DIR
        prefix = "dark_pdl_"
    else:
        auto_dir = AUTO_ROOT_MASK_DIR
        prefix = "polynomial_root_"

    # Your automatic masks may be saved using the same original extension.
    valid_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for ext in valid_ext:
        candidate = os.path.join(
            auto_dir,
            f"{prefix}{name}{ext}"
        )

        if os.path.exists(candidate):
            return candidate

    return None


def get_manual_mask_path(file_name):
    name, _ = os.path.splitext(file_name)

    output_dir, prefix = get_output_dir_and_prefix()

    # Always save manual masks as PNG.
    return os.path.join(
        output_dir,
        f"{prefix}{name}.png"
    )


def create_overlay(image, mask, brush_radius, mode_text, file_name):
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = base.copy()

    if ANNOTATION_TYPE.lower() == "pdl":
        color = np.array([0, 255, 255], dtype=np.uint8)  # yellow
    else:
        color = np.array([0, 0, 255], dtype=np.uint8)  # red

    idx = mask > 0

    overlay[idx] = (
        0.45 * overlay[idx] + 0.55 * color
    ).astype(np.uint8)

    text_lines = [
        f"{ANNOTATION_TYPE.upper()} manual mask: {file_name}",
        f"Mode: {mode_text}",
        f"Brush: {brush_radius}",
        "Left mouse: draw | Right mouse: erase",
        "s: save | n: save next | c: clear | r: reload | +/-: brush | q: quit"
    ]

    y = 22

    for line in text_lines:
        cv2.putText(
            overlay,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 20

    if DISPLAY_SCALE != 1.0:
        overlay = cv2.resize(
            overlay,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_NEAREST
        )

    return overlay


# ============================================================
# Annotation state
# ============================================================

class AnnotationState:
    def __init__(self):
        self.image = None
        self.mask = None
        self.file_name = None
        self.brush_radius = DEFAULT_BRUSH_RADIUS
        self.drawing = False
        self.erasing = False
        self.changed = False

    def draw_at(self, x_display, y_display):
        if self.mask is None:
            return

        x = int(round(x_display / DISPLAY_SCALE))
        y = int(round(y_display / DISPLAY_SCALE))

        h, w = self.mask.shape

        if x < 0 or x >= w or y < 0 or y >= h:
            return

        if self.drawing:
            cv2.circle(
                self.mask,
                (x, y),
                self.brush_radius,
                255,
                -1
            )
            self.changed = True

        if self.erasing:
            cv2.circle(
                self.mask,
                (x, y),
                self.brush_radius,
                0,
                -1
            )
            self.changed = True


state = AnnotationState()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.erasing = False
        state.draw_at(x, y)

    elif event == cv2.EVENT_RBUTTONDOWN:
        state.erasing = True
        state.drawing = False
        state.draw_at(x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing or state.erasing:
            state.draw_at(x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False

    elif event == cv2.EVENT_RBUTTONUP:
        state.erasing = False


# ============================================================
# Mask loading
# ============================================================

def load_mask_for_image(file_name, image_shape):
    manual_path = get_manual_mask_path(file_name)

    # If manual mask already exists, continue editing it.
    if os.path.exists(manual_path):
        mask = read_binary_mask(
            manual_path,
            target_shape=image_shape
        )

        if mask is not None:
            print("Loaded existing manual mask:", manual_path)
            return mask

    # Otherwise optionally start from automatic mask.
    if START_FROM_AUTO_MASK:
        auto_path = get_auto_mask_path(file_name)

        if auto_path is not None:
            mask = read_binary_mask(
                auto_path,
                target_shape=image_shape
            )

            if mask is not None:
                print("Loaded automatic mask as starting point:", auto_path)
                return mask

    # Otherwise start blank.
    return np.zeros(image_shape, dtype=np.uint8)


# ============================================================
# Main annotation loop
# ============================================================

def run():
    print("Manual mask annotation tool")
    print("ANNOTATION_TYPE:", ANNOTATION_TYPE)
    print("IMAGE_DIR:", IMAGE_DIR)

    if not os.path.exists(IMAGE_DIR):
        print("ERROR: IMAGE_DIR does not exist.")
        return

    output_dir, _ = get_output_dir_and_prefix()
    os.makedirs(output_dir, exist_ok=True)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    files = [
        f for f in os.listdir(IMAGE_DIR)
        if any(f.lower().endswith(ext) for ext in valid_ext)
    ]

    files.sort()

    if len(files) == 0:
        print("No images found.")
        return

    print("Total images:", len(files))

    window_name = "Manual Mask Annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    index = 0

    while index < len(files):
        file_name = files[index]
        image_path = os.path.join(IMAGE_DIR, file_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = ensure_uint8_gray(img)

        if img is None:
            print("Could not read image:", image_path)
            index += 1
            continue

        state.image = img
        state.file_name = file_name
        state.mask = load_mask_for_image(file_name, img.shape)
        state.changed = False

        print()
        print(f"[{index + 1}/{len(files)}] Editing:", file_name)

        while True:
            mode_text = "DRAW" if state.drawing else "ERASE" if state.erasing else "IDLE"

            display = create_overlay(
                image=state.image,
                mask=state.mask,
                brush_radius=state.brush_radius,
                mode_text=mode_text,
                file_name=file_name
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            # Save
            if key == ord("s"):
                manual_path = get_manual_mask_path(file_name)
                save_mask(manual_path, state.mask)
                state.changed = False

            # Save and next
            elif key == ord("n"):
                manual_path = get_manual_mask_path(file_name)
                save_mask(manual_path, state.mask)
                state.changed = False
                index += 1
                break

            # Previous image
            elif key == ord("p"):
                manual_path = get_manual_mask_path(file_name)
                save_mask(manual_path, state.mask)
                state.changed = False
                index = max(0, index - 1)
                break

            # Clear mask
            elif key == ord("c"):
                state.mask[:, :] = 0
                state.changed = True
                print("Mask cleared.")

            # Reload manual/auto mask
            elif key == ord("r"):
                state.mask = load_mask_for_image(file_name, img.shape)
                state.changed = False
                print("Mask reloaded.")

            # Increase brush
            elif key == ord("+") or key == ord("="):
                state.brush_radius += 1
                print("Brush radius:", state.brush_radius)

            # Decrease brush
            elif key == ord("-") or key == ord("_"):
                state.brush_radius = max(1, state.brush_radius - 1)
                print("Brush radius:", state.brush_radius)

            # Quit
            elif key == ord("q"):
                if state.changed:
                    manual_path = get_manual_mask_path(file_name)
                    save_mask(manual_path, state.mask)

                cv2.destroyAllWindows()
                print("Stopped annotation.")
                return

    cv2.destroyAllWindows()
    print("Finished all images.")


if __name__ == "__main__":
    run()