import os
import cv2
from matplotlib import pyplot as plt

from base_preprocess import base_preprocess

INPUT_DIR = "data/working_length/raw/images"

image_files = sorted(os.listdir(INPUT_DIR))

for file in image_files:
    path = os.path.join(INPUT_DIR, file)

    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = base_preprocess(path)

    # Resize original for display (optional)
    original_disp = cv2.resize(original, (400, 400))
    processed_disp = cv2.resize((processed * 255).astype("uint8"), (400, 400))

    # Combine side by side
    combined = cv2.hconcat([original_disp, processed_disp])

    cv2.imshow("Original (Left) | Processed (Right)", combined)

    print(f"Showing: {file}")
    key = cv2.waitKey(0)

    if key == 27:  # ESC key to exit early
        break
    
    
    print(f"Original min/max: {original.min()} / {original.max()}")
    print(f"Processed min/max: {processed.min():.2f} / {processed.max():.2f}")
    
    # plt.hist(processed.ravel(), bins=50)
    # plt.title(file)
    # plt.show()
cv2.destroyAllWindows()