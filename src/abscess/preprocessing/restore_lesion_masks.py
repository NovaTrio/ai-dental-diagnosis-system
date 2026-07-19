import os
import cv2
import numpy as np

def restore_masks(
    masks_dir,
    crops_dir,
    black_remove_dir,
    output_dir
):
    """
    Restores segmented lesion masks from crown-cropped images back to their 
    original BlackRemove image size. 
    Uses template matching to automatically find the crop coordinates, 
    so no CSV file is needed!
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg'}
    
    if not os.path.exists(masks_dir):
        print(f"[ERROR] Masks directory not found: {masks_dir}")
        return
        
    mask_files = [f for f in os.listdir(masks_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not mask_files:
        print(f"[WARNING] No valid mask images found in {masks_dir}")
        return

    # Process all PNG, JPG, and JPEG masks
    for filename in mask_files:
        # If there are overlay images in the folder, skip them. We only want masks.
        if filename.startswith('overlay_'):
            continue
            
        mask_path = os.path.join(masks_dir, filename)
        
        # Get the original base filename by removing the prefix 'lesion_mask_'
        base_name = filename
        if base_name.startswith('lesion_mask_'):
            base_name = base_name.replace('lesion_mask_', '')
            
        crop_path = os.path.join(crops_dir, base_name)
        black_remove_path = os.path.join(black_remove_dir, base_name)
        
        # Check if original BlackRemove image exists
        if not os.path.exists(black_remove_path):
            print(f"[WARNING] Original BlackRemove image not found: {black_remove_path}. Skipping.")
            continue
            
        # Check if crop image exists
        if not os.path.exists(crop_path):
            print(f"[WARNING] Crop image not found: {crop_path}. Skipping.")
            continue
            
        # Read the segmented lesion mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARNING] Could not read mask: {mask_path}. Skipping.")
            continue
            
        # Ensure mask is strictly binary 0 and 255
        mask = (mask > 127).astype(np.uint8) * 255
            
        # Read original image and cropped image for template matching
        orig_gray = cv2.imread(black_remove_path, cv2.IMREAD_GRAYSCALE)
        crop_gray = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        
        if orig_gray is None or crop_gray is None:
            print(f"[WARNING] Could not read original or crop image for {filename}. Skipping.")
            continue
            
        # Use template matching to find EXACTLY where the crop came from
        res = cv2.matchTemplate(orig_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        
        x1, y1 = max_loc
        crop_h, crop_w = crop_gray.shape
        x2, y2 = x1 + crop_w, y1 + crop_h
        
        # Read original image to get its exact canvas dimensions
        orig_img = cv2.imread(black_remove_path, cv2.IMREAD_UNCHANGED)
        orig_h, orig_w = orig_img.shape[:2]
        
        # Create a black canvas with the same height and width as the matching BlackRemove image
        restored_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # If the segmented mask size is different from the crop region size, resize it
        if mask.shape[1] != crop_w or mask.shape[0] != crop_h:
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            
        # Paste the segmented lesion mask into the original crop position
        restored_mask[y1:y2, x1:x2] = mask
        
        # Save the restored mask
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, restored_mask)
        print(f"[INFO] Restored mask saved successfully: {filename} at position (x:{x1}, y:{y1})")

if __name__ == "__main__":
    # Define paths relative to the project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    
    # Updated input folder to "lesion_postprocessed" as requested
    MASKS_DIR = os.path.join(PROJECT_ROOT, "data", "abscess", "raw", "lesion_postprocessed")
    CROPS_DIR = os.path.join(PROJECT_ROOT, "data", "abscess", "raw", "croun_crops")
    BLACK_REMOVE_DIR = os.path.join(PROJECT_ROOT, "data", "abscess", "raw", "BlackRemove")
    
    # Output folder for restored postprocessed masks
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "abscess", "raw", "lesion_postprocessed_restored")
    
    print("Starting lesion mask restoration...")
    restore_masks(
        masks_dir=MASKS_DIR,
        crops_dir=CROPS_DIR,
        black_remove_dir=BLACK_REMOVE_DIR,
        output_dir=OUTPUT_DIR
    )
    print("Done.")
