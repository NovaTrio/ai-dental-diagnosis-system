import os
import cv2
import numpy as np


input_dir = '../../../data/abscess/raw/images'
output_dir = '../../../data/abscess/raw/BlackRemove'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    

def remove_black_background_threshold(image_path, output_path, threshold=50):
    """Remove black background and add alpha channel"""
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Could not read {image_path}")
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([threshold, threshold, threshold])
    mask = cv2.inRange(img_rgb, lower_black, upper_black)
    mask_inv = cv2.bitwise_not(mask)
    
    b, g, r = cv2.split(img_rgb)
    rgba = cv2.merge([b, g, r, mask_inv])
    
    cv2.imwrite(output_path, rgba)
    return True

def sigmoid_enhancement(image, k=12, mid=0.38):
    """Sigmoid transformation for contrast enhancement"""
    has_alpha = False
    alpha_channel = None
    
    if len(image.shape) == 3 and image.shape[2] == 4:
        has_alpha = True
        b, g, r, a = cv2.split(image)
        alpha_channel = a
        img_rgb = cv2.merge([b, g, r])
    else:
        img_rgb = image.copy()
    
    img_float = img_rgb.astype(np.float32) / 255.0
    sigmoid_img = 1.0 / (1.0 + np.exp(-k * (img_float - mid)))
    sigmoid_img = np.clip(sigmoid_img, 0, 1)
    sigmoid_img = (sigmoid_img * 255).astype(np.uint8)
    result = cv2.cvtColor(sigmoid_img, cv2.COLOR_RGB2BGR)
    
    if has_alpha:
        result = cv2.merge([result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha_channel])
    
    return result

def clahe_enhancement(image, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE for local contrast enhancement"""
    has_alpha = False
    alpha_channel = None
    
    if len(image.shape) == 3 and image.shape[2] == 4:
        has_alpha = True
        b, g, r, a = cv2.split(image)
        alpha_channel = a
        img_rgb = cv2.merge([b, g, r])
    else:
        img_rgb = image.copy()
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if has_alpha:
        enhanced = cv2.merge([enhanced[:, :, 0], enhanced[:, :, 1], enhanced[:, :, 2], alpha_channel])
    
    return enhanced

def clahe_sigmoid_combined(image):
    """
    Apply CLAHE first, then Sigmoid for maximum lesion visibility
    """
    # First apply CLAHE
    clahe_result = clahe_enhancement(image, clipLimit=2.0, tileGridSize=(8, 8))
    
    # Then apply sigmoid on CLAHE result
    final_result = sigmoid_enhancement(clahe_result, k=12, mid=0.38)
    
    return final_result

# ============================================
# Process images - Output ONLY CLAHE+Sigmoid
# ============================================
print("=" * 60)
print("PROCESSING IMAGES - CLAHE + SIGMOID COMBINED")
print("=" * 60)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        base_name = filename.rsplit('.', 1)[0]
        
        # Step 1: Remove black background (temporary)
        temp_path = os.path.join(output_dir, '_temp_.png')
        remove_black_background_threshold(input_path, temp_path, threshold=50)
        
        # Step 2: Read image with alpha channel
        img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
        
        if img is not None:
            # Step 3: Apply CLAHE + Sigmoid combined enhancement
            enhanced_img = clahe_sigmoid_combined(img)
            
            # Step 4: Save final output
            output_path = os.path.join(output_dir, f"{base_name}_clahe_sigmoid.png")
            cv2.imwrite(output_path, enhanced_img)
            
            # Step 5: Remove temporary file
            os.remove(temp_path)
            
            print(f"✓ Created: {base_name}_clahe_sigmoid.png")
        else:
            print(f"✗ Failed: {filename}")

print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
print("\n Output files in 'output_images' folder:")
print("    [filename]_clahe_sigmoid.png - CLAHE + Sigmoid combined")
print("\n This combination is optimized for lesion visibility.")
print("   CLAHE enhances local contrast, then Sigmoid enhances overall contrast.")
print("=" * 60)