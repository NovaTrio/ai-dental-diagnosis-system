import os
import cv2
import numpy as np
import math

def get_tooth_midpoints_and_axis(image, middle_region_ratio=0.5):
    """
    Find the row-wise midpoints of a tooth in a BlackRemove image (alpha channel or non-black pixels)
    and fit a robust straight line to estimate its longitudinal axis.
    
    Parameters:
    - image: BGR or BGRA numpy array
    - middle_region_ratio: What fraction of the middle height to use for line fitting (default 0.5 means middle 50%)
    
    Returns:
    - midpoints: list of (x, y) tuples
    - line_params: (vx, vy, x0, y0) from cv2.fitLine
    - angle_degrees: Angle of the axis from vertical
    """
    # Check if image has alpha channel
    if len(image.shape) == 3 and image.shape[2] == 4:
        mask = image[:, :, 3]
    else:
        # Fallback for RGB/BGR images: threshold the non-black regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
    h, w = mask.shape
    midpoints = []
    
    # 1. Calculate row-wise midpoints (root boundary pairs)
    for y in range(h):
        row = mask[y, :]
        non_zero_indices = np.where(row > 0)[0]
        if len(non_zero_indices) > 0:
            left_bound = non_zero_indices[0]
            right_bound = non_zero_indices[-1]
            midpoint = (left_bound + right_bound) / 2.0
            midpoints.append((int(midpoint), y))
            
    if not midpoints:
        return None, None, None
        
    pts = np.array(midpoints, dtype=np.int32)
    
    # 2. Filter points to only include the middle region (middle-root region)
    # This ignores the crown (top) and apex (bottom) which can skew the axis
    if middle_region_ratio < 1.0:
        y_min = int(h * (1 - middle_region_ratio) / 2)
        y_max = int(h * (1 + middle_region_ratio) / 2)
        middle_pts = [p for p in pts if y_min <= p[1] <= y_max]
        if len(middle_pts) > 10: # Ensure we have enough points
            pts_for_fitting = np.array(middle_pts, dtype=np.int32)
        else:
            pts_for_fitting = pts
    else:
        pts_for_fitting = pts
        
    # 3. Fit a robust straight line (using Huber loss)
    # cv2.DIST_HUBER is robust to outliers, behaving like L2 near zero and L1 far away.
    line = cv2.fitLine(pts_for_fitting, cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line[0][0], line[1][0], line[2][0], line[3][0]
    
    # Calculate angle in degrees from vertical (0 is straight up/down)
    angle_rad = math.atan2(vx, vy)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize angle to be between -90 and 90 (relative to vertical)
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
        
    return midpoints, (vx, vy, x0, y0), angle_deg

def draw_axis(image, midpoints, line_params):
    """Draw the midpoints and the fitted longitudinal axis on the image."""
    out_img = image.copy()
    if len(out_img.shape) == 3 and out_img.shape[2] == 4:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGRA2BGR)
        
    h, w = out_img.shape[:2]
    
    # Draw midpoints
    for x, y in midpoints:
        cv2.circle(out_img, (x, y), 1, (0, 0, 255), -1)
        
    # Draw line
    if line_params is not None:
        vx, vy, x0, y0 = line_params
        # Avoid division by zero
        if abs(vy) > 1e-6:
            y1 = 0
            x1 = int(x0 + vx*(y1 - y0)/vy)
            y2 = h - 1
            x2 = int(x0 + vx*(y2 - y0)/vy)
        else:
            x1 = int(x0)
            y1 = 0
            x2 = int(x0)
            y2 = h - 1
            
        cv2.line(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    return out_img

def main():
    # Setup paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    in_dir = os.path.join(project_root, "data", "abscess", "raw", "BlackRemove")
    out_dir = os.path.join(project_root, "data", "abscess", "raw", "ToothAxis")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if not os.path.exists(in_dir):
        print(f"Input directory does not exist: {in_dir}")
        return
        
    print("=" * 60)
    print("ESTIMATING TOOTH AXIS FROM BlackRemove IMAGES")
    print("=" * 60)
    
    success = 0
    for filename in os.listdir(in_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(in_dir, filename)
            
            # Read image with alpha channel
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Could not read {filename}")
                continue
                
            midpoints, line_params, angle = get_tooth_midpoints_and_axis(img, middle_region_ratio=0.6)
            
            if midpoints and line_params:
                out_img = draw_axis(img, midpoints, line_params)
                
                # Add angle text
                cv2.putText(out_img, f"Axis Angle: {angle:.2f} deg", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                out_path = os.path.join(out_dir, f"axis_{filename}")
                cv2.imwrite(out_path, out_img)
                print(f"[OK] Processed {filename} | Angle: {angle:.2f} deg")
                success += 1
            else:
                print(f"[FAIL] Failed to find axis for {filename}")
                
    print(f"\nCompleted: {success} images processed.")

if __name__ == "__main__":
    main()
