import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import csv
import argparse
from tqdm import tqdm

def process_image(image_path, model, density_factor, save_debug=False, debug_dir="debug_output"):
    """
    Process a single image to estimate fruit weight.
    Returns a dictionary with results or None if failed.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None

    # 1. Classification
    results = model(img, verbose=False) # verbose=False to reduce noise
    class_name = "Unknown"
    confidence = 0.0
    
    if results:
        top1 = results[0].probs.top1
        class_name = results[0].names[top1]
        confidence = results[0].probs.top1conf.item()

    # 2. Image Processing for Area
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range 1: Dark colors (Black/Very dark purple)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    
    # Range 2: Purple
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])
    
    # Create masks
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Combine masks
    fruit_mask = cv2.bitwise_or(mask_black, mask_purple)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_cnt = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < 0.2: continue 
            
        if area > max_area:
            max_area = area
            best_cnt = cnt
            
    estimated_weight = max_area * density_factor

    if best_cnt is not None:
        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            
            # Draw contour
            debug_img = img.copy()
            cv2.drawContours(debug_img, [best_cnt], -1, (0, 255, 0), 2)
            label = f"{class_name}: {estimated_weight:.1f}g"
            cv2.putText(debug_img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imwrite(os.path.join(debug_dir, f"debug_{filename}"), debug_img)
            cv2.imwrite(os.path.join(debug_dir, f"mask_{filename}"), fruit_mask)

    return {
        "filename": os.path.basename(image_path),
        "path": image_path,
        "class": class_name,
        "confidence": confidence,
        "area_pixels": max_area,
        "weight_grams": estimated_weight,
        "density_factor": density_factor
    }

def main():
    parser = argparse.ArgumentParser(description="Estimate fruit weight from images.")
    parser.add_argument("input_path", help="Path to an image file or a directory of images.")
    parser.add_argument("--model", default='/home/shooxd/Documentos/model-train/runs/classify/runs_zarzamora/modelo_v8/weights/best.pt', help="Path to YOLO model.")
    parser.add_argument("--factor", type=float, default=0.0012, help="Density factor (grams/pixel).")
    parser.add_argument("--output", default="weight_estimates.csv", help="Output CSV file for batch processing.")
    parser.add_argument("--debug", action="store_true", help="Save debug images.")
    
    args = parser.parse_args()

    # Load Model
    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Process Input
    results = []
    
    if os.path.isdir(args.input_path):
        print(f"Processing directory: {args.input_path}")
        image_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for img_file in tqdm(image_files, desc="Processing Images"):
            res = process_image(img_file, model, args.factor, args.debug)
            if res:
                results.append(res)
    else:
        print(f"Processing single file: {args.input_path}")
        res = process_image(args.input_path, model, args.factor, args.debug)
        if res:
            results.append(res)
            print(f"Result: {res['class']} - {res['weight_grams']:.2f}g")

    # Save to CSV
    if results:
        print(f"\nSaving results to {args.output}...")
        keys = results[0].keys()
        with open(args.output, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print("Done.")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
