import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import random

def test_multiple_images():
    """Test multiple images"""
    
    # Hardcoded paths - change to the newly trained model
    best_model_path = Path("W:/hiwi/Fine-tuning icon_detect/icon_detect/model.pt")
    image_dir = Path("W:/hiwi/Fine-tuning icon_detect")
    results_dir = Path("test_results_old")
    
    # List of images to test
    test_images = [f"image{i}.png" for i in range(1, 11)]  # image1.png to image5.png
    
    # Check model file
    if not best_model_path.exists():
        print(f"❌ Model file does not exist: {best_model_path}")
        return
    print(f"✅ Using model: {best_model_path}")
    
    # Load model
    print("🔄 Loading model...")
    model = YOLO(best_model_path)
    
    # Create results directory
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Starting detection for {len(test_images)} images...")
    print("="*70)
    
    # Process each image
    for img_idx, image_name in enumerate(test_images, 1):
        test_image_path = image_dir / image_name
        
        print(f"\n📷 [{img_idx}/{len(test_images)}] Processing: {image_name}")
        print("-"*70)
        
        # Check if image exists
        if not test_image_path.exists():
            print(f"⚠️  Image does not exist: {test_image_path}")
            print(f"   Skipping {image_name}...")
            continue
        
        try:
            # Predict - lower confidence threshold to detect more objects
            results = model(test_image_path, conf=0.5, save=False, verbose=False)
            result = results[0]
            
            # Get detection results
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                detections = len(boxes)
                confidences = boxes.conf.cpu().numpy()
                max_conf = float(np.max(confidences))
                avg_conf = float(np.mean(confidences))
                
                print(f"🎯 Detected {detections} objects!")
                print(f"📊 Confidence range: {np.min(confidences):.3f} - {max_conf:.3f}")
                print(f"📊 Average confidence: {avg_conf:.3f}")
                
                # Read original image
                img = cv2.imread(str(test_image_path))
                img_height, img_width = img.shape[:2]
                
                # Define multiple colors (BGR format)
                colors = [
                    (255, 0, 0),    # Red
                    (0, 255, 0),    # Green
                    (0, 0, 255),    # Blue
                    (255, 255, 0),  # Cyan
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Yellow
                    (128, 0, 128),  # Purple
                    (255, 165, 0),  # Orange
                    (0, 128, 255),  # Dark Blue
                    (128, 255, 0),  # Spring Green
                ]
                
                # Draw detection boxes and IDs
                for j, (box, conf) in enumerate(zip(boxes.xyxy.cpu().numpy(), confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Choose color (cycle through)
                    color = colors[j % len(colors)]
                    
                    # Draw thin rectangle (thickness=1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
                    
                    # Prepare label text: ID 
                    label = f"{j+1}"
                    
                    # Calculate text size
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Text background box position
                    text_x = x1
                    text_y = y1 - 5 if y1 - 5 > text_height else y1 + text_height + 5
                    
                    # Draw text background (semi-transparent)
                    overlay = img.copy()
                    cv2.rectangle(overlay, 
                                (text_x, text_y - text_height - 3), 
                                (text_x + text_width + 6, text_y + 3), 
                                color, -1)
                    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                    
                    # Draw white text
                    cv2.putText(img, label, (text_x + 3, text_y), 
                               font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                # Save annotated image with unique name
                image_basename = test_image_path.stem  # e.g., "image1"
                output_path = results_dir / f"{image_basename}_detection_result.png"
                cv2.imwrite(str(output_path), img)
                print(f"💾 Detection result image: {output_path}")
                
                # Print detailed detection results
                print(f"\n📋 Detected icon details:")
                print("-" * 70)
                print(f"{'ID':>3} {'Center Position':>15} {'Size':>12} {'Confidence':>8}")
                print("-" * 70)
                for j, (box, conf) in enumerate(zip(boxes.xyxy.cpu().numpy(), confidences)):
                    x1, y1, x2, y2 = box
                    w, h = x2-x1, y2-y1
                    center_x, center_y = (x1+x2)/2, (y1+y2)/2
                    print(f"{j+1:3d} ({center_x:6.1f},{center_y:6.1f}) ({w:5.1f}×{h:4.1f}) {conf:7.3f}")
                print("-" * 70)

                # Save detection results to file with unique name
                txt_path = results_dir / f"{image_basename}_detection_results.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("Building Icon Detection Results\n")
                    f.write("="*30 + "\n\n")
                    f.write(f"Image: {test_image_path.name}\n")
                    f.write(f"Image Size: {img_width}×{img_height}\n")
                    f.write(f"Number of Detected Icons: {detections}\n")
                    f.write(f"Highest Confidence: {max_conf:.3f}\n")
                    f.write(f"Average Confidence: {avg_conf:.3f}\n\n")
                    f.write("Detailed Detection Results:\n")
                    f.write(f"{'ID':>3} {'Center Position':>15} {'Size':>12} {'Confidence':>8}\n")
                    f.write("-" * 50 + "\n")
                    for j, (box, conf) in enumerate(zip(boxes.xyxy.cpu().numpy(), confidences)):
                        x1, y1, x2, y2 = box
                        w, h = x2-x1, y2-y1
                        center_x, center_y = (x1+x2)/2, (y1+y2)/2
                        f.write(f"{j+1:3d} ({center_x:6.1f},{center_y:6.1f}) ({w:5.1f}×{h:4.1f}) {conf:7.3f}\n")
                
                print(f"\n📄 Detailed results file: {txt_path}")
                
                # Classify and display results by confidence
                high_conf = [(i, c) for i, c in enumerate(confidences) if c >= 0.5]
                med_conf = [(i, c) for i, c in enumerate(confidences) if 0.25 <= c < 0.5]
                low_conf = [(i, c) for i, c in enumerate(confidences) if c < 0.25]
                
                print(f"\n📈 Detection Quality Analysis:")
                print(f"  High Confidence (≥0.5): {len(high_conf)} - IDs: {[i+1 for i, c in high_conf]}")
                print(f"  Medium Confidence (0.25-0.5): {len(med_conf)} - IDs: {[i+1 for i, c in med_conf]}") 
                print(f"  Low Confidence (<0.25): {len(low_conf)} - IDs: {[i+1 for i, c in low_conf]}")
                
            else:
                print("❌ No icons detected")
                print("💡 Possible reasons:")
                print("  - The image does not contain trained icon types")
                print("  - Confidence threshold is too high")
                print("  - The model needs more training")
                
                # Save original image
                img = cv2.imread(str(test_image_path))
                output_path = results_dir / "no_detection_result.png"
                cv2.imwrite(str(output_path), img)
                print(f"💾 Original image saved: {output_path}")
                
        except Exception as e:
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Detection completed!")
    print(f"📁 All results saved in: {results_dir}/")

if __name__ == "__main__":
    test_multiple_images()  # 修正: 调用正确的函数名