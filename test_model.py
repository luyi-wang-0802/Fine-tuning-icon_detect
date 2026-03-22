import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference on selected images.")
    parser.add_argument("--model", default="icon_detect/model.pt", help="Model checkpoint path.")
    parser.add_argument("--image-dir", default=".", help="Directory containing test images.")
    parser.add_argument("--output-dir", default="test_results_old", help="Directory for inference outputs.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument(
        "--images",
        nargs="*",
        default=[f"image{i}.png" for i in range(1, 11)],
        help="Image filenames to process.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def test_multiple_images() -> None:
    args = parse_args()

    best_model_path = resolve_path(args.model)
    image_dir = resolve_path(args.image_dir)
    results_dir = resolve_path(args.output_dir)
    test_images = args.images

    if not best_model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {best_model_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    print(f"Using model: {best_model_path}")
    model = YOLO(str(best_model_path))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting detection for {len(test_images)} images...")
    print("=" * 70)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 165, 0),
        (0, 128, 255),
        (128, 255, 0),
    ]

    for img_idx, image_name in enumerate(test_images, 1):
        test_image_path = image_dir / image_name

        print(f"\n[{img_idx}/{len(test_images)}] Processing: {image_name}")
        print("-" * 70)

        if not test_image_path.exists():
            print(f"Image does not exist: {test_image_path}")
            print(f"Skipping {image_name}...")
            continue

        try:
            results = model(str(test_image_path), conf=args.conf, save=False, verbose=False)
            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                print("No icons detected.")
                img = cv2.imread(str(test_image_path))
                if img is not None:
                    output_path = results_dir / f"{test_image_path.stem}_no_detection.png"
                    cv2.imwrite(str(output_path), img)
                    print(f"Saved original image to {output_path}")
                continue

            detections = len(boxes)
            confidences = boxes.conf.cpu().numpy()
            max_conf = float(np.max(confidences))
            avg_conf = float(np.mean(confidences))

            print(f"Detected {detections} objects")
            print(f"Confidence range: {np.min(confidences):.3f} - {max_conf:.3f}")
            print(f"Average confidence: {avg_conf:.3f}")

            img = cv2.imread(str(test_image_path))
            if img is None:
                print(f"Failed to read image: {test_image_path}")
                continue
            img_height, img_width = img.shape[:2]

            for j, box in enumerate(boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                color = colors[j % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

                label = f"{j + 1}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > text_height else y1 + text_height + 5

                overlay = img.copy()
                cv2.rectangle(
                    overlay,
                    (text_x, text_y - text_height - 3),
                    (text_x + text_width + 6, text_y + 3),
                    color,
                    -1,
                )
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                cv2.putText(img, label, (text_x + 3, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            output_path = results_dir / f"{test_image_path.stem}_detection_result.png"
            cv2.imwrite(str(output_path), img)
            print(f"Saved detection image: {output_path}")

            print("\nDetected icon details:")
            print("-" * 70)
            print(f"{'ID':>3} {'Center Position':>15} {'Size':>12} {'Confidence':>10}")
            print("-" * 70)

            txt_path = results_dir / f"{test_image_path.stem}_detection_results.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Icon Detection Results\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Image: {test_image_path.name}\n")
                f.write(f"Image Size: {img_width}x{img_height}\n")
                f.write(f"Number of Detected Icons: {detections}\n")
                f.write(f"Highest Confidence: {max_conf:.3f}\n")
                f.write(f"Average Confidence: {avg_conf:.3f}\n\n")
                f.write("Detailed Detection Results:\n")
                f.write(f"{'ID':>3} {'Center Position':>15} {'Size':>12} {'Confidence':>10}\n")
                f.write("-" * 55 + "\n")

                for j, (box, conf) in enumerate(zip(boxes.xyxy.cpu().numpy(), confidences)):
                    x1, y1, x2, y2 = box
                    width, height = x2 - x1, y2 - y1
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    line = f"{j + 1:3d} ({center_x:6.1f},{center_y:6.1f}) ({width:5.1f}x{height:4.1f}) {conf:9.3f}"
                    print(line)
                    f.write(line + "\n")

            print("-" * 70)
            print(f"Saved text report: {txt_path}")

        except Exception as exc:
            print(f"Detection error: {exc}")
            raise

    print(f"\nDetection completed. All results saved in: {results_dir}")


if __name__ == "__main__":
    test_multiple_images()
