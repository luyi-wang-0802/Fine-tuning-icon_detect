import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


ROOT = Path(__file__).parent


@dataclass
class EvalConfig:
    before_model_path: Path
    after_model_path: Path
    test_images_dir: Path
    test_labels_dir: Path
    output_dir: Path
    conf_thresholds: List[float]
    iou_match_thresholds: List[float]
    infer_conf_min: float
    nms_iou: float
    max_det: int
    save_error_examples: bool
    num_error_images: int
    draw_iou_match_threshold: float
    draw_conf_threshold: float


def calculate_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union = area1 + area2 - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def load_ground_truth_yolo(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    boxes = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                xc = float(parts[1]) * img_w
                yc = float(parts[2]) * img_h
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
                x1 = xc - bw / 2
                y1 = yc - bh / 2
                x2 = xc + bw / 2
                y2 = yc + bh / 2
                boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def match_predictions_to_gt(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_thr: float,
) -> Tuple[int, int, int, List[float], List[int], List[int]]:
    if pred_xyxy.size == 0 and gt_xyxy.size == 0:
        return 0, 0, 0, [], [], []
    if pred_xyxy.size == 0:
        return 0, 0, int(len(gt_xyxy)), [], [], list(range(len(gt_xyxy)))
    if gt_xyxy.size == 0:
        return 0, int(len(pred_xyxy)), 0, [], list(range(len(pred_xyxy))), []

    matched_gt = set()
    tp = 0
    matched_ious = []
    fp_pred_indices = []

    for p_i, p in enumerate(pred_xyxy):
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_xyxy):
            if j in matched_gt:
                continue
            iou = calculate_iou_xyxy(p, g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
            matched_ious.append(best_iou)
        else:
            fp_pred_indices.append(p_i)

    fn = int(len(gt_xyxy) - len(matched_gt))
    fn_gt_indices = [j for j in range(len(gt_xyxy)) if j not in matched_gt]
    fp = int(len(pred_xyxy) - tp)
    return tp, fp, fn, matched_ious, fp_pred_indices, fn_gt_indices


def calc_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_predictions_once(
    model: YOLO,
    image_files: List[Path],
    labels_dir: Path,
    infer_conf_min: float,
    nms_iou: float,
    max_det: int,
) -> Dict[str, Dict]:
    cache: Dict[str, Dict] = {}

    for img_path in tqdm(image_files, desc="Collecting predictions", ncols=100):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_ground_truth_yolo(labels_dir / f"{img_path.stem}.txt", w, h)

        res = model(str(img_path), conf=infer_conf_min, iou=nms_iou, max_det=max_det, verbose=False)[0]

        if res.boxes is not None and len(res.boxes) > 0:
            pred_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_conf = res.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            pred_xyxy = np.zeros((0, 4), dtype=np.float32)
            pred_conf = np.zeros((0,), dtype=np.float32)

        cache[img_path.stem] = {
            "img_path": img_path,
            "img_w": w,
            "img_h": h,
            "gt": gt,
            "pred_xyxy": pred_xyxy,
            "pred_conf": pred_conf,
        }

    return cache


def evaluate_from_cache(cache: Dict[str, Dict], conf_thr: float, iou_match_thr: float) -> Dict:
    tp_total = fp_total = fn_total = 0
    ious_all: List[float] = []
    num_preds_per_img = []
    num_gt_per_img = []

    for _, item in cache.items():
        gt = item["gt"]
        pred_xyxy = item["pred_xyxy"]
        pred_conf = item["pred_conf"]

        keep = pred_conf >= conf_thr
        filtered_pred = pred_xyxy[keep] if pred_xyxy.size else pred_xyxy

        tp, fp, fn, matched_ious, _, _ = match_predictions_to_gt(filtered_pred, gt, iou_match_thr)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        ious_all.extend(matched_ious)
        num_preds_per_img.append(int(len(filtered_pred)))
        num_gt_per_img.append(int(len(gt)))

    metrics = calc_metrics(tp_total, fp_total, fn_total)
    return {
        "conf_thr": float(conf_thr),
        "iou_match_thr": float(iou_match_thr),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mean_iou_of_tps": float(np.mean(ious_all)) if ious_all else 0.0,
        "avg_preds_per_img": float(np.mean(num_preds_per_img)) if num_preds_per_img else 0.0,
        "avg_gts_per_img": float(np.mean(num_gt_per_img)) if num_gt_per_img else 0.0,
    }


def sweep_eval(cache: Dict[str, Dict], conf_thresholds: List[float], iou_match_thresholds: List[float]) -> List[Dict]:
    rows = []
    for iou_thr in iou_match_thresholds:
        for conf_thr in conf_thresholds:
            rows.append(evaluate_from_cache(cache, conf_thr=conf_thr, iou_match_thr=iou_thr))
    return rows


def pick_operating_points(rows: List[Dict]) -> Dict[str, Dict]:
    best = {}
    by_iou: Dict[float, List[Dict]] = {}
    for row in rows:
        by_iou.setdefault(row["iou_match_thr"], []).append(row)

    for iou_thr, candidates in by_iou.items():
        best[f"best_f1_iou_{iou_thr}"] = max(candidates, key=lambda item: item["f1"])
    return best


def draw_boxes(img: np.ndarray, boxes: np.ndarray, color: Tuple[int, int, int], label: str) -> np.ndarray:
    out = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def save_error_examples(
    cache: Dict[str, Dict],
    out_dir: Path,
    conf_thr: float,
    iou_match_thr: float,
    num_images: int,
) -> None:
    ensure_dir(out_dir / "errors_fp")
    ensure_dir(out_dir / "errors_fn")
    ensure_dir(out_dir / "errors_both")

    fp_saved = fn_saved = both_saved = 0

    for _, item in cache.items():
        if fp_saved >= num_images and fn_saved >= num_images and both_saved >= num_images:
            break

        img_path: Path = item["img_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt = item["gt"]
        pred_xyxy = item["pred_xyxy"]
        pred_conf = item["pred_conf"]
        keep = pred_conf >= conf_thr
        filtered_pred = pred_xyxy[keep] if pred_xyxy.size else pred_xyxy

        _, _, _, _, fp_pred_idx, fn_gt_idx = match_predictions_to_gt(filtered_pred, gt, iou_match_thr)
        fp_boxes = filtered_pred[fp_pred_idx] if fp_pred_idx else np.zeros((0, 4), dtype=np.float32)
        fn_boxes = gt[fn_gt_idx] if fn_gt_idx else np.zeros((0, 4), dtype=np.float32)

        if fp_saved < num_images and len(fp_boxes) > 0:
            vis = draw_boxes(img, fp_boxes, (0, 0, 255), "FP")
            cv2.imwrite(str(out_dir / "errors_fp" / f"{img_path.stem}_fp.png"), vis)
            fp_saved += 1

        if fn_saved < num_images and len(fn_boxes) > 0:
            vis = draw_boxes(img, fn_boxes, (255, 0, 0), "FN")
            cv2.imwrite(str(out_dir / "errors_fn" / f"{img_path.stem}_fn.png"), vis)
            fn_saved += 1

        if both_saved < num_images and (len(fp_boxes) > 0 or len(fn_boxes) > 0):
            vis = img.copy()
            if len(fp_boxes) > 0:
                vis = draw_boxes(vis, fp_boxes, (0, 0, 255), "FP")
            if len(fn_boxes) > 0:
                vis = draw_boxes(vis, fn_boxes, (255, 0, 0), "FN")
            cv2.imwrite(str(out_dir / "errors_both" / f"{img_path.stem}_both.png"), vis)
            both_saved += 1


def plot_pr_curves(rows_before: List[Dict], rows_after: List[Dict], out_dir: Path) -> None:
    ensure_dir(out_dir)
    iou_values = sorted(set([r["iou_match_thr"] for r in rows_before] + [r["iou_match_thr"] for r in rows_after]))

    for iou_thr in iou_values:
        before_points = sorted(
            [row for row in rows_before if row["iou_match_thr"] == iou_thr],
            key=lambda item: item["recall"],
        )
        after_points = sorted(
            [row for row in rows_after if row["iou_match_thr"] == iou_thr],
            key=lambda item: item["recall"],
        )

        plt.figure(figsize=(7, 6))
        plt.plot([x["recall"] for x in before_points], [x["precision"] for x in before_points], marker="o", label=f"Before (IoU={iou_thr})")
        plt.plot([x["recall"] for x in after_points], [x["precision"] for x in after_points], marker="o", label=f"After (IoU={iou_thr})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve (Match IoU = {iou_thr})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"pr_curve_iou_{iou_thr}.png", dpi=200)
        plt.close()


def plot_f1_vs_conf(rows_before: List[Dict], rows_after: List[Dict], out_dir: Path) -> None:
    ensure_dir(out_dir)
    iou_values = sorted(set([r["iou_match_thr"] for r in rows_before] + [r["iou_match_thr"] for r in rows_after]))

    for iou_thr in iou_values:
        before_points = sorted(
            [row for row in rows_before if row["iou_match_thr"] == iou_thr],
            key=lambda item: item["conf_thr"],
        )
        after_points = sorted(
            [row for row in rows_after if row["iou_match_thr"] == iou_thr],
            key=lambda item: item["conf_thr"],
        )

        plt.figure(figsize=(7, 6))
        plt.plot([x["conf_thr"] for x in before_points], [x["f1"] for x in before_points], marker="o", label="Before")
        plt.plot([x["conf_thr"] for x in after_points], [x["f1"] for x in after_points], marker="o", label="After")
        plt.xlabel("Confidence threshold")
        plt.ylabel("F1")
        plt.title(f"F1 vs Conf (Match IoU = {iou_thr})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"f1_vs_conf_iou_{iou_thr}.png", dpi=200)
        plt.close()


def plot_conf_hist_from_cache(cache: Dict[str, Dict], out_path: Path, title: str) -> None:
    confs = []
    for _, item in cache.items():
        if item["pred_conf"] is not None and len(item["pred_conf"]) > 0:
            confs.extend(item["pred_conf"].tolist())

    plt.figure(figsize=(7, 6))
    if confs:
        plt.hist(confs, bins=50)
        plt.axvline(np.mean(confs), linestyle="--", linewidth=2, label=f"Mean: {np.mean(confs):.3f}")
        plt.legend()
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare detector quality before and after fine-tuning.")
    parser.add_argument("--before-model", default="icon_detect/model.pt", help="Checkpoint before fine-tuning.")
    parser.add_argument(
        "--after-model",
        default="runs/detect/icon_detect_new_dataset/weights/best.pt",
        help="Checkpoint after fine-tuning.",
    )
    parser.add_argument("--test-images", default="dataset_split/images/test", help="Directory containing test images.")
    parser.add_argument("--test-labels", default="dataset_split/labels/test", help="Directory containing test labels.")
    parser.add_argument("--output-dir", default="model_comparison_results_v2", help="Directory for comparison outputs.")
    parser.add_argument("--infer-conf-min", type=float, default=0.001, help="Low confidence used to collect raw predictions.")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU used during inference.")
    parser.add_argument("--max-det", type=int, default=500, help="Maximum detections per image.")
    parser.add_argument("--draw-conf-threshold", type=float, default=0.5, help="Confidence threshold for error visualizations.")
    parser.add_argument("--draw-iou-threshold", type=float, default=0.5, help="IoU threshold for error visualizations.")
    parser.add_argument("--num-error-images", type=int, default=50, help="Maximum saved error images per category.")
    parser.add_argument("--no-error-examples", action="store_true", help="Disable saving FP/FN example images.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()

    cfg = EvalConfig(
        before_model_path=resolve_path(args.before_model),
        after_model_path=resolve_path(args.after_model),
        test_images_dir=resolve_path(args.test_images),
        test_labels_dir=resolve_path(args.test_labels),
        output_dir=resolve_path(args.output_dir),
        conf_thresholds=[round(x, 3) for x in np.linspace(0.05, 0.95, 19).tolist()],
        iou_match_thresholds=[0.3, 0.5],
        infer_conf_min=args.infer_conf_min,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        save_error_examples=not args.no_error_examples,
        num_error_images=args.num_error_images,
        draw_iou_match_threshold=args.draw_iou_threshold,
        draw_conf_threshold=args.draw_conf_threshold,
    )

    for path, name in [
        (cfg.before_model_path, "Before model"),
        (cfg.after_model_path, "After model"),
        (cfg.test_images_dir, "Test images"),
        (cfg.test_labels_dir, "Test labels"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    ensure_dir(cfg.output_dir)
    with open(cfg.output_dir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}, f, indent=2)

    image_files = sorted(list(cfg.test_images_dir.glob("*.png")) + list(cfg.test_images_dir.glob("*.jpg")) + list(cfg.test_images_dir.glob("*.jpeg")))
    print(f"Found {len(image_files)} test images")

    before_model = YOLO(str(cfg.before_model_path))
    after_model = YOLO(str(cfg.after_model_path))

    print("\n[1/4] Collect BEFORE predictions...")
    cache_before = collect_predictions_once(
        before_model,
        image_files,
        cfg.test_labels_dir,
        infer_conf_min=cfg.infer_conf_min,
        nms_iou=cfg.nms_iou,
        max_det=cfg.max_det,
    )

    print("\n[2/4] Collect AFTER predictions...")
    cache_after = collect_predictions_once(
        after_model,
        image_files,
        cfg.test_labels_dir,
        infer_conf_min=cfg.infer_conf_min,
        nms_iou=cfg.nms_iou,
        max_det=cfg.max_det,
    )

    print("\n[3/4] Sweeping thresholds...")
    rows_before = sweep_eval(cache_before, cfg.conf_thresholds, cfg.iou_match_thresholds)
    rows_after = sweep_eval(cache_after, cfg.conf_thresholds, cfg.iou_match_thresholds)

    with open(cfg.output_dir / "sweep_before.json", "w", encoding="utf-8") as f:
        json.dump(rows_before, f, indent=2)
    with open(cfg.output_dir / "sweep_after.json", "w", encoding="utf-8") as f:
        json.dump(rows_after, f, indent=2)

    best_before = pick_operating_points(rows_before)
    best_after = pick_operating_points(rows_after)
    with open(cfg.output_dir / "best_points.json", "w", encoding="utf-8") as f:
        json.dump({"best_before": best_before, "best_after": best_after}, f, indent=2)

    print("\n=== BEST F1 OPERATING POINTS ===")
    for iou_thr in cfg.iou_match_thresholds:
        key = f"best_f1_iou_{iou_thr}"
        before = best_before.get(key)
        after = best_after.get(key)
        if before and after:
            print(f"\nMatch IoU = {iou_thr}")
            print(
                f"  BEFORE: conf={before['conf_thr']:.3f} P={before['precision']:.4f} "
                f"R={before['recall']:.4f} F1={before['f1']:.4f} TP={before['tp']} FP={before['fp']} FN={before['fn']}"
            )
            print(
                f"  AFTER : conf={after['conf_thr']:.3f} P={after['precision']:.4f} "
                f"R={after['recall']:.4f} F1={after['f1']:.4f} TP={after['tp']} FP={after['fp']} FN={after['fn']}"
            )

    print("\n[4/4] Saving plots...")
    plot_pr_curves(rows_before, rows_after, cfg.output_dir / "plots")
    plot_f1_vs_conf(rows_before, rows_after, cfg.output_dir / "plots")
    plot_conf_hist_from_cache(cache_before, cfg.output_dir / "plots" / "conf_hist_before.png", "Confidence Distribution (Before)")
    plot_conf_hist_from_cache(cache_after, cfg.output_dir / "plots" / "conf_hist_after.png", "Confidence Distribution (After)")

    if cfg.save_error_examples:
        print(
            f"\nSaving error examples at conf={cfg.draw_conf_threshold}, "
            f"match IoU={cfg.draw_iou_match_threshold} ..."
        )
        save_error_examples(
            cache_before,
            cfg.output_dir / "error_examples_before",
            conf_thr=cfg.draw_conf_threshold,
            iou_match_thr=cfg.draw_iou_match_threshold,
            num_images=cfg.num_error_images,
        )
        save_error_examples(
            cache_after,
            cfg.output_dir / "error_examples_after",
            conf_thr=cfg.draw_conf_threshold,
            iou_match_thr=cfg.draw_iou_match_threshold,
            num_images=cfg.num_error_images,
        )

    print(f"\nDone. Outputs: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
