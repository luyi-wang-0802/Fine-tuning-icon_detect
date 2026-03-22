import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional


# -----------------------------
# Config
# -----------------------------
@dataclass
class EvalConfig:
    before_model_path: Path
    after_model_path: Path
    test_images_dir: Path
    test_labels_dir: Path

    # Evaluation sweep
    conf_thresholds: List[float] = None
    iou_match_thresholds: List[float] = None  # IoU for TP match (not NMS)
    # Inference settings (fixed for reproducibility)
    infer_conf_min: float = 0.001            # collect almost-all predictions
    nms_iou: float = 0.7
    max_det: int = 500

    # Output
    output_dir: Path = Path("model_comparison_results_v2")
    save_error_examples: bool = True
    num_error_images: int = 50               # per category (FP / FN / both)
    draw_iou_match_threshold: float = 0.5    # which IoU to use for error visualization
    draw_conf_threshold: float = 0.5         # which conf to use for error visualization

    def __post_init__(self):
        if self.conf_thresholds is None:
            # wide sweep; you can adjust granularity
            self.conf_thresholds = [round(x, 3) for x in np.linspace(0.05, 0.95, 19).tolist()]
        if self.iou_match_thresholds is None:
            self.iou_match_thresholds = [0.3, 0.5]


# -----------------------------
# Helpers
# -----------------------------
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
    """
    Returns: (N,4) xyxy ground truth boxes. One-class use-case; ignores cls_id.
    YOLO format: cls x_center y_center width height (normalized)
    """
    boxes = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # cls_id = int(parts[0])  # ignored (one class)
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
    """
    Greedy one-to-one matching:
      - For each prediction, assign to best unmatched GT
      - TP if IoU >= iou_thr, else FP
      - FN = remaining unmatched GT
    Returns: tp, fp, fn, matched_ious, fp_pred_indices, fn_gt_indices
    """
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
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core evaluation
# -----------------------------
def collect_predictions_once(
    model: YOLO,
    image_files: List[Path],
    labels_dir: Path,
    infer_conf_min: float,
    nms_iou: float,
    max_det: int,
) -> Dict[str, Dict]:
    """
    Run inference once with very low conf threshold to collect predictions, then reuse for all eval thresholds.
    Returns dict keyed by image stem:
      {
        "img_path": Path,
        "img_w": int,
        "img_h": int,
        "gt": (N,4),
        "pred_xyxy": (M,4),
        "pred_conf": (M,)
      }
    """
    cache: Dict[str, Dict] = {}

    for img_path in tqdm(image_files, desc="Collecting predictions", ncols=100):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_ground_truth_yolo(labels_dir / f"{img_path.stem}.txt", w, h)

        # Inference
        res = model(
            str(img_path),
            conf=infer_conf_min,
            iou=nms_iou,
            max_det=max_det,
            verbose=False
        )[0]

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


def evaluate_from_cache(
    cache: Dict[str, Dict],
    conf_thr: float,
    iou_match_thr: float,
) -> Dict:
    """
    Evaluate at (conf_thr, iou_match_thr) using cached predictions.
    """
    tp_total = fp_total = fn_total = 0
    ious_all: List[float] = []
    num_preds_per_img = []
    num_gt_per_img = []

    for _, item in cache.items():
        gt = item["gt"]
        pred_xyxy = item["pred_xyxy"]
        pred_conf = item["pred_conf"]

        keep = pred_conf >= conf_thr
        pxy = pred_xyxy[keep] if pred_xyxy.size else pred_xyxy

        tp, fp, fn, matched_ious, _, _ = match_predictions_to_gt(pxy, gt, iou_match_thr)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        ious_all.extend(matched_ious)

        num_preds_per_img.append(int(len(pxy)))
        num_gt_per_img.append(int(len(gt)))

    metrics = calc_metrics(tp_total, fp_total, fn_total)
    out = {
        "conf_thr": float(conf_thr),
        "iou_match_thr": float(iou_match_thr),
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mean_iou_of_tps": float(np.mean(ious_all)) if len(ious_all) else 0.0,
        "avg_preds_per_img": float(np.mean(num_preds_per_img)) if num_preds_per_img else 0.0,
        "avg_gts_per_img": float(np.mean(num_gt_per_img)) if num_gt_per_img else 0.0,
    }
    return out


def sweep_eval(
    cache: Dict[str, Dict],
    conf_thresholds: List[float],
    iou_match_thresholds: List[float],
) -> List[Dict]:
    rows = []
    for iou_thr in iou_match_thresholds:
        for conf_thr in conf_thresholds:
            rows.append(evaluate_from_cache(cache, conf_thr=conf_thr, iou_match_thr=iou_thr))
    return rows


def pick_operating_points(rows: List[Dict]) -> Dict[str, Dict]:
    """
    Choose:
      - best_f1 per IoU match threshold
      - best_precision subject to recall>=x (optional)
      - best_recall subject to precision>=x (optional)
    Here keep it simple: best F1 for each IoU.
    """
    best = {}
    by_iou: Dict[float, List[Dict]] = {}
    for r in rows:
        by_iou.setdefault(r["iou_match_thr"], []).append(r)

    for iou_thr, lst in by_iou.items():
        best_f1 = max(lst, key=lambda x: x["f1"])
        best[f"best_f1_iou_{iou_thr}"] = best_f1

    return best


# -----------------------------
# Error visualization
# -----------------------------
def draw_boxes(img: np.ndarray, boxes: np.ndarray, color: Tuple[int, int, int], label: str) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
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
    """
    Save images with:
      - FP boxes (pred that didn't match any GT)
      - FN boxes (GT not matched by any pred)
      - Both overlays
    """
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
        pxy = pred_xyxy[keep] if pred_xyxy.size else pred_xyxy

        tp, fp, fn, _, fp_pred_idx, fn_gt_idx = match_predictions_to_gt(pxy, gt, iou_match_thr)

        fp_boxes = pxy[fp_pred_idx] if len(fp_pred_idx) else np.zeros((0, 4), dtype=np.float32)
        fn_boxes = gt[fn_gt_idx] if len(fn_gt_idx) else np.zeros((0, 4), dtype=np.float32)

        # FP-only
        if fp_saved < num_images and len(fp_boxes) > 0:
            vis = draw_boxes(img, fp_boxes, (0, 0, 255), "FP")  # red
            out_path = out_dir / "errors_fp" / f"{img_path.stem}_fp.png"
            cv2.imwrite(str(out_path), vis)
            fp_saved += 1

        # FN-only
        if fn_saved < num_images and len(fn_boxes) > 0:
            vis = draw_boxes(img, fn_boxes, (255, 0, 0), "FN")  # blue
            out_path = out_dir / "errors_fn" / f"{img_path.stem}_fn.png"
            cv2.imwrite(str(out_path), vis)
            fn_saved += 1

        # Both
        if both_saved < num_images and (len(fp_boxes) > 0 or len(fn_boxes) > 0):
            vis = img.copy()
            if len(fp_boxes) > 0:
                vis = draw_boxes(vis, fp_boxes, (0, 0, 255), "FP")
            if len(fn_boxes) > 0:
                vis = draw_boxes(vis, fn_boxes, (255, 0, 0), "FN")
            out_path = out_dir / "errors_both" / f"{img_path.stem}_both.png"
            cv2.imwrite(str(out_path), vis)
            both_saved += 1


# -----------------------------
# Plotting
# -----------------------------
def plot_pr_curves(rows_before: List[Dict], rows_after: List[Dict], out_dir: Path) -> None:
    """
    PR curve per IoU threshold (match IoU). Since we sweep conf, connect points.
    """
    ensure_dir(out_dir)
    iou_values = sorted(set([r["iou_match_thr"] for r in rows_before] + [r["iou_match_thr"] for r in rows_after]))

    for iou_thr in iou_values:
        b = [r for r in rows_before if r["iou_match_thr"] == iou_thr]
        a = [r for r in rows_after if r["iou_match_thr"] == iou_thr]

        # sort by recall for nicer curve
        b = sorted(b, key=lambda x: x["recall"])
        a = sorted(a, key=lambda x: x["recall"])

        plt.figure(figsize=(7, 6))
        plt.plot([x["recall"] for x in b], [x["precision"] for x in b], marker="o", label=f"Before (IoU={iou_thr})")
        plt.plot([x["recall"] for x in a], [x["precision"] for x in a], marker="o", label=f"After (IoU={iou_thr})")
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
        b = sorted([r for r in rows_before if r["iou_match_thr"] == iou_thr], key=lambda x: x["conf_thr"])
        a = sorted([r for r in rows_after if r["iou_match_thr"] == iou_thr], key=lambda x: x["conf_thr"])

        plt.figure(figsize=(7, 6))
        plt.plot([x["conf_thr"] for x in b], [x["f1"] for x in b], marker="o", label="Before")
        plt.plot([x["conf_thr"] for x in a], [x["f1"] for x in a], marker="o", label="After")
        plt.xlabel("Confidence threshold")
        plt.ylabel("F1")
        plt.title(f"F1 vs Conf (Match IoU = {iou_thr})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"f1_vs_conf_iou_{iou_thr}.png", dpi=200)
        plt.close()


def plot_conf_hist_from_cache(cache: Dict[str, Dict], out_path: Path, title: str) -> None:
    """
    True confidence distribution (from infer_conf_min), not truncated by eval conf.
    """
    confs = []
    for _, item in cache.items():
        c = item["pred_conf"]
        if c is not None and len(c) > 0:
            confs.extend(c.tolist())

    plt.figure(figsize=(7, 6))
    if len(confs) > 0:
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


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = EvalConfig(
        before_model_path=Path(r"W:/hiwi/Fine-tuning icon_detect/icon_detect/model.pt"),
        after_model_path=Path(r"W:/hiwi/Fine-tuning icon_detect/runs/detect/icon_detect_new_dataset/weights/best.pt"),
        test_images_dir=Path(r"W:/hiwi/Fine-tuning icon_detect/dataset_split/images/test"),
        test_labels_dir=Path(r"W:/hiwi/Fine-tuning icon_detect/dataset_split/labels/test"),
    )

    # Checks
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

    # Load images
    image_files = sorted(list(cfg.test_images_dir.glob("*.png")) + list(cfg.test_images_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} test images")

    # Load models
    before_model = YOLO(str(cfg.before_model_path))
    after_model = YOLO(str(cfg.after_model_path))

    # Collect predictions once per model (low conf)
    print("\n[1/4] Collect BEFORE predictions (single pass)...")
    cache_before = collect_predictions_once(
        before_model, image_files, cfg.test_labels_dir,
        infer_conf_min=cfg.infer_conf_min, nms_iou=cfg.nms_iou, max_det=cfg.max_det
    )

    print("\n[2/4] Collect AFTER predictions (single pass)...")
    cache_after = collect_predictions_once(
        after_model, image_files, cfg.test_labels_dir,
        infer_conf_min=cfg.infer_conf_min, nms_iou=cfg.nms_iou, max_det=cfg.max_det
    )

    # Sweep evaluation
    print("\n[3/4] Sweeping conf & IoU thresholds...")
    rows_before = sweep_eval(cache_before, cfg.conf_thresholds, cfg.iou_match_thresholds)
    rows_after = sweep_eval(cache_after, cfg.conf_thresholds, cfg.iou_match_thresholds)

    # Save raw sweep
    with open(cfg.output_dir / "sweep_before.json", "w", encoding="utf-8") as f:
        json.dump(rows_before, f, indent=2)
    with open(cfg.output_dir / "sweep_after.json", "w", encoding="utf-8") as f:
        json.dump(rows_after, f, indent=2)

    # Pick operating points (best F1 per IoU)
    best_before = pick_operating_points(rows_before)
    best_after = pick_operating_points(rows_after)

    summary = {
        "best_before": best_before,
        "best_after": best_after,
    }
    with open(cfg.output_dir / "best_points.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print compact summary
    print("\n=== BEST F1 OPERATING POINTS (per match IoU) ===")
    for iou_thr in cfg.iou_match_thresholds:
        kb = f"best_f1_iou_{iou_thr}"
        bb = best_before.get(kb)
        aa = best_after.get(kb)
        if bb and aa:
            print(f"\nMatch IoU = {iou_thr}")
            print(f"  BEFORE: conf={bb['conf_thr']:.3f}  P={bb['precision']:.4f} R={bb['recall']:.4f} F1={bb['f1']:.4f}  TP={bb['tp']} FP={bb['fp']} FN={bb['fn']}")
            print(f"  AFTER : conf={aa['conf_thr']:.3f}  P={aa['precision']:.4f} R={aa['recall']:.4f} F1={aa['f1']:.4f}  TP={aa['tp']} FP={aa['fp']} FN={aa['fn']}")

    # Plots
    print("\n[4/4] Saving plots...")
    plot_pr_curves(rows_before, rows_after, cfg.output_dir / "plots")
    plot_f1_vs_conf(rows_before, rows_after, cfg.output_dir / "plots")
    plot_conf_hist_from_cache(cache_before, cfg.output_dir / "plots" / "conf_hist_before.png", "Confidence Distribution (Before, untruncated)")
    plot_conf_hist_from_cache(cache_after, cfg.output_dir / "plots" / "conf_hist_after.png", "Confidence Distribution (After, untruncated)")

    # Error examples at chosen thresholds (use your fixed business thresholds)
    if cfg.save_error_examples:
        err_dir_before = cfg.output_dir / "error_examples_before"
        err_dir_after = cfg.output_dir / "error_examples_after"
        ensure_dir(err_dir_before)
        ensure_dir(err_dir_after)

        print(f"\nSaving error examples at conf={cfg.draw_conf_threshold}, match IoU={cfg.draw_iou_match_threshold} ...")
        save_error_examples(
            cache_before, err_dir_before,
            conf_thr=cfg.draw_conf_threshold,
            iou_match_thr=cfg.draw_iou_match_threshold,
            num_images=cfg.num_error_images,
        )
        save_error_examples(
            cache_after, err_dir_after,
            conf_thr=cfg.draw_conf_threshold,
            iou_match_thr=cfg.draw_iou_match_threshold,
            num_images=cfg.num_error_images,
        )

    print("\nDone.")
    print(f"Outputs: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
