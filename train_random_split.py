import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).parent


def plot_training_curves(save_dir: Path) -> None:
    """Create a compact summary plot from Ultralytics results.csv."""
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        print(f"Training metrics file not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Curves", fontsize=14)

    if {"train/box_loss", "val/box_loss"} <= set(df.columns):
        axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="train box")
        axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="val box")
        axes[0, 0].set_title("Box Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    if {"metrics/mAP50(B)", "metrics/mAP50-95(B)"} <= set(df.columns):
        axes[0, 1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
        axes[0, 1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
        axes[0, 1].set_title("mAP")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    if "lr/pg0" in df.columns:
        axes[1, 0].plot(df["epoch"], df["lr/pg0"], label="lr")
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    if {"metrics/precision(B)", "metrics/recall(B)"} <= set(df.columns):
        axes[1, 1].plot(df["epoch"], df["metrics/precision(B)"], label="precision")
        axes[1, 1].plot(df["epoch"], df["metrics/recall(B)"], label="recall")
        axes[1, 1].set_title("Precision / Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    out_path = save_dir / "training_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {out_path}")


def eval_split(model: YOLO, data_yaml: str, split: str) -> dict:
    """Evaluate one dataset split and return compact metrics."""
    metrics = model.val(data=data_yaml, split=split, verbose=False).box
    result = {
        "mAP50": float(metrics.map50),
        "mAP50_95": float(metrics.map),
    }
    if metrics.p is not None and len(metrics.p) > 0:
        result["precision"] = float(metrics.p[0])
        result["recall"] = float(metrics.r[0])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on a train/val/test YOLO dataset and evaluate all splits."
    )
    parser.add_argument(
        "--data",
        default="dataset_split/icon.yaml",
        help="Path to YOLO dataset yaml.",
    )
    parser.add_argument(
        "--model",
        default="first_finetune/weights/best.pt",
        help="Starting model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=800, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="icon_detect_new_dataset", help="Run name.")
    parser.add_argument("--device", default=None, help="CUDA device string passed to Ultralytics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    model_path = Path(args.model)

    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = YOLO(str(model_path))

    train_config = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": 0.0015,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "patience": 15,
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "degrees": 5,
        "translate": 0.08,
        "scale": 0.3,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.3,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "save": True,
        "cache": True,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }
    if args.device:
        train_config["device"] = args.device

    print("Starting training...")
    results = model.train(**train_config)
    save_dir = Path(results.save_dir)
    print(f"Training finished. Output directory: {save_dir}")

    plot_training_curves(save_dir)

    best_model = YOLO(str(save_dir / "weights" / "best.pt"))
    val_res = eval_split(best_model, str(data_yaml), "val")
    test_res = eval_split(best_model, str(data_yaml), "test")
    train_res = eval_split(best_model, str(data_yaml), "train")

    eval_results = {
        "validation": val_res,
        "test": test_res,
        "train": train_res,
    }

    eval_file = save_dir / "evaluation_results.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation results to {eval_file}")

    val_test_diff = val_res["mAP50"] - test_res["mAP50"]
    train_val_diff = train_res["mAP50"] - val_res["mAP50"]

    print("\n=== Model Diagnostics ===")
    print(f"train - val mAP50 difference: {train_val_diff:.4f}")
    print(f"val - test mAP50 difference: {val_test_diff:.4f}")

    if train_val_diff > 0.1:
        print("Possible overfitting: train performance is much higher than validation.")
    elif val_test_diff > 0.05:
        print("Possible dataset shift: validation and test performance differ noticeably.")
    else:
        print("Train/val/test performance is broadly consistent.")


if __name__ == "__main__":
    main()
