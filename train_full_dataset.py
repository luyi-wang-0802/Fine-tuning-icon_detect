"""
Train on a dataset that already has separate train and val splits.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).parent


def check_dataset(dataset_path: Path):
    """Validate the dataset structure and report image counts."""
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: configuration file does not exist: {data_yaml}")
        return None

    print(f"Loaded configuration: {data_yaml}")
    print("\nConfiguration contents:")
    print("-" * 60)
    print(data_yaml.read_text(encoding="utf-8"))
    print("-" * 60)

    train_dir = dataset_path / "images" / "train"
    val_dir = dataset_path / "images" / "val"

    image_patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    train_count = sum(len(list(train_dir.glob(pattern))) for pattern in image_patterns) if train_dir.exists() else 0
    val_count = sum(len(list(val_dir.glob(pattern))) for pattern in image_patterns) if val_dir.exists() else 0

    print("Dataset statistics:")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Total: {train_count + val_count} images\n")

    return data_yaml, train_count, val_count


def plot_training_results(results_dir: Path) -> None:
    """Generate a summary plot from Ultralytics results.csv."""
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        print(f"Training results file not found: {results_csv}")
        return

    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Process Monitoring - Split Dataset Training", fontsize=16, fontweight="bold")

        if {"train/box_loss", "val/box_loss"} <= set(df.columns):
            axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Train", color="blue", linewidth=2)
            axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="Val", color="red", linewidth=2)
            axes[0, 0].set_title("Box Loss", fontweight="bold")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        if {"metrics/mAP50(B)", "metrics/mAP50-95(B)"} <= set(df.columns):
            axes[0, 1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", color="green", linewidth=2)
            axes[0, 1].plot(
                df["epoch"],
                df["metrics/mAP50-95(B)"],
                label="mAP@0.5:0.95",
                color="orange",
                linewidth=2,
            )
            axes[0, 1].set_title("Validation mAP", fontweight="bold")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("mAP")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        if "lr/pg0" in df.columns:
            axes[1, 0].plot(df["epoch"], df["lr/pg0"], label="Learning Rate", color="purple", linewidth=2)
            axes[1, 0].set_title("Learning Rate", fontweight="bold")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("LR")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        if {"metrics/precision(B)", "metrics/recall(B)"} <= set(df.columns):
            axes[1, 1].plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="red", linewidth=2)
            axes[1, 1].plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="blue", linewidth=2)
            axes[1, 1].set_title("Precision / Recall", fontweight="bold")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = results_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved training curves to {plot_path}")
    except Exception as exc:
        print(f"Error while plotting training curves: {exc}")
        raise


def evaluate_model(model: YOLO, data_yaml: str):
    """Evaluate train and val performance."""
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    print("\nValidation split:")
    try:
        val_metrics = model.val(data=data_yaml, split="val")
        print(f"  mAP@0.5:      {val_metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {val_metrics.box.map:.4f}")
        print(f"  Precision:    {val_metrics.box.p[0]:.4f}")
        print(f"  Recall:       {val_metrics.box.r[0]:.4f}")
        print(f"  F1 Score:     {val_metrics.box.f1[0]:.4f}")
    except Exception as exc:
        print(f"Validation evaluation failed: {exc}")
        val_metrics = None

    print("\nTraining split:")
    try:
        train_metrics = model.val(data=data_yaml, split="train")
        print(f"  mAP@0.5:      {train_metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {train_metrics.box.map:.4f}")
        print(f"  Precision:    {train_metrics.box.p[0]:.4f}")
        print(f"  Recall:       {train_metrics.box.r[0]:.4f}")
        print(f"  F1 Score:     {train_metrics.box.f1[0]:.4f}")
    except Exception as exc:
        print(f"Training evaluation failed: {exc}")
        train_metrics = None

    if val_metrics and train_metrics:
        train_val_diff = train_metrics.box.map50 - val_metrics.box.map50
        print("\n" + "=" * 60)
        print("Overfitting Diagnosis")
        print("=" * 60)
        print(f"  Train mAP@0.5: {train_metrics.box.map50:.4f}")
        print(f"  Val mAP@0.5:   {val_metrics.box.map50:.4f}")
        print(f"  Difference:    {train_val_diff:.4f}")

        if train_val_diff > 0.1:
            print("\nPossible overfitting detected.")
        elif train_val_diff > 0.05:
            print("\nSlight overfitting, still within a tolerable range.")
        else:
            print("\nNo obvious overfitting.")
        print("=" * 60)

    return val_metrics, train_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on a dataset with separate train/val splits."
    )
    parser.add_argument(
        "--dataset",
        default="new_dataset",
        help="Dataset directory containing data.yaml and images/train, images/val.",
    )
    parser.add_argument(
        "--model",
        default="icon_detect/model.pt",
        help="Starting model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", default="runs/split_train", help="Output project directory.")
    parser.add_argument("--name", default="icon_detect_split_dataset", help="Run name.")
    parser.add_argument("--device", default=None, help="CUDA device string passed to Ultralytics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)

    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    print("\n" + "=" * 60)
    print("Split Dataset Training")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print("Training strategy: existing train/val split")
    print("=" * 60 + "\n")

    result = check_dataset(dataset_path)
    if result is None:
        return

    data_yaml, train_count, val_count = result

    print(f"Loading model from {model_path}")
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

    print("\nTraining configuration:")
    print("=" * 60)
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    print(f"Starting training for {train_config['epochs']} epochs")
    print(f"Train images: {train_count}")
    print(f"Val images:   {val_count}")
    print(f"Start time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = model.train(**train_config)

    print("\n" + "=" * 60)
    print("Training completed")
    print("=" * 60)
    print(f"End time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Save directory: {results.save_dir}")
    print(f"Best model:     {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last model:     {results.save_dir / 'weights' / 'last.pt'}")
    print("=" * 60 + "\n")

    plot_training_results(Path(results.save_dir))

    print("Loading best model for evaluation...")
    best_model = YOLO(str(Path(results.save_dir) / "weights" / "best.pt"))
    val_metrics, train_metrics = evaluate_model(best_model, str(data_yaml))

    if val_metrics and train_metrics:
        diff = train_metrics.box.map50 - val_metrics.box.map50
        eval_results = {
            "training_mode": "split_dataset",
            "dataset": str(dataset_path),
            "train_count": train_count,
            "val_count": val_count,
            "total_epochs": train_config["epochs"],
            "timestamp": datetime.now().isoformat(),
            "validation_metrics": {
                "mAP50": float(val_metrics.box.map50),
                "mAP50_95": float(val_metrics.box.map),
                "precision": float(val_metrics.box.p[0]),
                "recall": float(val_metrics.box.r[0]),
                "f1": float(val_metrics.box.f1[0]),
            },
            "train_metrics": {
                "mAP50": float(train_metrics.box.map50),
                "mAP50_95": float(train_metrics.box.map),
                "precision": float(train_metrics.box.p[0]),
                "recall": float(train_metrics.box.r[0]),
                "f1": float(train_metrics.box.f1[0]),
            },
            "overfitting_analysis": {
                "train_val_diff": float(diff),
                "status": "good" if diff <= 0.05 else "acceptable" if diff <= 0.1 else "overfitting",
            },
            "model_paths": {
                "best": str(Path(results.save_dir) / "weights" / "best.pt"),
                "last": str(Path(results.save_dir) / "weights" / "last.pt"),
            },
        }

        eval_file = Path(results.save_dir) / "evaluation_results.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved evaluation results to {eval_file}")


if __name__ == "__main__":
    main()
