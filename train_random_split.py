from ultralytics import YOLO
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(save_dir: Path):
    """根据 results.csv 画一张综合训练曲线（可选）"""
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        print(f"未找到 {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Curves", fontsize=14)

    # box loss
    if {"train/box_loss", "val/box_loss"} <= set(df.columns):
        axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="train box")
        axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="val box")
        axes[0, 0].set_title("Box Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # mAP
    if {"metrics/mAP50(B)", "metrics/mAP50-95(B)"} <= set(df.columns):
        axes[0, 1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
        axes[0, 1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
        axes[0, 1].set_title("mAP")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # lr
    if "lr/pg0" in df.columns:
        axes[1, 0].plot(df["epoch"], df["lr/pg0"], label="lr")
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # precision / recall
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
    print(f"训练曲线保存到: {out_path}")


def eval_split(model: YOLO, data_yaml: str, split: str):
    """在指定 split 上评估，返回一个 dict"""
    metrics = model.val(data=data_yaml, split=split, verbose=False).box
    result = {
        "mAP50": float(metrics.map50),
        "mAP50_95": float(metrics.map),
    }
    if metrics.p is not None and len(metrics.p) > 0:
        result["precision"] = float(metrics.p[0])
        result["recall"] = float(metrics.r[0])
    return result


def main():
    data_yaml = "dataset_split/icon.yaml"
    base_model_path = "first_finetune/weights/best.pt"

    # 1. 加载预训练/微调起点模型
    model = YOLO(base_model_path)

    # 2. 训练
    train_config = {
        "data": data_yaml,
        "epochs": 60,
        "imgsz": 800,
        "batch": 8,
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
        "project": "runs/detect",
        "name": "icon_detect_new_dataset",
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }

    print("开始训练...")
    results = model.train(**train_config)
    save_dir = Path(results.save_dir)
    print(f"训练完成，结果目录: {save_dir}")

    # 3. 绘制训练曲线（可选，自定义版；Ultralytics 已经有 results.png）
    plot_training_curves(save_dir)

    # 4. 用 best.pt 重新加载模型做评估
    best_model = YOLO(save_dir / "weights" / "best.pt")

    val_res = eval_split(best_model, data_yaml, "val")
    test_res = eval_split(best_model, data_yaml, "test")
    train_res = eval_split(best_model, data_yaml, "train")

    eval_results = {"validation": val_res, "test": test_res, "train": train_res}

    eval_file = save_dir / "evaluation_results.json"
    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"评估结果保存到: {eval_file}")

    # 5. 简单过拟合诊断
    val_test_diff = val_res["mAP50"] - test_res["mAP50"]
    train_val_diff = train_res["mAP50"] - val_res["mAP50"]

    print("\n=== 模型诊断 ===")
    print(f"train - val mAP50 差异: {train_val_diff:.4f}")
    print(f"val - test mAP50 差异: {val_test_diff:.4f}")

    if train_val_diff > 0.1:
        print("提示：可能过拟合（train 明显高于 val），可考虑更强增强 / 降低 lr / 减少 epoch。")
    elif val_test_diff > 0.05:
        print("提示：验证集和测试集差异较大，可能存在 domain gap，需要检查测试集分布。")
    else:
        print("train/val/test 表现接近，暂未见明显过拟合。")


if __name__ == "__main__":
    main()
