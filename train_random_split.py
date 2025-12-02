from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_results(results_dir):
    """绘制训练结果图表"""
    results_dir = Path(results_dir)
    results_csv = results_dir / "results.csv"
    
    if not results_csv.exists():
        print(f"未找到训练结果文件: {results_csv}")
        return
    
    try:
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # 去除列名中的空格
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练结果监控', fontsize=16)
        
        # 损失曲线
        if 'train/box_loss' in df.columns:
            axes[0,0].plot(df['epoch'], df['train/box_loss'], label='训练box损失', color='blue')
            axes[0,0].plot(df['epoch'], df['val/box_loss'], label='验证box损失', color='red')
            axes[0,0].set_title('Box损失曲线')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].legend()
            axes[0,0].grid(True)
        
        # mAP曲线
        if 'metrics/mAP50(B)' in df.columns:
            axes[0,1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
            axes[0,1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
            axes[0,1].set_title('mAP性能曲线')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('mAP')
            axes[0,1].legend()
            axes[0,1].grid(True)
        
        # 学习率曲线
        if 'lr/pg0' in df.columns:
            axes[1,0].plot(df['epoch'], df['lr/pg0'], label='学习率', color='purple')
            axes[1,0].set_title('学习率变化')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].legend()
            axes[1,0].grid(True)
        
        # 精度和召回率
        if 'metrics/precision(B)' in df.columns:
            axes[1,1].plot(df['epoch'], df['metrics/precision(B)'], label='精度', color='red')
            axes[1,1].plot(df['epoch'], df['metrics/recall(B)'], label='召回率', color='blue')
            axes[1,1].set_title('精度和召回率')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Score')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = results_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线保存到: {plot_path}")
        plt.show()
        
        # 显示最佳结果
        best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax()]
        print(f"\n=== 最佳性能 (Epoch {int(best_epoch['epoch'])}) ===")
        print(f"mAP@0.5: {best_epoch['metrics/mAP50(B)']:.4f}")
        print(f"mAP@0.5:0.95: {best_epoch['metrics/mAP50-95(B)']:.4f}")
        print(f"精度: {best_epoch['metrics/precision(B)']:.4f}")
        print(f"召回率: {best_epoch['metrics/recall(B)']:.4f}")
        
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")

def evaluate_model(model, data_yaml):
    """详细评估模型性能"""
    print("\n" + "="*50)
    print("模型评估")
    print("="*50)
    
    # 验证集评估
    print("\n--- 验证集评估 ---")
    val_metrics = model.val(data=data_yaml, split='val')
    print(f"验证集 mAP@0.5: {val_metrics.box.map50:.4f}")
    print(f"验证集 mAP@0.5:0.95: {val_metrics.box.map:.4f}")
    print(f"验证集 精度: {val_metrics.box.p[0]:.4f}")
    print(f"验证集 召回率: {val_metrics.box.r[0]:.4f}")
    
    # 测试集评估
    print("\n--- 测试集评估 ---")
    test_metrics = model.val(data=data_yaml, split='test')
    print(f"测试集 mAP@0.5: {test_metrics.box.map50:.4f}")
    print(f"测试集 mAP@0.5:0.95: {test_metrics.box.map:.4f}")
    print(f"测试集 精度: {test_metrics.box.p[0]:.4f}")
    print(f"测试集 召回率: {test_metrics.box.r[0]:.4f}")
    
    # 训练集评估（可选)
    print("\n--- 训练集评估 ---")
    train_metrics = model.val(data=data_yaml, split='train')
    print(f"训练集 mAP@0.5: {train_metrics.box.map50:.4f}")
    print(f"训练集 mAP@0.5:0.95: {train_metrics.box.map:.4f}")
    
    return val_metrics, test_metrics, train_metrics

def main():
    # 使用新处理的数据集
    data_yaml = 'dataset_random_split/icon.yaml'
    if not os.path.exists(data_yaml):
        print("请先运行 pack_icondetect_dataset.py 准备数据集")
        print("命令: python pack_icondetect_dataset.py --dataset w:\\Downloads\\new_dataset --output new_dataset_split")
        return
    
    print("数据集配置文件:", data_yaml)
    
    # 加载模型
    print("加载预训练模型...")
    model = YOLO('icon_detect/model.pt')
    
    # 训练配置
    train_config = {
        'data': data_yaml,
        'epochs': 60,           # 训练轮数
        'imgsz': 640,
        'batch': 8,
        'lr0': 0.0015,  # 初始学习率
        'lrf': 0.01,         
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'patience': 15,        

        'hsv_h': 0.01,         # 降低（原 0.015）
        'hsv_s': 0.4,          # 降低（原 0.7）
        'hsv_v': 0.3,          # 降低（原 0.4）
        'degrees': 5,          # 降低（原 10）
        'translate': 0.08,     # 降低（原 0.1）
        'scale': 0.3,          # 降低（原 0.5）
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.3,         # 大幅降低（原 1.0）⬇️⬇️⬇️
        'mixup': 0.0,          # 关闭（原 0.1）⬇️
        'copy_paste': 0.0,
        
        'save': True,
        'cache': True,
        'project': 'runs/detect',
        'name': 'icon_detect_new_dataset',
        'exist_ok': True,
        'plots': True,
        'verbose': True,
    }
    
    print("\n" + "="*50)
    print("开始训练")
    print("="*50)
    print(f"训练配置:")
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    results = model.train(**train_config)
    
    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)
    print(f"最佳模型保存在: {results.save_dir}")
    print(f"训练结果目录: {results.save_dir}")
    
    # 绘制训练曲线
    plot_training_results(results.save_dir)
    
    # 详细评估模型
    val_metrics, test_metrics, train_metrics = evaluate_model(model, data_yaml)
    
    # 保存评估结果
    eval_results = {
        'validation': {
            'mAP50': float(val_metrics.box.map50),
            'mAP50_95': float(val_metrics.box.map),
            'precision': float(val_metrics.box.p[0]),
            'recall': float(val_metrics.box.r[0])
        },
        'test': {
            'mAP50': float(test_metrics.box.map50),
            'mAP50_95': float(test_metrics.box.map),
            'precision': float(test_metrics.box.p[0]),
            'recall': float(test_metrics.box.r[0])
        },
        'train': {
            'mAP50': float(train_metrics.box.map50),
            'mAP50_95': float(train_metrics.box.map)
        }
    }
    
    import json
    eval_file = Path(results.save_dir) / "evaluation_results.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n详细评估结果保存到: {eval_file}")
    
    # 检查过拟合
    val_test_diff = val_metrics.box.map50 - test_metrics.box.map50
    train_val_diff = train_metrics.box.map50 - val_metrics.box.map50
    
    print(f"\n=== 模型诊断 ===")
    print(f"验证集与测试集差异: {val_test_diff:.4f}")
    print(f"训练集与验证集差异: {train_val_diff:.4f}")
    
    if train_val_diff > 0.1:
        print("可能存在过拟合，建议:")
        print("增加数据增强")
        print("降低学习率")
        print("增加dropout")
        print("早停策略")
    elif val_test_diff > 0.05:
        print("验证集与测试集表现差异较大")
    else:
        print("模型表现良好，没有明显过拟合")

if __name__ == "__main__":
    main()