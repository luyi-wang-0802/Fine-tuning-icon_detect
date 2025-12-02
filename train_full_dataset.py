"""
Split Dataset Training Script
Functionality: Trains using separate train and val datasets
Applicable Scenarios: When distinct training and validation sets are available to evaluate model generalization capabilities
"""

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

root = Path(__file__).parent


def check_dataset(dataset_path):
    """Check dataset structure"""
    dataset_path = Path(dataset_path)
    data_yaml = dataset_path / "data.yaml"
    
    if not data_yaml.exists():
        print(f"❌ Error: Configuration file does not exist: {data_yaml}")
        return None
    
    # Load configuration
    import yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded configuration: {data_yaml}")
    print(f"\n📄 Configuration Content:")
    print(f"{'-'*60}")
    with open(data_yaml, 'r', encoding='utf-8') as f:
        print(f.read())
    print(f"{'-'*60}\n")
    
    # Dataset statistics
    train_dir = dataset_path / "images" / "train"
    val_dir = dataset_path / "images" / "val"
    
    train_count = len(list(train_dir.glob("*.png"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.png"))) if val_dir.exists() else 0
    
    print(f"📊 Dataset Statistics:")
    print(f"   Training Set: {train_count} images")
    print(f"   Validation Set: {val_count} images")
    print(f"   Total:   {train_count + val_count} images\n")
    
    return data_yaml, train_count, val_count


def plot_training_results(results_dir):
    """Plot training results"""
    results_dir = Path(results_dir)
    results_csv = results_dir / "results.csv"
    
    if not results_csv.exists():
        print(f"⚠️  Training results file not found: {results_csv}")
        return
    
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Process Monitoring - Split Dataset Training', fontsize=16, fontweight='bold')
        
        # 1. Box Loss (Training vs Validation)
        if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
            axes[0,0].plot(df['epoch'], df['train/box_loss'], 
                          label='Training Set', color='blue', linewidth=2)
            axes[0,0].plot(df['epoch'], df['val/box_loss'], 
                          label='Validation Set', color='red', linewidth=2)
            axes[0,0].set_title('Box Loss', fontweight='bold')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. mAP Curves
        if 'metrics/mAP50(B)' in df.columns:
            axes[0,1].plot(df['epoch'], df['metrics/mAP50(B)'], 
                          label='mAP@0.5', color='green', linewidth=2)
            axes[0,1].plot(df['epoch'], df['metrics/mAP50-95(B)'], 
                          label='mAP@0.5:0.95', color='orange', linewidth=2)
            axes[0,1].set_title('mAP Performance (Validation Set)', fontweight='bold')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('mAP')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Learning Rate
        if 'lr/pg0' in df.columns:
            axes[1,0].plot(df['epoch'], df['lr/pg0'], 
                          label='Learning Rate', color='purple', linewidth=2)
            axes[1,0].set_title('Learning Rate Changes', fontweight='bold')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Precision and Recall
        if 'metrics/precision(B)' in df.columns:
            axes[1,1].plot(df['epoch'], df['metrics/precision(B)'], 
                          label='Precision', color='red', linewidth=2)
            axes[1,1].plot(df['epoch'], df['metrics/recall(B)'], 
                          label='Recall', color='blue', linewidth=2)
            axes[1,1].set_title('Precision and Recall (Validation Set)', fontweight='bold')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Score')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = results_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved: {plot_path}")
        
        plt.close()
        
        # Print best performance
        if 'metrics/mAP50(B)' in df.columns:
            best_epoch_idx = df['metrics/mAP50(B)'].idxmax()
            best_epoch = df.loc[best_epoch_idx]
            
            print(f"\n{'='*60}")
            print(f"Best Performance (Epoch {int(best_epoch['epoch'])})")
            print(f"{'='*60}")
            print(f"  mAP@0.5:      {best_epoch['metrics/mAP50(B)']:.4f}")
            print(f"  mAP@0.5:0.95: {best_epoch['metrics/mAP50-95(B)']:.4f}")
            print(f"  Precision:    {best_epoch['metrics/precision(B)']:.4f}")
            print(f"  Recall:       {best_epoch['metrics/recall(B)']:.4f}")
            print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model(model, data_yaml):
    """Detailed evaluation of model performance"""
    print(f"\n{'='*60}")
    print("Model Evaluation")
    print(f"{'='*60}")
    
    # Validation set evaluation
    print("\n📊 Validation Set Evaluation:")
    try:
        val_metrics = model.val(data=data_yaml, split='val')
        print(f"  mAP@0.5:      {val_metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {val_metrics.box.map:.4f}")
        print(f"  Precision:    {val_metrics.box.p[0]:.4f}")
        print(f"  Recall:       {val_metrics.box.r[0]:.4f}")
        print(f"  F1 Score:     {val_metrics.box.f1[0]:.4f}")
    except Exception as e:
        print(f"❌ Validation set evaluation failed: {e}")
        val_metrics = None
    
    # Training set evaluation
    print("\n📊 Training Set Evaluation:")
    try:
        train_metrics = model.val(data=data_yaml, split='train')
        print(f"  mAP@0.5:      {train_metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {train_metrics.box.map:.4f}")
        print(f"  Precision:    {train_metrics.box.p[0]:.4f}")
        print(f"  Recall:       {train_metrics.box.r[0]:.4f}")
        print(f"  F1 Score:     {train_metrics.box.f1[0]:.4f}")
    except Exception as e:
        print(f"❌ Training set evaluation failed: {e}")
        train_metrics = None
    
    # Overfitting diagnosis
    if val_metrics and train_metrics:
        print(f"\n{'='*60}")
        print("Overfitting Diagnosis")
        print(f"{'='*60}")
        
        train_val_diff = train_metrics.box.map50 - val_metrics.box.map50
        print(f"  Training mAP@0.5: {train_metrics.box.map50:.4f}")
        print(f"  Validation mAP@0.5: {val_metrics.box.map50:.4f}")
        print(f"  Difference:       {train_val_diff:.4f}")
        
        if train_val_diff > 0.1:
            print(f"\n⚠️  Possible overfitting detected!")
            print(f"  Recommendations:")
            print(f"    - Increase data augmentation")
            print(f"    - Reduce learning rate")
            print(f"    - Use dropout")
            print(f"    - Decrease patience")
        elif train_val_diff > 0.05:
            print(f"\n✅ Slight overfitting, within acceptable range")
        else:
            print(f"\n✅ Model performance is good, no obvious overfitting")
        
        print(f"{'='*60}")
    
    return val_metrics, train_metrics


def main():
    dataset_name = "new_dataset"    # Dataset path
    dataset_path = root / dataset_name
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory does not exist: {dataset_path}")
        print(f"\nPlease ensure the dataset is located at: {dataset_path}")
        return
    
    print(f"\n{'='*60}")
    print("Split Dataset Training")
    print(f"{'='*60}")
    print(f"Dataset path: {dataset_path}")
    print(f"Training strategy: Using separate train and val datasets")
    print(f"{'='*60}\n")
    
    # Check dataset
    result = check_dataset(dataset_path)
    
    if result is None:
        return
    
    data_yaml, train_count, val_count = result
    
    # Load pre-trained model
    print("Loading pre-trained model: icon_detect/model.pt")
    model = YOLO('icon_detect/model.pt')
    
    # Training configuration
    train_config = {
        'data': str(data_yaml),
        'epochs': 100,          # Number of training epochs
        'imgsz': 640,
        'batch': 8,
        
        'lr0': 0.0015,         # Initial learning rate
        'lrf': 0.01,           # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'patience': 15,        # Early stopping patience
        
        # Data augmentation - moderate settings
        'hsv_h': 0.01,         # Hue augmentation
        'hsv_s': 0.4,          # Saturation augmentation
        'hsv_v': 0.3,          # Value augmentation
        'degrees': 5,          # Rotation degrees
        'translate': 0.08,     # Translation
        'scale': 0.3,          # Scaling
        'shear': 0.0,          # Shearing
        'perspective': 0.0,    # Perspective transformation
        'flipud': 0.0,         # Vertical flip
        'fliplr': 0.5,         # Horizontal flip
        'mosaic': 0.3,         # Mosaic augmentation
        'mixup': 0.0,          # Mixup augmentation
        'copy_paste': 0.0,     # Copy-paste augmentation
        
        'save': True,
        'cache': True,
        'project': 'runs/split_train',
        'name': 'icon_detect_split_dataset',
        'exist_ok': True,
        'plots': True,
        'verbose': True,
    }
    
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Start training
    print(f"Starting training... (Total {train_config['epochs']} epochs)")
    print(f"📊 Training set: {train_count} images")
    print(f"📊 Validation set: {val_count} images")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = model.train(**train_config)
    
    print(f"\n{'='*60}")
    print("✅ Training completed!")
    print(f"{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Save directory: {results.save_dir}")
    print(f"Best model: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {results.save_dir / 'weights' / 'last.pt'}")
    print(f"{'='*60}\n")
    
    # Plot training curves
    plot_training_results(results.save_dir)
    
    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    best_model = YOLO(results.save_dir / 'weights' / 'best.pt')
    
    val_metrics, train_metrics = evaluate_model(best_model, str(data_yaml))
    
    # Save evaluation results
    if val_metrics and train_metrics:
        eval_results = {
            'training_mode': 'split_dataset',
            'dataset': str(dataset_path),
            'train_count': train_count,
            'val_count': val_count,
            'total_epochs': train_config['epochs'],
            'timestamp': datetime.now().isoformat(),
            
            'validation_metrics': {
                'mAP50': float(val_metrics.box.map50),
                'mAP50_95': float(val_metrics.box.map),
                'precision': float(val_metrics.box.p[0]),
                'recall': float(val_metrics.box.r[0]),
                'f1': float(val_metrics.box.f1[0])
            },
            
            'train_metrics': {
                'mAP50': float(train_metrics.box.map50),
                'mAP50_95': float(train_metrics.box.map),
                'precision': float(train_metrics.box.p[0]),
                'recall': float(train_metrics.box.r[0]),
                'f1': float(train_metrics.box.f1[0])
            },
            
            'overfitting_analysis': {
                'train_val_diff': float(train_metrics.box.map50 - val_metrics.box.map50),
                'status': 'good' if (train_metrics.box.map50 - val_metrics.box.map50) <= 0.05 else 
                         'acceptable' if (train_metrics.box.map50 - val_metrics.box.map50) <= 0.1 else 
                         'overfitting'
            },
            
            'model_paths': {
                'best': str(results.save_dir / 'weights' / 'best.pt'),
                'last': str(results.save_dir / 'weights' / 'last.pt')
            }
        }
        
        eval_file = results.save_dir / "evaluation_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Evaluation results saved: {eval_file}")
    
    # Training summary
    print(f"\n{'='*60}")
    print("📝 Training Summary")
    print(f"{'='*60}")
    print(f"✅ Dataset mode: Split training (independent train/val)")
    print(f"✅ Training set: {train_count} images")
    print(f"✅ Validation set: {val_count} images")
    print(f"✅ Ratio: {train_count/(train_count+val_count)*100:.1f}% / {val_count/(train_count+val_count)*100:.1f}%")
    print(f"\n💡 Advantages:")
    print(f"  - Independent validation set provides a true assessment of generalization")
    print(f"  - Early stopping based on validation performance")
    print(f"  - Can monitor overfitting")
    print(f"\n⚠️  Note:")
    print(f"  - No independent test set; final performance is based on the validation set")
    print(f"  - Cross-validation is recommended for further verification")
    print(f"{'='*60}\n")
    
    print("🎉 Training process completed!")


if __name__ == "__main__":
    main()