import os, shutil, random, glob
from pathlib import Path
import argparse

root = Path(__file__).parent

def find_all_image_label_pairs(dataset_path):
    """Scan the dataset directory and find all image and corresponding label pairs"""
    dataset_path = Path(dataset_path)
    pairs = []
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # Count scanned files
    images_found = 0
    labels_found = 0
    
    # For your dataset structure: images/train/
    images_dir = dataset_path / "images" / "train"  
    labels_dir = dataset_path / "labels" / "train"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return []
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return []
    
    print(f"📂 Scanning images from: {images_dir}")
    print(f"📂 Scanning labels from: {labels_dir}")
    
    # Walk through all subdirectories
    for img_path in images_dir.rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            images_found += 1
            
            # Find corresponding label file
            # Get relative path from train directory
            rel_path = img_path.relative_to(images_dir)
            
            # Construct label path
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / rel_path.parent / label_name
            
            if label_path.exists():
                labels_found += 1
                pairs.append((img_path, label_path))
    
    print(f"\n📊 Scan Results:")
    print(f"   Images found: {images_found}")
    print(f"   Labels found: {labels_found}")
    print(f"   Valid pairs: {len(pairs)}")
    
    return pairs

def copy_pair(img_path: Path, lbl_path: Path, split: str, out_root: Path):
    """Copy image and label to the specified directory"""
    # Keep original filenames
    dst_img = out_root / "images" / split / img_path.name
    dst_lbl = out_root / "labels" / split / (img_path.stem + ".txt")
    
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(img_path, dst_img)
    shutil.copy2(lbl_path, dst_lbl)

def extract_task_name(filename):
    """从文件名提取任务名称（去除时间戳和编号）"""
  
    
    name = Path(filename).stem
    
  
    if name.startswith("masked_"):
        name = name[7:]
    
 
    import re
    name = re.sub(r'_\d+_times(_\d+)?$', '', name)
    
    return name

def stratified_split_by_filename(pairs, train_ratio=0.8, val_ratio=0.1, seed=42):
    """根据文件名（任务类型）进行分层随机分割"""
    from collections import defaultdict
    task_groups = defaultdict(list)
    
    for img_path, lbl_path in pairs:
        task_name = extract_task_name(img_path.name)
        task_groups[task_name].append((img_path, lbl_path))
    
    
    random.seed(seed)
    train_pairs, val_pairs, test_pairs = [], [], []
    
    small_tasks = []  
    
    for task_name, samples in task_groups.items():
        shuffled = samples.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        if n < 3:
            small_tasks.append((task_name, shuffled))
            continue
        
        train_n = max(1, int(n * train_ratio))
        val_n = max(1, int(n * val_ratio))
        
        train_pairs.extend(shuffled[:train_n])
        val_pairs.extend(shuffled[train_n:train_n + val_n])
        test_pairs.extend(shuffled[train_n + val_n:])

    if small_tasks:
        print(f"\n⚠️  处理 {len(small_tasks)} 个小样本任务 (样本数 < 3):")
        all_small_samples = []
        for task_name, samples in small_tasks:
            print(f"    {task_name}: {len(samples)} 样本")
            all_small_samples.extend(samples)
        
        random.shuffle(all_small_samples)
        n_small = len(all_small_samples)
        train_n = int(n_small * train_ratio)
        val_n = int(n_small * val_ratio)
        
        train_pairs.extend(all_small_samples[:train_n])
        val_pairs.extend(all_small_samples[train_n:train_n + val_n])
        test_pairs.extend(all_small_samples[train_n + val_n:])
    
    # 再次打乱以避免按任务排序
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)
    
    # 验证任务分布
    def count_tasks(pair_list):
        tasks = set()
        for img, lbl in pair_list:
            tasks.add(extract_task_name(img.name))
        return len(tasks)
    
    print(f"\n任务类型分布:")
    print(f"  训练集包含: {count_tasks(train_pairs)} 个任务")
    print(f"  验证集包含: {count_tasks(val_pairs)} 个任务")
    print(f"  测试集包含: {count_tasks(test_pairs)} 个任务")
    
    return train_pairs, val_pairs, test_pairs

def create_dataset_splits(dataset_path, output_name="dataset_random_split", split_method="filename"):
    """Create dataset splits: 80% train, 10% val, 10% test
    
    Args:
        split_method: 'random' | 'filename'
    """
    
    dataset_path = Path(dataset_path)
    
    # 🔍 找到所有图像-标注对
    print("🔍 扫描数据集...")
    all_pairs = find_all_image_label_pairs(dataset_path)
    
    if len(all_pairs) == 0:
        print("❌ 未找到任何有效的图像-标注对！")
        print("请检查数据集目录结构")
        return None
    
    print(f"\n✅ 找到 {len(all_pairs)} 个有效的图像-标注对")
    
    # 选择分割方式
    if split_method == "filename":
        print("\n使用基于文件名的分层随机分割 (Filename-based Stratified Split)")
        train_pairs, val_pairs, test_pairs = stratified_split_by_filename(
            all_pairs, train_ratio=0.8, val_ratio=0.1, seed=42
        )
    else:  # random
        print("\n使用完全随机分割 (Random Split)")
        random.seed(42)
        pairs_shuffled = all_pairs.copy()
        random.shuffle(pairs_shuffled)
        
        total_count = len(pairs_shuffled)
        train_count = int(total_count * 0.8)
        val_count = int(total_count * 0.1)
        
        train_pairs = pairs_shuffled[:train_count]
        val_pairs = pairs_shuffled[train_count:train_count + val_count]
        test_pairs = pairs_shuffled[train_count + val_count:]
    
    out_root = root / output_name
    
    # Clean output directory
    if out_root.exists():
        print(f"\n🧹 清理已存在的目录: {out_root}")
        shutil.rmtree(out_root)
    
    # Copy files
    print(f"\n📦 复制文件到输出目录...")
    for img, lbl in train_pairs:
        copy_pair(img, lbl, "train", out_root)
    
    for img, lbl in val_pairs:
        copy_pair(img, lbl, "val", out_root)
    
    for img, lbl in test_pairs:
        copy_pair(img, lbl, "test", out_root)
    
    print(f"\n✅ 数据集分割完成!")
    print(f"   训练集: {len(train_pairs)} 样本")
    print(f"   验证集: {len(val_pairs)} 样本")
    print(f"   测试集: {len(test_pairs)} 样本")
    print(f"   输出目录: {out_root}")
    
    # Generate config file - use relative paths
    yaml_content = f"""path: ./{output_name}
train: images/train
val: images/val
test: images/test

names:
  0: interactive_icon
"""
    
    yaml_path = out_root / "icon.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    
    print(f"\n📝 配置文件已生成: {yaml_path}")
    
    return out_root

def main():
    parser = argparse.ArgumentParser(description="Process dataset and split into train/val/test sets")
    parser.add_argument("--dataset", type=str, default="merged_dataset", 
                       help="Dataset path (default: merged_dataset)")
    parser.add_argument("--output", type=str, default="dataset_split",
                       help="Output directory name (default: dataset_split)")
    parser.add_argument("--method", type=str, default="filename",
                       choices=["random", "class", "filename"],
                       help="Split method: random | class | filename (default: filename)")
    args = parser.parse_args()
    
    # Convert to absolute path
    if not Path(args.dataset).is_absolute():
        dataset_path = root / args.dataset
    else:
        dataset_path = Path(args.dataset)
    
    print(f"\n{'='*70}")
    print(f"数据集分割工具")
    print(f"{'='*70}")
    print(f"数据集路径: {dataset_path}")
    print(f"输出目录: {args.output}")
    print(f"分割方法: {args.method}")
    print(f"{'='*70}\n")
    
    create_dataset_splits(dataset_path, args.output, split_method=args.method)

if __name__ == "__main__":
    main()
