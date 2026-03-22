import os
import shutil
from pathlib import Path
from collections import defaultdict

def get_image_label_pairs(dataset_path):
    """获取数据集中的图像和标注文件对"""
    images_dir = Path(dataset_path) / "images" / "train"
    labels_dir = Path(dataset_path) / "labels" / "train"
    
    # 获取所有图像文件
    image_files = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(Path(root) / file)
    
    # 为每个图像找到对应的标注文件
    pairs = []
    for img_path in image_files:
        # 获取相对于train目录的路径
        rel_path = img_path.relative_to(images_dir)
        # 构造对应的标注文件路径
        label_name = img_path.stem + '.txt'
        label_path = labels_dir / rel_path.parent / label_name
        
        if label_path.exists():
            pairs.append((img_path, label_path))
    
    return pairs

def ensure_unique_name(existing_names, base_name, extension):
    """确保文件名唯一"""
    if base_name + extension not in existing_names:
        return base_name + extension
    
    counter = 1
    while f"{base_name}_{counter}{extension}" in existing_names:
        counter += 1
    return f"{base_name}_{counter}{extension}"

def merge_datasets(dataset_paths, output_path, dataset2_limit=44):
    """合并多个数据集"""
    output_images = Path(output_path) / "images" / "train"
    output_labels = Path(output_path) / "labels" / "train"
    
    # 创建输出目录
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    existing_names = set()
    copied_count = defaultdict(int)
    
    for idx, dataset_path in enumerate(dataset_paths):
        pairs = get_image_label_pairs(dataset_path)
        
        # 如果是dataset_2，只取前40个
        if "dataset_2" in str(dataset_path):
            pairs = pairs[:dataset2_limit]
            print(f"  从dataset_2提取前{len(pairs)}个样本")
        
        for img_path, label_path in pairs:
            # 获取原始文件名
            img_extension = img_path.suffix
            base_name = img_path.stem
            
            # 确保名称唯一
            unique_img_name = ensure_unique_name(existing_names, base_name, img_extension)
            unique_label_name = ensure_unique_name(existing_names, base_name, '.txt')
            
            # 更新已存在的名称集合
            existing_names.add(unique_img_name)
            existing_names.add(unique_label_name)
            
            # 复制文件
            shutil.copy2(img_path, output_images / unique_img_name)
            shutil.copy2(label_path, output_labels / unique_label_name)
            
            copied_count[f"dataset_{idx+1}"] += 1
    
    # 创建train.txt文件
    train_txt_path = Path(output_path) / "train.txt"
    with open(train_txt_path, 'w') as f:
        for img_file in sorted(output_images.glob('*')):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                f.write(f"./images/train/{img_file.name}\n")
    
    # 复制data.yaml（使用dataset_1的作为模板）
    if (Path(dataset_paths[0]) / "data.yaml").exists():
        shutil.copy2(Path(dataset_paths[0]) / "data.yaml", Path(output_path) / "data.yaml")
    
    print("\n" + "="*50)
    print("合并完成！统计信息：")
    for dataset, count in copied_count.items():
        print(f"  {dataset}: {count} 个样本")
    print(f"  总计: {sum(copied_count.values())} 个样本")
    print(f"\n输出路径: {output_path}")

if __name__ == "__main__":
    # 设置数据集路径
    base_path = r"w:\hiwi\Fine-tuning icon_detect"
    dataset_paths = [
        os.path.join(base_path, "dataset_1"),
        os.path.join(base_path, "dataset_2"),
        os.path.join(base_path, "dataset_3")
    ]
    
    # 设置输出路径
    output_path = os.path.join(base_path, "merged_dataset")
    
    # 执行合并
    merge_datasets(dataset_paths, output_path, dataset2_limit=44)
    
    print("\n完成！")