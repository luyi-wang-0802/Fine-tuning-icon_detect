# Functionality: Process the dataset and split it into 80% training, 10% validation, and 10% testing.
import os, shutil, random, glob
from pathlib import Path
import argparse

root = Path(__file__).parent

def find_all_image_label_pairs(dataset_path):
    """Scan the dataset directory and find all image and corresponding label pairs"""
    dataset_path = Path(dataset_path)
    pairs = []
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    print(f"\n{'='*60}")
    print(f"Scanning dataset: {dataset_path}")
    print(f"{'='*60}")
    
    # Check if directory exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory does not exist: {dataset_path}")
        return pairs
    
    # Count scanned files
    images_found = 0
    labels_found = 0
    
    # For your dataset structure: images/train/screenshots/
    images_dir = dataset_path / "images" / "train" / "screenshots"
    labels_dir = dataset_path / "labels" / "train" / "screenshots"
    
    print(f"\nSearching image directory: {images_dir}")
    print(f"Searching label directory: {labels_dir}")
    
    if not images_dir.exists():
        print(f"⚠️  Warning: Image directory does not exist: {images_dir}")
        # Try recursive search
        print("Attempting to recursively scan the entire dataset...")
        for ext in image_extensions:
            for img_path in dataset_path.rglob(f"*{ext}"):
                images_found += 1
                label_path = find_label_for_image(img_path, dataset_path)
                if label_path and label_path.exists():
                    pairs.append((img_path, label_path))
                    labels_found += 1
                else:
                    print(f"⚠️  Skipping (no label): {img_path.name}")
    else:
        # Directly scan in known directory
        print(f"\n✅ Found image directory, starting scan...")
        for ext in image_extensions:
            for img_path in images_dir.glob(f"*{ext}"):
                images_found += 1
                
                # Construct corresponding label path
                label_name = img_path.stem + ".txt"
                label_path = labels_dir / label_name
                
                if label_path.exists():
                    pairs.append((img_path, label_path))
                    labels_found += 1
                else:
                    print(f"⚠️  Skipping (no label): {img_path.name}")
    
    print(f"\nScan results:")
    print(f"  Images found: {images_found}")
    print(f"  Labels found: {labels_found}")
    print(f"  Valid pairs: {len(pairs)}")
    
    return pairs

def find_label_for_image(img_path: Path, dataset_root: Path) -> Path | None:
    """Find the corresponding label file for a given image"""
    # Method 1: Same name txt file in the same directory
    label_name = img_path.stem + ".txt"
    
    same_dir_label = img_path.parent / label_name
    if same_dir_label.exists():
        return same_dir_label
    
    # Method 2: Look in the corresponding labels directory
    # If image is in images/train/screenshots, label is in labels/train/screenshots
    img_relative = img_path.relative_to(dataset_root)
    img_parts = img_relative.parts
    
    if 'images' in img_parts:
        # Construct corresponding labels path
        labels_parts = list(img_parts)
        labels_parts[labels_parts.index('images')] = 'labels'
        labels_parts[-1] = label_name
        label_path = dataset_root / Path(*labels_parts)
        if label_path.exists():
            return label_path
    
    # Method 3: Recursively search in labels directories
    for labels_dir in dataset_root.rglob("labels"):
        potential_label = labels_dir / label_name
        if potential_label.exists():
            return potential_label
    
    # Method 4: Recursively search the entire dataset for the same name txt
    for txt_file in dataset_root.rglob(label_name):
        if txt_file.suffix == ".txt" and txt_file != img_path:
            return txt_file
    
    return None

def copy_pair(img_path: Path, lbl_path: Path, split: str, out_root: Path):
    """Copy image and label to the specified directory"""
    # Keep original filenames
    dst_img = out_root / "images" / split / img_path.name
    dst_lbl = out_root / "labels" / split / (img_path.stem + ".txt")
    
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(img_path, dst_img)
    shutil.copy2(lbl_path, dst_lbl)

def create_dataset_splits(dataset_path, output_name="dataset_random_split"):
    """Create dataset splits: 80% train, 10% val, 10% test"""
    
    dataset_path = Path(dataset_path)
    
    # Check if dataset directory exists
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory does not exist: {dataset_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_path}")
    print(f"{'='*60}\n")
    
    # Find all image-label pairs
    all_pairs = find_all_image_label_pairs(dataset_path)
    
    print(f"\nFound {len(all_pairs)} valid image-label pairs")
    
    if len(all_pairs) == 0:
        print("ERROR: No image-label pairs found!")
        print("Please check the dataset directory structure")
        return None
    
    # Show the first few found pairs
    print("\nFirst 5 found pairs:")
    for i, (img, lbl) in enumerate(all_pairs[:5]):
        print(f"  {i+1}. Image: {img.name}")
        print(f"      Label: {lbl.name}")
    
    # Fix random seed for reproducibility
    random.seed(42)
    pairs_shuffled = all_pairs.copy()
    random.shuffle(pairs_shuffled)
    
    total_count = len(pairs_shuffled)
    
    # Split: 80% train, 10% val, 10% test
    train_count = int(total_count * 0.8)
    val_count = int(total_count * 0.1)
    
    train_pairs = pairs_shuffled[:train_count]
    val_pairs = pairs_shuffled[train_count:train_count + val_count]
    test_pairs = pairs_shuffled[train_count + val_count:]
    
    out_root = root / output_name
    
    # Clean output directory
    if out_root.exists():
        print(f"\nCleaning existing directory: {out_root}")
        shutil.rmtree(out_root)
    
    # Copy files
    print(f"\nCreating dataset directory: {out_root}")
    print("Copying files...")
    
    print(f"  Copying training set: {len(train_pairs)} files...")
    for img, lbl in train_pairs:
        copy_pair(img, lbl, "train", out_root)
    
    print(f"  Copying validation set: {len(val_pairs)} files...")
    for img, lbl in val_pairs:
        copy_pair(img, lbl, "val", out_root)
    
    print(f"  Copying test set: {len(test_pairs)} files...")
    for img, lbl in test_pairs:
        copy_pair(img, lbl, "test", out_root)
    
    # Generate config file - use relative paths
    yaml_content = f"""path: ./{output_name}
train: images/train
val: images/val
test: images/test

names:
  0: icon
"""
    
    yaml_path = out_root / "icon.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Dataset split completed!")
    print(f"{'='*60}")
    print(f"Training set: {len(train_pairs)} images ({len(train_pairs)/total_count*100:.1f}%)")
    print(f"Validation set: {len(val_pairs)} images ({len(val_pairs)/total_count*100:.1f}%)")
    print(f"Test set: {len(test_pairs)} images ({len(test_pairs)/total_count*100:.1f}%)")
    print(f"Total:   {total_count} images")
    print(f"\nOutput directory: {out_root}")
    print(f"Config file: {yaml_path}")
    print(f"{'='*60}\n")
    
    return out_root

def main():
    parser = argparse.ArgumentParser(description="Process dataset and split into train/val/test sets")
    parser.add_argument("--dataset", type=str, default="dataset", 
                       help="Dataset path (default: dataset)")
    parser.add_argument("--output", type=str, default="dataset_split",
                       help="Output directory name (default: dataset_split)")
    args = parser.parse_args()
    
    # Convert to absolute path
    if not Path(args.dataset).is_absolute():
        dataset_path = root / args.dataset
    else:
        dataset_path = Path(args.dataset)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {args.output}")
    
    create_dataset_splits(dataset_path, args.output)

if __name__ == "__main__":
    main()
