import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def get_image_label_pairs(dataset_path: Path):
    """Return matched image/label pairs from images/train and labels/train."""
    images_dir = dataset_path / "images" / "train"
    labels_dir = dataset_path / "labels" / "train"

    image_files = []
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            image_files.append(img_path)

    pairs = []
    for img_path in image_files:
        rel_path = img_path.relative_to(images_dir)
        label_path = labels_dir / rel_path.parent / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))

    return pairs


def ensure_unique_name(existing_names, base_name: str, extension: str) -> str:
    candidate = base_name + extension
    if candidate not in existing_names:
        return candidate

    counter = 1
    while f"{base_name}_{counter}{extension}" in existing_names:
        counter += 1
    return f"{base_name}_{counter}{extension}"


def merge_datasets(dataset_paths, output_path: Path, limited_dataset_name=None, limited_dataset_max=None):
    """Merge multiple train-only YOLO datasets into one train-only dataset."""
    output_images = output_path / "images" / "train"
    output_labels = output_path / "labels" / "train"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    existing_names = set()
    copied_count = defaultdict(int)

    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        pairs = get_image_label_pairs(dataset_path)

        if limited_dataset_name and limited_dataset_max is not None and dataset_path.name == limited_dataset_name:
            pairs = pairs[:limited_dataset_max]
            print(f"Limiting {dataset_path.name} to the first {len(pairs)} samples")

        for img_path, label_path in pairs:
            img_extension = img_path.suffix
            base_name = img_path.stem

            unique_img_name = ensure_unique_name(existing_names, base_name, img_extension)
            unique_label_name = ensure_unique_name(existing_names, base_name, ".txt")

            existing_names.add(unique_img_name)
            existing_names.add(unique_label_name)

            shutil.copy2(img_path, output_images / unique_img_name)
            shutil.copy2(label_path, output_labels / unique_label_name)
            copied_count[dataset_path.name] += 1

    train_txt_path = output_path / "train.txt"
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for img_file in sorted(output_images.glob("*")):
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                f.write(f"./images/train/{img_file.name}\n")

    template_yaml = Path(dataset_paths[0]) / "data.yaml"
    if template_yaml.exists():
        shutil.copy2(template_yaml, output_path / "data.yaml")

    print("\n" + "=" * 50)
    print("Merge completed. Summary:")
    for dataset_name, count in copied_count.items():
        print(f"  {dataset_name}: {count} samples")
    print(f"  Total: {sum(copied_count.values())} samples")
    print(f"\nOutput path: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple YOLO datasets into one train-only dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset directories to merge.",
    )
    parser.add_argument(
        "--output",
        default="merged_dataset",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--limit-dataset",
        default=None,
        help="Optional dataset directory name to limit, for example dataset_2.",
    )
    parser.add_argument(
        "--limit-count",
        type=int,
        default=None,
        help="Maximum number of samples to keep from --limit-dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = [Path(path) for path in args.datasets]
    output_path = Path(args.output)

    merge_datasets(
        dataset_paths=dataset_paths,
        output_path=output_path,
        limited_dataset_name=args.limit_dataset,
        limited_dataset_max=args.limit_count,
    )


if __name__ == "__main__":
    main()
