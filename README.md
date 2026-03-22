# Fine-tuning icon_detect

This repository is used to fine-tune a single-class icon detection model on Vectorworks UI screenshots. The current task is to detect `interactive_icon`. The project is built around Ultralytics YOLO and already includes a split dataset, training outputs, comparison scripts, and visualization artifacts.

## 1. Project Overview

The main directories currently present in this repository are:

```text
.
├─ dataset_split/      # Split YOLO dataset
├─ icon_detect/        # Base model files and model config
├─ runs/               # Training and validation outputs
├─ vw_compare_vis/     # Comparison visualizations
├─ random_split.py     # Re-split a train set into train/val/test
├─ merge_datasets.py   # Merge multiple datasets
├─ train_full_dataset.py
├─ train_random_split.py
├─ test_model.py
├─ comparison.py
└─ requirements.txt
```

## 2. Current Data and Results

### Dataset

The dataset config currently available in the repository is [`dataset_split/icon.yaml`](./dataset_split/icon.yaml):

```yaml
path: ./dataset_split
train: images/train
val: images/val
test: images/test

names:
  0: interactive_icon
```

Actual image counts:

- `train`: 426
- `val`: 71
- `test`: 50

### Existing Training Result

The repository already contains a full training run under `runs/detect/icon_detect_new_dataset/`. Its `evaluation_results.json` reports the following metrics:

| Split | mAP50 | mAP50-95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| val | 0.9937 | 0.8817 | 0.9916 | 0.9846 |
| test | 0.9931 | 0.8847 | 0.9946 | 0.9846 |
| train | 0.9935 | 0.8903 | 0.9943 | 0.9839 |

The main training settings for that run can be found in `runs/detect/icon_detect_new_dataset/args.yaml`. Key values include:

- Base model recorded in that run: `first_finetune/weights/best.pt`
- Dataset config: `dataset_split/icon.yaml`
- `epochs=60`
- `imgsz=800`
- `batch=8`

Note: the repository currently does not contain `first_finetune/`, `new_dataset/`, `merged_dataset/`, or `dataset_1/2/3/`. Some historical outputs still reference them, and some workflows may still expect you to provide equivalent inputs.

## 3. Environment Setup

Python 3.10 is recommended.

```bash
conda create -n icon_detect_ft python=3.10
conda activate icon_detect_ft
pip install -r requirements.txt
```

Core dependencies listed in `requirements.txt` include:

- `torch`
- `torchvision`
- `ultralytics`
- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `PyYAML`
- `tqdm`

The dependency file has been cleaned and should now work directly with `pip install -r requirements.txt`.

## 4. Dataset Format

The repository uses the standard YOLO detection format for a single class:

```text
dataset_root/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

Each annotation line follows:

```text
class_id x_center y_center width height
```

Coordinates are normalized. The current class mapping is:

- `0: interactive_icon`

## 5. Script Reference

### `random_split.py`

Purpose: split an existing YOLO training set into `train/val/test`.

Behavior:

- Scans `images/train` and `labels/train`
- Uses filename-based stratified splitting by default
- Writes a new `icon.yaml`
- Uses an `80/10/10` split by default

Typical usage:

```bash
python random_split.py --dataset merged_dataset --output dataset_split --method filename
```

Arguments:

- `--dataset`: input dataset root
- `--output`: output directory name
- `--method`: `filename` or `random`

Expected input:

- The source dataset must contain at least `images/train` and `labels/train`
- Label filenames must match image filenames

### `merge_datasets.py`

Purpose: merge multiple datasets into one new training dataset.

The script now takes dataset paths from the command line instead of hardcoding them.

Example:

```bash
python merge_datasets.py --datasets dataset_1 dataset_2 dataset_3 --output merged_dataset --limit-dataset dataset_2 --limit-count 44
```

### `train_random_split.py`

Purpose: train on `dataset_split/icon.yaml` and evaluate on `train/val/test`.

Current behavior:

- Loads `icon_detect/model.pt` by default
- Uses `dataset_split/icon.yaml`
- Writes outputs to `runs/detect/icon_detect_new_dataset`
- Also generates:
  - `training_curves.png`
  - `evaluation_results.json`

Run:

```bash
python train_random_split.py
```

Requirements before running:

- `icon_detect/model.pt` must exist, or pass a different checkpoint with `--model`
- `dataset_split/icon.yaml` must be valid

Example with explicit arguments:

```bash
python train_random_split.py --data dataset_split/icon.yaml --model icon_detect/model.pt --epochs 60
```

### `train_full_dataset.py`

Purpose: train on a dataset that already has separate `train/val` splits and perform a simple overfitting check.

The script currently assumes this dataset structure:

```text
new_dataset/
├─ data.yaml
├─ images/train
└─ images/val
```

Current behavior:

- Loads `icon_detect/model.pt`
- Uses `new_dataset/data.yaml`
- Writes outputs to `runs/split_train/icon_detect_split_dataset`
- Evaluates on both `train` and `val` after training
- Generates an overfitting summary and `evaluation_results.json`

Run:

```bash
python train_full_dataset.py
```

Note: the repository currently does not include `new_dataset/`. If you want to run this script, you need to prepare that dataset or pass a different directory with `--dataset`.

### `comparison.py`

Purpose: compare the detection quality of the model before and after fine-tuning on the test set.

Current default comparison:

- Before fine-tuning: `icon_detect/model.pt`
- After fine-tuning: `runs/detect/icon_detect_new_dataset/weights/best.pt`
- Test images: `dataset_split/images/test`
- Test labels: `dataset_split/labels/test`

Outputs include:

- threshold sweep results
- PR curves
- F1 vs confidence curves
- confidence histograms
- FP / FN error examples

Run:

```bash
python comparison.py
```

Example with explicit arguments:

```bash
python comparison.py --before-model icon_detect/model.pt --after-model runs/detect/icon_detect_new_dataset/weights/best.pt --test-images dataset_split/images/test --test-labels dataset_split/labels/test
```

### `test_model.py`

Purpose: run batch inference on a fixed list of images and save visualized outputs.

Current script characteristics:

- Defaults to `icon_detect/model.pt`
- By default it looks for `image1.png` through `image10.png` in the repository root
- Outputs are written to `test_results_old/`

Run:

```bash
python test_model.py
```

Example with explicit arguments:

```bash
python test_model.py --model icon_detect/model.pt --image-dir . --images image1.png image2.png --conf 0.5
```

## 6. Recommended Workflows

### Option A: Use the existing `dataset_split`

If you want to keep the current `426/71/50` split:

```bash
python train_random_split.py
python comparison.py
```

### Option B: Re-split an existing training dataset

If you only have one unsplit training dataset:

```bash
python random_split.py --dataset <your_dataset> --output dataset_split --method filename
python train_random_split.py
python comparison.py
```

### Option C: Merge datasets first, then split and train

```bash
python merge_datasets.py
python random_split.py --dataset merged_dataset --output dataset_split --method filename
python train_random_split.py
```

This assumes you provide the dataset paths on the command line.

## 7. Important Outputs

Common outputs under `runs/` include:

- `weights/best.pt`: best checkpoint based on validation performance
- `weights/last.pt`: final checkpoint
- `results.csv`: per-epoch metrics
- `results.png`: default Ultralytics training plot
- `training_curves.png`: custom combined training plot generated by the script
- `evaluation_results.json`: post-training evaluation summary
- `confusion_matrix.png`: confusion matrix

The `vw_compare_vis/` directory also contains saved comparison images that can be used for quick inspection of qualitative improvements.

## 8. Known Issues

- Some scripts contain garbled non-ASCII comments due to encoding issues, but the main logic is still readable.
- `train_full_dataset.py` and `train_random_split.py` assume different dataset layouts, so make sure you are following the correct workflow before running either one.

## 9. Suggested Cleanup

If you plan to keep maintaining this repository, the three highest-value improvements are:

1. Unify the training entry points so the repository has one clear dataset convention and one clear experiment workflow.
2. Clean up remaining garbled comments and log messages in older scripts.
3. Add a small smoke-test script for the main workflows.

## 10. Acknowledgments

- Ultralytics YOLO
- PyTorch
- OpenCV
