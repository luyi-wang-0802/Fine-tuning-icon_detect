# Finetune icon_detect model

This finetuning work is based on the icon_detect model from the Omniparser framework. The objective is to adapt and specialize the original model for accurately detecting interactable icons within the Vectorworks UI. Since architectural design software interfaces often contain densely packed, visually similar, and resolution-dependent icon elements, a generic icon detector is not sufficient for reliable downstream processing. By building a tailored detector through targeted finetuning, this project aims to establish a solid foundation for subsequent tasks such as UI parsing, interaction modeling, and automated workflow extraction.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
conda create -n icon_detect_post_training python=3.10

# Activate environment
conda activate icon_detect_post_training

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
```

### 2. Dataset Preparation

#### Option 1: Use Existing Split Dataset

If you already have a pre-split train/val dataset (like `new_dataset`), you can use it directly:

```bash
# Dataset structure should be:
new_dataset/
  ├── data.yaml
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
```

`data.yaml` configuration example:
```yaml
path: ./new_dataset
train: images/train
val: images/val

names:
  0: interactive_icon
```

#### Option 2: Random Split Dataset

If you have an unsplit raw dataset, use `random_split.py` for automatic splitting:

```bash
# Basic usage (80% train / 10% val / 10% test)
python random_split.py --dataset /path/to/your/dataset --output dataset_split

# Example
python random_split.py --dataset new_dataset --output dataset_random_split
```

**Supported dataset structures:**
- Standard YOLO format (images/ and labels/ directories)
- Mixed directory structure (automatic recursive search)
- Automatic image-label matching

## 📚 Training Modes

### Mode 1: Split Dataset Training (Recommended)

**Use Case**: Have independent train/val datasets, need to evaluate model generalization

**Features:**
- ✅ Independent validation set for true performance evaluation
- ✅ Early stopping mechanism to prevent overfitting
- ✅ Complete training monitoring and evaluation

**Usage:**
```bash
python train_full_dataset.py
```

### Mode 2: Random Split Training

**Use Case**: Sufficient data, need complete train/val/test three-way split

**Features:**
- ✅ Automatic dataset splitting (80%/10%/10%)
- ✅ Independent test set for more reliable final evaluation
- ✅ Complete performance analysis report

**Usage:**
```bash
# 1. Split dataset first
python random_split.py --dataset new_dataset --output dataset_random_split

# 2. Start training
python train_random_split.py
```


## 📊 Training Outputs

After training, the following files will be generated in the `runs/` directory:

```
runs/split_train/icon_detect_split_dataset/
├── weights/
│   ├── best.pt              # Best model (based on validation mAP)
│   └── last.pt              # Last epoch model
├── results.csv              # Training metrics
├── training_curves.png      # Training curve plots
├── evaluation_results.json  # Evaluation results (JSON format)
└── [Other visualization files]
```

## 🧪 Model Testing

Test your trained model using `test_model.py`:

```bash
python test_model.py
```

**Configure test parameters** (modify in script):
```python
# Model path
best_model_path = Path("runs/split_train/icon_detect_split_dataset/weights/best.pt")

# Test image
test_image_path = Path("new_dataset/images/full_train/Confirm Layer Name_1_times_2172.png")

# Detection confidence threshold
conf = 0.5
```


## 📧 Contact & Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Join our discussion forum
- **Email**: [luyi.wang@tum.de]
- **Documentation**: [https://docs.ultralytics.com](https://docs.ultralytics.com)

## 🙏 Acknowledgments

- **Ultralytics YOLO**: For the excellent YOLOv8 framework
- **PyTorch**: For the deep learning framework
- **OpenCV**: For computer vision tools

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [YOLO Paper](https://arxiv.org/abs/2304.00501)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

**Last Updated**: 2025-12-02  
**Version**: v1.0.0  
**Python**: 3.8-3.11  
**Framework**: Ultralytics YOLOv8