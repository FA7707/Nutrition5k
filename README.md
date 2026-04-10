# Nutrition5k Food Image Classifier & Calorie Estimator

A multi-task deep learning system that identifies food dishes/ingredients and estimates nutritional content (calories, fat, carbohydrates, protein) from overhead food images. Built on the [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k) (CVPR 2021, Thames et al.).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Requirements & Installation](#requirements--installation)
- [Dataset Setup](#dataset-setup)
- [Workflow](#workflow)
  - [1. Smoke Test (No Data Required)](#1-smoke-test-no-data-required)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Inference](#4-inference)
  - [5. ONNX Export](#5-onnx-export)
- [Script Reference](#script-reference)
  - [train.py](#trainpy)
  - [evaluate.py](#evaluatepy)
  - [inference.py](#inferencepy)
  - [export_onnx.py](#export_onnxpy)
  - [smoke_test.py](#smoke_testpy)
- [Data Format](#data-format)
- [Model Details](#model-details)
- [References](#references)

---

## Overview

This project tackles food recognition and nutritional estimation as a joint learning problem. Given a single overhead RGB image of a food dish, the model simultaneously:

1. **Classifies ingredients** — multi-label prediction over a vocabulary of ~550 ingredients
2. **Regresses nutrition** — estimates total calories (kcal), fat (g), carbohydrates (g), and protein (g)

The two tasks share a common ResNet-50 backbone, with separate prediction heads fine-tuned end-to-end. The trained model can be exported to ONNX for deployment on edge devices, NPUs, or DirectML-compatible hardware.

---

## Architecture

```
Input Image [B, 3, 224, 224]
         │
         ▼
  ┌─────────────────┐
  │  ResNet-50       │  ← ImageNet pretrained backbone
  │  (frozen early,  │
  │  unfrozen later) │
  └────────┬────────┘
           │
    Shared features [B, 2048]
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌──────────┐  ┌──────────┐
│  Class.  │  │  Regr.   │
│   Head   │  │   Head   │
│          │  │          │
│ Linear   │  │ Linear   │
│ 2048→512 │  │ 2048→512 │
│ ReLU     │  │ ReLU     │
│ Dropout  │  │ Dropout  │
│ 512→N    │  │ 512→128  │
│          │  │ ReLU     │
│          │  │ 128→4    │
└────┬─────┘  └────┬─────┘
     │              │
     ▼              ▼
cls_logits [B, N]   reg_output [B, 4]
(ingredient IDs)    (cal, fat, carb, protein)
```

**Key design choices:**

| Component | Detail |
|-----------|--------|
| Backbone | ResNet-50, ImageNet pretrained (`torchvision`) |
| Classification | Multi-label sigmoid, `BCEWithLogitsLoss` |
| Regression | 4-output linear, `MSELoss` |
| Training strategy | Freeze backbone for warmup epochs, then unfreeze with lower LR |
| Optimizer | AdamW with gradient clipping (max norm = 5.0) |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.3 (configurable) |

---

## Repository Structure

```
Nutrition5k-SPC4001/
│
├── model.py                  # NutritionModel: ResNet-50 dual-head architecture
├── dataset.py                # Nutrition5kDataset: PyTorch Dataset class
├── train.py                  # Training pipeline with checkpointing
├── evaluate.py               # Per-dish evaluation with detailed metrics
├── inference.py              # Single-image inference (PyTorch or ONNX)
├── export_onnx.py            # Export trained model to ONNX format
├── smoke_test.py             # Unit tests — no dataset required
│
├── requirements.txt          # Python dependencies
├── download_dataset.ps1      # PowerShell script to download the dataset
├── .gitignore
│
├── metadata/
│   ├── dish_metadata_cafe1.csv       # Per-dish nutrition & ingredient data (cafe 1)
│   ├── dish_metadata_cafe2.csv       # Per-dish nutrition & ingredient data (cafe 2)
│   └── ingredients_metadata.csv      # Ingredient vocabulary with per-gram nutrition stats
│
└── dish_ids/
    └── splits/
        ├── rgb_train_ids.txt          # Training split dish IDs
        └── rgb_test_ids.txt           # Test split dish IDs
```

> The `imagery/` directory (actual food images, ~3 GB) is excluded from version control via `.gitignore`. See [Dataset Setup](#dataset-setup) to download it.

---

## Requirements & Installation

**Python 3.8+** is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | Core deep learning framework |
| `torchvision` | >=0.15.0 | ResNet-50 pretrained model, transforms |
| `Pillow` | >=9.0.0 | Image loading |
| `pandas` | >=1.5.0 | CSV parsing for metadata |
| `scikit-learn` | >=1.2.0 | Utilities |
| `onnx` | >=1.14.0 | ONNX model format |
| `onnxruntime` | >=1.15.0 | ONNX inference runtime |
| `tqdm` | >=4.65.0 | Progress bars |
| `matplotlib` | >=3.7.0 | (Optional) Visualisation |

**GPU support:** PyTorch will automatically use CUDA if available. CPU training is supported but slow for full dataset runs. For ONNX inference on DirectML/NPU, install `onnxruntime-directml` in place of `onnxruntime`.

---

## Dataset Setup

The full Nutrition5k imagery is hosted on Google Cloud Storage. The metadata and split files are already included in this repository under `metadata/` and `dish_ids/splits/`.

### Option A — PowerShell (Windows)

```powershell
.\download_dataset.ps1
```

This script downloads all RGB and depth images (~3 GB) from `gs://nutrition5k_dataset` using `gsutil` and places them under the expected directory layout.

### Option B — Manual download

Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) to get `gsutil`, then run:

```bash
gsutil -m cp -r gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead \
    data/nutrition5k/imagery/
```

### Required directory layout

After downloading, the dataset root must look like this:

```
data/
└── nutrition5k/
    ├── dish_ids/
    │   └── splits/
    │       ├── rgb_train_ids.txt
    │       └── rgb_test_ids.txt
    ├── metadata/
    │   ├── dish_metadata_cafe1.csv
    │   ├── dish_metadata_cafe2.csv
    │   └── ingredients_metadata.csv
    └── imagery/
        └── realsense_overhead/
            ├── dish_1551457878/
            │   └── rgb.png
            ├── dish_1551458051/
            │   └── rgb.png
            └── ...
```

The `--data_root` argument in all scripts should point to the `nutrition5k/` folder above (e.g., `data/nutrition5k`).

---

## Workflow

### 1. Smoke Test (No Data Required)

Validate your environment and model code without needing any images:

```bash
python smoke_test.py
```

This runs five unit tests:
- Model forward pass with correct output shapes
- Backbone freeze / unfreeze mechanics
- Loss computation and backpropagation
- ONNX export and round-trip inference
- Checkpoint save and load

All tests pass on a CPU-only machine in under 60 seconds. Run this first whenever setting up a new environment.

---

### 2. Training

```bash
python train.py --data_root data/nutrition5k --epochs 50 --batch_size 32
```

**What happens:**

1. Loads ingredient vocabulary from `ingredients_metadata.csv` (sorted ingredient names → integer indices)
2. Creates train and validation `DataLoader`s from the split ID files
3. Initialises `NutritionModel` with ImageNet-pretrained ResNet-50
4. **Warmup phase** (first `--freeze_epochs` epochs): backbone is frozen; only the classification and regression heads are trained
5. **Fine-tuning phase** (remaining epochs): backbone is unfrozen and trained at `--backbone_lr` (typically 10× lower than head LR)
6. Saves the best checkpoint (by calorie MAE on validation set) to `checkpoints/best_model.pth`
7. Saves the final checkpoint to `checkpoints/last_model.pth`
8. Writes a `training_history.json` with per-epoch losses and metrics

**Training transforms:**

| Split | Transforms |
|-------|-----------|
| Train | `RandomResizedCrop(224)`, `ColorJitter`, `RandomHorizontalFlip`, `Normalize` |
| Val | `Resize(256)`, `CenterCrop(224)`, `Normalize` |

**All training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | *(required)* | Path to the dataset root directory |
| `--epochs` | `50` | Total training epochs |
| `--batch_size` | `32` | Dataloader batch size |
| `--lr` | `1e-3` | Learning rate for classification/regression heads |
| `--backbone_lr` | `1e-4` | Learning rate for backbone after unfreeze |
| `--freeze_epochs` | `5` | Epochs to keep backbone frozen (warmup) |
| `--cls_weight` | `1.0` | Weight applied to classification loss |
| `--reg_weight` | `0.01` | Weight applied to regression loss |
| `--dropout` | `0.3` | Dropout probability in prediction heads |
| `--num_workers` | `4` | DataLoader worker processes |
| `--device` | auto | `cuda`, `cpu`, or `mps` |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--finetune_cls_only` | `False` | Only train the classification head |

**Resuming training:**

```bash
python train.py --data_root data/nutrition5k --resume checkpoints/last_model.pth
```

If the vocabulary size in the checkpoint differs from the current dataset (e.g., you added new ingredient data), the classification head weights are skipped and re-initialised automatically.

**Fine-tuning the classification head only:**

Useful when adapting to a new ingredient list without disrupting the regression head:

```bash
python train.py --data_root data/nutrition5k \
    --resume checkpoints/best_model.pth \
    --finetune_cls_only
```

---

### 3. Evaluation

Run detailed per-dish evaluation on the test split:

```bash
python evaluate.py --data_root data/nutrition5k \
    --checkpoint checkpoints/best_model.pth
```

**Output includes:**

- Per-dish table: predicted vs actual calories, fat, carbs, protein
- Predicted vs actual ingredients for each dish
- Aggregate metrics:
  - MAE for each of the 4 nutrition targets
  - Ingredient precision, recall, and F1 score

**Save results to CSV:**

```bash
python evaluate.py --data_root data/nutrition5k \
    --checkpoint checkpoints/best_model.pth \
    --save_csv results/eval_results.csv
```

**All evaluation arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | *(required)* | Path to the dataset root directory |
| `--checkpoint` | *(required)* | Path to `.pth` checkpoint file |
| `--threshold` | `0.3` | Sigmoid threshold for ingredient prediction |
| `--top_k` | `None` | If set, show only top-K predictions per dish |
| `--batch_size` | `32` | Evaluation batch size |
| `--save_csv` | `None` | If set, write per-dish results to this CSV path |
| `--device` | auto | `cuda`, `cpu`, or `mps` |

---

### 4. Inference

Run the model on a single food image.

**PyTorch inference:**

```bash
python inference.py \
    --image path/to/food.jpg \
    --checkpoint checkpoints/best_model.pth
```

**ONNX inference:**

```bash
python inference.py \
    --image path/to/food.jpg \
    --onnx model.onnx
```

**Example output:**

```
Nutrition Estimate:
  Calories : 487.3 kcal
  Fat      : 18.6 g
  Carbs    : 62.1 g
  Protein  : 21.4 g

Top Predicted Ingredients:
  0.94  white_rice
  0.87  chicken_breast
  0.61  broccoli
  0.44  olive_oil
  0.31  garlic
```

**All inference arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | *(required)* | Path to input image |
| `--checkpoint` | `None` | PyTorch `.pth` checkpoint (mutually exclusive with `--onnx`) |
| `--onnx` | `None` | ONNX model file (mutually exclusive with `--checkpoint`) |
| `--data_root` | `None` | Required for PyTorch mode (to load vocabulary) |
| `--threshold` | `0.3` | Sigmoid threshold for ingredient inclusion |
| `--top_k` | `5` | Number of top ingredients to display |
| `--device` | auto | `cuda`, `cpu`, or `mps` (PyTorch mode only) |

---

### 5. ONNX Export

Export a trained checkpoint to ONNX for deployment:

```bash
python export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output nutrition5k.onnx
```

**With dynamic batch size (for batched server inference):**

```bash
python export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output nutrition5k_dynamic.onnx \
    --dynamic_batch
```

The script automatically validates the exported model with `onnxruntime` and prints a diff between PyTorch and ONNX outputs to confirm numerical agreement.

**ONNX model I/O:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input: `input` | `[B, 3, 224, 224]` | Normalised RGB image batch |
| Output: `cls_logits` | `[B, N]` | Raw ingredient logits (apply sigmoid) |
| Output: `nutrition` | `[B, 4]` | `[calories, fat, carbs, protein]` |

**All export arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | *(required)* | Path to `.pth` checkpoint |
| `--output` | `model.onnx` | Output ONNX file path |
| `--opset` | `18` | ONNX opset version |
| `--dynamic_batch` | `False` | Enable dynamic batch dimension |
| `--data_root` | `None` | Required to load vocabulary from metadata |
| `--device` | `cpu` | Device for export (cpu recommended) |

---

## Script Reference

### `model.py`

Defines `NutritionModel(nn.Module)`.

```python
from model import NutritionModel

model = NutritionModel(num_classes=555, dropout=0.3)
cls_logits, nutrition = model(images)  # images: [B, 3, 224, 224]
```

Key methods:

| Method | Description |
|--------|-------------|
| `freeze_backbone()` | Freeze all ResNet-50 parameters (warmup phase) |
| `unfreeze_backbone()` | Unfreeze backbone for end-to-end fine-tuning |
| `forward(x)` | Returns `(cls_logits, reg_output)` tuple |

---

### `dataset.py`

Defines `Nutrition5kDataset(Dataset)`.

```python
from dataset import Nutrition5kDataset

dataset = Nutrition5kDataset(
    data_root="data/nutrition5k",
    split="train",            # or "test"
    ingredient_vocab=vocab,   # list of ingredient name strings
    transform=transforms,
)
image, nutrition, labels, dish_id = dataset[0]
# image:     Tensor [3, 224, 224]
# nutrition: Tensor [4]  (cal, fat, carb, protein)
# labels:    Tensor [N]  (multi-hot ingredient vector)
# dish_id:   str
```

The `ingredient_vocab` list is built by sorting all ingredient names from `ingredients_metadata.csv`. The same sorted order must be used consistently between training and inference.

---

### `train.py`

Main training entry point. Handles data loading, model initialisation, the warmup + fine-tuning schedule, checkpointing, and metric logging. See [Training](#2-training) for full argument reference.

---

### `evaluate.py`

Loads a checkpoint and runs inference over the full test split. Reports per-dish predicted vs actual values, plus aggregate MAE and ingredient F1. See [Evaluation](#3-evaluation) for full argument reference.

---

### `inference.py`

Single-image prediction. Supports both PyTorch (`.pth`) and ONNX (`.onnx`) backends. Pre-processes the input image with the same normalisation used during training. See [Inference](#4-inference) for full argument reference.

---

### `export_onnx.py`

Traces the PyTorch model and serialises it to ONNX, then validates with `onnxruntime`. See [ONNX Export](#5-onnx-export) for full argument reference.

---

### `smoke_test.py`

Self-contained unit test suite. Creates a tiny synthetic model and runs five tests. No dataset or GPU required.

```bash
python smoke_test.py
# Expected: 5/5 tests passed
```

---

## Data Format

### `metadata/ingredients_metadata.csv`

Fixed-format CSV with a header row. Used to build the ingredient vocabulary.

```
ingr_name,ingr_id,cal/g,fat(g),carb(g),protein(g)
white_rice,1001,1.30,0.003,0.284,0.027
chicken_breast,1002,1.65,0.036,0.0,0.310
...
```

The vocabulary is constructed by **sorting ingredient names alphabetically** and assigning integer indices. This ordering must be consistent across training, evaluation, and inference.

### `metadata/dish_metadata_cafe1.csv` / `dish_metadata_cafe2.csv`

Variable-width CSV. Each row represents one dish. The first six columns are fixed, followed by a repeating 7-field group per ingredient:

```
dish_id, total_cal, total_mass_g, total_fat_g, total_carb_g, total_protein_g,
  [ingr_id, ingr_name, ingr_grams, ingr_cal, ingr_fat, ingr_carb, ingr_protein, ...]
```

Example:
```
dish_1551457878,487.3,412.0,18.6,62.1,21.4,1001,white_rice,220.0,286.0,...,1002,chicken_breast,...
```

### `dish_ids/splits/rgb_train_ids.txt` / `rgb_test_ids.txt`

Plain text, one dish ID per line:

```
dish_1551457878
dish_1551458051
dish_1551459012
...
```

### Images

Each dish's overhead RGB image lives at:

```
<data_root>/imagery/realsense_overhead/<dish_id>/rgb.png
```

Images are variable resolution; they are resized to 224×224 during loading.

---

## Model Details

### Loss Function

The combined training loss is a weighted sum:

```
loss = cls_weight * BCEWithLogitsLoss(cls_logits, ingredient_labels)
     + reg_weight * MSELoss(nutrition_pred, nutrition_target)
```

Default weights (`cls_weight=1.0`, `reg_weight=0.01`) are chosen to balance the scale difference between classification (per-label binary cross-entropy) and regression (raw nutritional values in kcal/g).

### Training Schedule

```
Epoch 0..freeze_epochs-1 : backbone FROZEN → train heads only at lr=1e-3
Epoch freeze_epochs..end  : backbone UNFROZEN → heads at lr=1e-3, backbone at lr=1e-4
```

CosineAnnealingLR decays over the total number of epochs.

### Checkpoints

Checkpoints are saved as standard PyTorch state dicts with additional metadata:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_cal_mae": float,
    "num_classes": int,
    "ingredient_vocab": list[str],
}
```

The `ingredient_vocab` embedded in the checkpoint ensures the model always knows the label mapping, regardless of what metadata files are available at inference time.

---

## References

- [Nutrition5k Dataset (Google Research)](https://github.com/google-research-datasets/Nutrition5k)
- Thames et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food", CVPR 2021
- [PyTorch ResNet-50 pretrained weights](https://pytorch.org/vision/stable/models/resnet.html)
- [ONNX Runtime documentation](https://onnxruntime.ai/)
