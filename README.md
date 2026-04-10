# Nutrition5k Food Image Classifier & Calorie Estimator

A multi-task deep learning system that identifies food dishes/ingredients and estimates nutritional content (calories, fat, carbohydrates, protein) from overhead food images. Built on the [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k) (CVPR 2021, Thames et al.).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Requirements & Installation](#requirements--installation)
- [Dataset](#dataset)
  - [Scale & Provenance](#scale--provenance)
  - [Train / Test Split](#train--test-split)
  - [Ingredient Vocabulary](#ingredient-vocabulary)
  - [Annotation Format](#annotation-format)
- [Dataset Setup](#dataset-setup)
- [Data Augmentation](#data-augmentation)
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
  - [Loss Function](#loss-function)
  - [Optimizer](#optimizer)
  - [Learning Rate Schedule](#learning-rate-schedule)
  - [Training Schedule & Warmup](#training-schedule--warmup)
  - [Hyperparameter Rationale](#hyperparameter-rationale)
  - [Checkpoints](#checkpoints)
- [References](#references)

---

## Overview

This project tackles food recognition and nutritional estimation as a joint learning problem. Given a single overhead RGB image of a food dish, the model simultaneously:

1. **Classifies ingredients** — multi-label prediction over a vocabulary of 554 ingredients
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

## Dataset

### Scale & Provenance

Nutrition5k was published by Google Research at CVPR 2021. Dishes were assembled at two university cafeterias (referred to as Cafe 1 and Cafe 2) using a controlled protocol: each dish was built ingredient-by-ingredient on a kitchen scale, so every ingredient weight is known precisely. A single overhead Intel RealSense D415 RGBD camera captured each dish from a fixed rig.

| | Count |
|--|-------|
| Total dishes in metadata | 5,006 (4,768 Cafe 1 + 238 Cafe 2) |
| Dishes with train/test splits | 4,768 |
| Unique ingredients | 554 |
| Image type | Overhead RGB PNG (Intel RealSense D415) |
| Approximate imagery size | ~3 GB |

The 238 Cafe 2 dishes are present in the metadata CSVs but are not assigned to a train/test split in the included `dish_ids/splits/` files. All 4,768 split dishes come from Cafe 1.

---

### Train / Test Split

The official split files are provided by Google Research alongside the dataset. The split was designed to avoid data leakage between dishes made from similar ingredient combinations.

| Split | Dishes | Percentage |
|-------|--------|------------|
| Train (`rgb_train_ids.txt`) | 4,059 | 85.1 % |
| Test (`rgb_test_ids.txt`) | 709 | 14.9 % |
| **Total** | **4,768** | **100 %** |

The model is trained exclusively on the 4,059 training dishes. The 709 test dishes are never seen during training and are used only for evaluation. There is no separate validation split — the test split is used for both checkpoint selection (by calorie MAE) and final reporting.

---

### Ingredient Vocabulary

`ingredients_metadata.csv` contains 554 unique ingredient entries, covering a wide range across:

- Grains and starches (white rice, brown rice, wheat berry, …)
- Proteins (chicken breast, pork, eggs, tofu, …)
- Vegetables (mixed greens, broccoli, cabbage, bok choy, …)
- Condiments and sauces (soy sauce, olive oil, mayonnaise, vinegar, …)
- Fruits, nuts, dairy, and more

Each row in the vocabulary contains the ingredient name, an ID, and per-gram nutrition stats (cal/g, fat, carb, protein). The vocabulary is sorted alphabetically by name at load time; this sorted order defines the integer index mapping used by the model's classification head and all downstream scripts.

A dish typically contains between 2 and 17 ingredients, meaning the ingredient label vector is very sparse (~1–3 % active entries on average).

---

### Annotation Format

For each dish the dataset provides:
- **Dish-level totals:** overall calories, mass (g), fat (g), carbs (g), protein (g)
- **Per-ingredient breakdown:** ingredient ID, name, grams used, plus derived calorie/fat/carb/protein contributions

The per-ingredient grams were measured directly on the scale; the per-ingredient nutritional values are computed by multiplying measured grams by the USDA-derived per-gram figures in `ingredients_metadata.csv`. The dish-level totals are the sum of all ingredient contributions.

The model regresses the dish-level totals (not per-ingredient values) because those are what you would want to know from a single photo in practice.

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

## Data Augmentation

Augmentation is applied only during training. The validation and test pipelines use a deterministic centre-crop to ensure reproducible evaluation.

### Training pipeline

```
RandomResizedCrop(224)
    → RandomHorizontalFlip()
    → ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    → ToTensor()
    → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| `RandomResizedCrop(224)` | Scale `[0.08, 1.0]`, ratio `[3/4, 4/3]` (PyTorch defaults) | Simulates dishes photographed at different distances and fills various fractions of the frame. Also prevents the model from relying on absolute dish position in the frame. |
| `RandomHorizontalFlip()` | p = 0.5 | Overhead food images have no intrinsic left/right orientation; flipping doubles the effective number of unique compositions at no labelling cost. |
| `ColorJitter` | brightness ±0.2, contrast ±0.2, saturation ±0.2, hue ±0.1 | Cafeteria lighting varies between sessions and between cafeterias. Jittering colour channels prevents the model from relying on absolute colour values and improves robustness to lighting changes. |
| `Normalize` | ImageNet mean/std | The ResNet-50 backbone was pretrained on ImageNet using these normalisation values. Applying the same normalisation ensures the feature distribution matches what the backbone expects. |

### Validation / Test pipeline

```
Resize(256)
    → CenterCrop(224)
    → ToTensor()
    → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Resizing to 256 before cropping to 224 is the standard ImageNet evaluation convention. It preserves slightly more context around the edges compared to resizing directly to 224, and matches the preprocessing used when the backbone was originally evaluated on ImageNet.

### What is intentionally not included

- **RandomRotation / RandomVerticalFlip** — overhead dishes are rotationally symmetric in principle, but the RealSense camera rig is fixed, so all training images share the same orientation. Rotating aggressively could misalign features the backbone has learned to expect upright.
- **Gaussian blur / random erasing** — not present in the current codebase. These are reasonable additions if overfitting is observed.
- **Mixup / CutMix** — the multi-hot label structure makes standard Mixup straightforward to apply, but it is not implemented here.

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

**Training transforms** (see [Data Augmentation](#data-augmentation) for full rationale):

| Split | Transforms |
|-------|-----------|
| Train | `RandomResizedCrop(224)` → `RandomHorizontalFlip(p=0.5)` → `ColorJitter(0.2, 0.2, 0.2, 0.1)` → `Normalize(ImageNet)` |
| Val/Test | `Resize(256)` → `CenterCrop(224)` → `Normalize(ImageNet)` |

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

The combined training loss is a weighted sum of two terms:

```
loss = cls_weight × BCEWithLogitsLoss(cls_logits, ingredient_labels)
     + reg_weight × MSELoss(nutrition_pred, nutrition_target)
```

**Classification term — `BCEWithLogitsLoss`**

Binary cross-entropy with logits is applied independently to each of the 554 ingredient dimensions. This is the correct loss for multi-label problems where multiple classes can be simultaneously active. The logit formulation (sigmoid computed internally) is numerically more stable than computing sigmoid first and passing probabilities to `BCELoss`.

**Regression term — `MSELoss`**

Mean squared error over the four nutrition targets (calories, fat, carbs, protein). Raw values in kcal and grams are used directly — the dataset does compute per-split mean/std statistics at load time (accessible via `dataset.nutrition_stats()`), but normalization is not applied before the loss. MSE penalises large errors more than small ones, which is appropriate here: predicting 500 kcal when the truth is 200 kcal is a qualitatively different kind of error than being off by 10 kcal.

**Loss weighting**

`cls_weight=1.0` and `reg_weight=0.01` are the defaults. The motivation:

- BCE on 554 binary labels produces values roughly in the range 0.3–0.7 per sample.
- MSE on raw calorie values (typical range 100–800 kcal) produces values on the order of 10,000–100,000.
- Without down-weighting the regression term, the MSE would dominate the gradient signal and effectively prevent the classification head from learning.
- `reg_weight=0.01` brings the regression contribution to a comparable scale, allowing both tasks to influence the shared backbone equally.

---

### Optimizer

**AdamW** (`torch.optim.AdamW`) is used with the following settings:

| Parameter | Value |
|-----------|-------|
| Head learning rate (`--lr`) | `1e-3` |
| Backbone learning rate (`--backbone_lr`) | `1e-4` |
| Weight decay | `1e-4` |
| Gradient clip (max norm) | `5.0` |

**Why AdamW over SGD?**

AdamW adapts the effective learning rate per parameter using running estimates of gradient mean and variance. This makes it much more forgiving of learning rate choices and typically reaches good solutions faster on smaller datasets, which matters here (4,059 training dishes is modest for fine-tuning a ResNet-50). SGD with momentum can match AdamW given careful tuning but requires a longer warm-up and more iterations.

The "W" in AdamW refers to decoupled weight decay. Standard Adam incorrectly folds L2 regularisation into the adaptive gradient scaling, which reduces its regularisation effect. AdamW applies weight decay directly to the weights, independent of the gradient estimates, restoring the intended regularisation behaviour.

**Why differential learning rates?**

Two separate parameter groups are registered with the optimizer:

1. **Heads** (`cls_head` + `reg_head`) at `lr=1e-3` — these are randomly initialised, so they need a higher rate to converge quickly.
2. **Backbone** (ResNet-50) at `lr=1e-4` — the backbone already contains general visual features learned from 1.2 million ImageNet images. A rate 10× lower than the heads prevents the pre-trained weights from being overwritten too quickly, which would destroy the useful representations before the heads can exploit them (catastrophic forgetting).

The backbone parameter group is added to the optimizer dynamically at epoch `freeze_epochs + 1` using `optimizer.add_param_group(...)`. This means the CosineAnnealingLR scheduler, which is set up before the backbone group is added, only governs the head LR. The backbone LR stays constant at `backbone_lr` for the remainder of training.

**Gradient clipping**

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` is applied before every optimizer step. This prevents the exploding-gradient problem that can occur early in training when the classification head is uninitialised and generating large loss gradients that propagate through 50 layers of backbone.

---

### Learning Rate Schedule

**CosineAnnealingLR** is applied with `T_max = total_epochs`:

```
lr(epoch) = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × epoch / T_max))
```

With `lr_min ≈ 0` (PyTorch default), this smoothly decays the head learning rate from `1e-3` at epoch 1 to approximately 0 at epoch `T_max`.

**Why cosine annealing over step decay?**

Step decay drops the LR by a fixed factor (e.g., ÷10) at predefined milestones. This works well but requires choosing the milestones manually and produces abrupt drops that can briefly destabilise training. Cosine annealing decays smoothly without any milestone tuning. It also has a desirable property near the end of training: the very gradual final decay allows the model to settle into a flat loss basin, which often corresponds to better generalisation.

A visual comparison for 50 epochs:

```
LR
1e-3 │▓▓▓
     │    ▓▓▓
     │        ▓▓▓
     │            ▓▓▓▓
     │                 ▓▓▓▓▓
     │                       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
~0   └──────────────────────────────────────────── epoch
     0                   25                      50
```

---

### Training Schedule & Warmup

Training proceeds in two phases governed by `--freeze_epochs` (default 5):

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Warmup (epochs 1 – freeze_epochs)                  │
│                                                             │
│  Backbone:  FROZEN  (gradients not computed)                │
│  cls_head:  training  at lr = 1e-3                          │
│  reg_head:  training  at lr = 1e-3                          │
│                                                             │
│  Purpose: let the randomly-initialised heads stabilise      │
│  before sending gradients back through the backbone.        │
│  Without this, large random head gradients would corrupt    │
│  the pre-trained backbone features in the first few steps.  │
└─────────────────────────────────────────────────────────────┘
                          ↓  (epoch freeze_epochs + 1)
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Full fine-tuning (epoch freeze_epochs+1 – end)     │
│                                                             │
│  Backbone:  UNFROZEN  at backbone_lr = 1e-4                 │
│  cls_head:  continues at lr = 1e-3                          │
│  reg_head:  continues at lr = 1e-3                          │
│                                                             │
│  Purpose: allow the backbone to adapt its low-level         │
│  features to overhead food images (vs. the upright          │
│  object-centric ImageNet images it was trained on).         │
└─────────────────────────────────────────────────────────────┘
```

When `--finetune_cls_only` is set, the backbone and regression head remain frozen for the entire run and only the classification head is updated. This is useful when adapting the model to a new or expanded ingredient vocabulary without disturbing the calorie/macro estimation.

---

### Hyperparameter Rationale

| Hyperparameter | Default | Rationale |
|---------------|---------|-----------|
| `--lr` | `1e-3` | Standard AdamW starting point for randomly-initialised heads. Higher values cause instability; lower values slow early convergence when the head weights are far from a good solution. |
| `--backbone_lr` | `1e-4` | 10× lower than head LR. Preserves pre-trained ImageNet features while still allowing the backbone to specialise for overhead food images over many epochs. |
| `--weight_decay` | `1e-4` | Light L2 regularisation. The dataset is small enough that some regularisation is beneficial, but the backbone's pre-training already acts as a strong prior, so a heavy penalty is not needed. |
| `--freeze_epochs` | `5` | 5 epochs is typically enough for random heads to produce meaningful gradients. Too few risks corrupting the backbone; too many wastes time training heads that cannot yet propagate useful signal back. |
| `--cls_weight` | `1.0` | Kept at 1.0 (no scaling). BCE on multi-label outputs naturally produces values in a reasonable range. |
| `--reg_weight` | `0.01` | Scales down MSE on raw nutritional values (range ~100–800 kcal, giving MSE ~10,000–100,000) to roughly match the scale of the BCE term (~0.3–0.7). Without this, the regression loss dominates and the model ignores ingredients. |
| `--dropout` | `0.3` | Standard value for fully-connected heads. Reduces co-adaptation between neurons. Higher values (>0.5) tend to slow convergence without meaningful regularisation benefit when the head is relatively small. |
| `--batch_size` | `32` | Fits comfortably in 8 GB VRAM with ResNet-50. Larger batches give smoother gradients but reduce the number of weight updates per epoch. 32 is a common default that works well across hardware. |
| `--epochs` | `50` | Empirically, 50 epochs is sufficient for the model to converge on this dataset size. The CosineAnnealingLR scheduler is tuned to this count; if you change `--epochs`, the schedule changes proportionally. |

---

### Checkpoints

Checkpoints are saved as standard PyTorch state dicts with additional metadata:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "num_classes": int,          # 554 for the full vocabulary
    "label_vocab": dict,         # ingredient_name → index mapping
    "nutrition_stats": dict,     # per-split mean/std for de-normalisation
    "calorie_mae": float,        # best validation calorie MAE seen so far
}
```

The `label_vocab` embedded in the checkpoint ensures the model always knows the label mapping, regardless of what metadata files are present at inference time. When `--resume` detects a `num_classes` mismatch (e.g., you added new ingredients), the `cls_head` weights are skipped and the head is re-initialised, while the backbone and regression head weights are loaded as-is.

---

## References

- [Nutrition5k Dataset (Google Research)](https://github.com/google-research-datasets/Nutrition5k)
- Thames et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food", CVPR 2021
- [PyTorch ResNet-50 pretrained weights](https://pytorch.org/vision/stable/models/resnet.html)
- [ONNX Runtime documentation](https://onnxruntime.ai/)
