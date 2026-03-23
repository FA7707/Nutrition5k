"""Nutrition5k dataset loader.

Handles CSV metadata parsing, train/test splits, and image loading for the
Nutrition5k dataset (Google Research, CVPR 2021).

Expected directory layout under data_root:
    dish_ids/splits/rgb_train_ids.txt
    dish_ids/splits/rgb_test_ids.txt
    metadata/dish_metadata_cafe1.csv
    metadata/dish_metadata_cafe2.csv
    metadata/ingredients_metadata.csv
    imagery/realsense_overhead/<dish_id>/rgb.png
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Nutrition5k dish_metadata CSV columns (no header row in original files):
# dish_id, total_calories, total_mass_g, total_fat_g, total_carb_g, total_protein_g,
# then per-ingredient groups: ingr_id, ingr_name, ingr_grams, ingr_cal, ingr_fat, ingr_carb, ingr_protein
DISH_META_COLS = [
    "dish_id", "total_calories", "total_mass_g",
    "total_fat_g", "total_carb_g", "total_protein_g",
]


def load_split_ids(split_file: str) -> list[str]:
    """Read dish IDs from a split file (one ID per line)."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def load_dish_metadata(metadata_dir: str) -> pd.DataFrame:
    """Load and merge dish metadata CSVs into a single DataFrame.

    The Nutrition5k CSV format has variable-length rows (dish-level cols +
    repeated ingredient groups). We only parse the first 6 fixed columns
    for the dish-level nutritional info.
    """
    frames = []
    for fname in sorted(Path(metadata_dir).glob("dish_metadata_cafe*.csv")):
        # Variable-width CSV — read raw lines and split manually
        rows = []
        with open(fname) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    rows.append(parts[:6])
        df = pd.DataFrame(rows, columns=DISH_META_COLS)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No dish_metadata_cafe*.csv found in {metadata_dir}"
        )

    meta = pd.concat(frames, ignore_index=True)
    # Convert numeric columns
    for col in DISH_META_COLS[1:]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.dropna(subset=["total_calories"])
    return meta


def load_ingredients_metadata(metadata_dir: str) -> pd.DataFrame:
    """Load the ingredients metadata to build a label vocabulary.

    Returns DataFrame with columns: ingr_id, ingr_name (deduplicated).
    """
    path = Path(metadata_dir) / "ingredients_metadata.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ingr_id", "ingr_name"])
    df = pd.read_csv(path, header=None, usecols=[0, 1], names=["ingr_id", "ingr_name"])
    return df.drop_duplicates(subset=["ingr_id"]).reset_index(drop=True)


def build_label_vocab(metadata_dir: str) -> dict[str, int]:
    """Build ingredient-name -> index mapping from ingredients_metadata.csv."""
    ingr = load_ingredients_metadata(metadata_dir)
    names = sorted(ingr["ingr_name"].unique())
    return {name: idx for idx, name in enumerate(names)}


def parse_dish_ingredients(metadata_dir: str) -> dict[str, list[str]]:
    """Parse per-dish ingredient lists from dish_metadata CSVs.

    Returns {dish_id: [ingredient_name, ...]}.
    """
    result: dict[str, list[str]] = {}
    for fname in sorted(Path(metadata_dir).glob("dish_metadata_cafe*.csv")):
        with open(fname) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                dish_id = parts[0]
                # Ingredient groups start at index 6, each group has 7 fields:
                # ingr_id, ingr_name, ingr_grams, ingr_cal, ingr_fat, ingr_carb, ingr_protein
                ingredients = []
                idx = 6
                while idx + 6 < len(parts):
                    ingr_name = parts[idx + 1].strip()
                    if ingr_name:
                        ingredients.append(ingr_name)
                    idx += 7
                result[dish_id] = ingredients
    return result


class Nutrition5kDataset(Dataset):
    """PyTorch Dataset for Nutrition5k.

    Each sample returns:
        image: Tensor [3, H, W]
        nutrition: Tensor [4] — (calories, fat, carbs, protein)
        ingredient_labels: Tensor [num_classes] — multi-hot ingredient vector
        dish_id: str
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        label_vocab: Optional[dict[str, int]] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.imagery_dir = self.data_root / "imagery" / "realsense_overhead"
        self.metadata_dir = self.data_root / "metadata"

        # Load split IDs
        split_file = self.data_root / "dish_ids" / "splits" / f"rgb_{split}_ids.txt"
        split_ids = set(load_split_ids(str(split_file)))

        # Load dish-level metadata and filter to split
        self.metadata = load_dish_metadata(str(self.metadata_dir))
        self.metadata = self.metadata[self.metadata["dish_id"].isin(split_ids)]

        # Filter to dishes that have images on disk
        valid = []
        for _, row in self.metadata.iterrows():
            img_path = self.imagery_dir / row["dish_id"] / "rgb.png"
            if img_path.exists():
                valid.append(row)
        self.metadata = pd.DataFrame(valid).reset_index(drop=True)

        # Label vocabulary (ingredient names -> indices)
        if label_vocab is not None:
            self.label_vocab = label_vocab
        else:
            self.label_vocab = build_label_vocab(str(self.metadata_dir))
        self.num_classes = len(self.label_vocab)

        # Per-dish ingredient lists
        self.dish_ingredients = parse_dish_ingredients(str(self.metadata_dir))

        # Image transforms
        if transform is not None:
            self.transform = transform
        else:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])

        # Compute nutrition stats for normalization
        self._cal_mean = self.metadata["total_calories"].mean()
        self._cal_std = self.metadata["total_calories"].std() + 1e-8
        self._fat_mean = self.metadata["total_fat_g"].mean()
        self._fat_std = self.metadata["total_fat_g"].std() + 1e-8
        self._carb_mean = self.metadata["total_carb_g"].mean()
        self._carb_std = self.metadata["total_carb_g"].std() + 1e-8
        self._prot_mean = self.metadata["total_protein_g"].mean()
        self._prot_std = self.metadata["total_protein_g"].std() + 1e-8

    def nutrition_stats(self) -> dict[str, tuple[float, float]]:
        """Return (mean, std) for each nutrition target, for de-normalization."""
        return {
            "calories": (self._cal_mean, self._cal_std),
            "fat": (self._fat_mean, self._fat_std),
            "carbs": (self._carb_mean, self._carb_std),
            "protein": (self._prot_mean, self._prot_std),
        }

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        dish_id = row["dish_id"]

        # Load image
        img_path = self.imagery_dir / dish_id / "rgb.png"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Nutrition targets (raw values, not normalized — let the loss handle it)
        nutrition = torch.tensor([
            row["total_calories"],
            row["total_fat_g"],
            row["total_carb_g"],
            row["total_protein_g"],
        ], dtype=torch.float32)

        # Multi-hot ingredient label
        ingr_label = torch.zeros(self.num_classes, dtype=torch.float32)
        for ingr_name in self.dish_ingredients.get(dish_id, []):
            if ingr_name in self.label_vocab:
                ingr_label[self.label_vocab[ingr_name]] = 1.0

        return image, nutrition, ingr_label, dish_id
