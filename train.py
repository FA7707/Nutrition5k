"""Training pipeline for the Nutrition5k multi-task model.

Usage:
    python train.py --data_root data/nutrition5k --epochs 50 --batch_size 32
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Nutrition5kDataset, build_label_vocab
from model import NutritionModel


def parse_args():
    p = argparse.ArgumentParser(description="Train Nutrition5k multi-task model")
    p.add_argument("--data_root", type=str, default="data/nutrition5k",
                   help="Path to Nutrition5k dataset root")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Head learning rate")
    p.add_argument("--backbone_lr", type=float, default=1e-4, help="Backbone learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_epochs", type=int, default=5,
                   help="Epochs to freeze backbone before unfreezing")
    p.add_argument("--cls_weight", type=float, default=1.0,
                   help="Weight for classification loss")
    p.add_argument("--reg_weight", type=float, default=0.01,
                   help="Weight for regression loss")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from (loads backbone + reg_head weights)")
    p.add_argument("--finetune_cls_only", action="store_true",
                   help="Only retrain classification head; freeze backbone and reg_head")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detect if not set)")
    return p.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n_batches = 0

    for images, nutrition, ingr_labels, _ in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        nutrition = nutrition.to(device)
        ingr_labels = ingr_labels.to(device)

        cls_logits, reg_output = model(images)
        cls_loss = cls_criterion(cls_logits, ingr_labels)
        reg_loss = reg_criterion(reg_output, nutrition)
        loss = cls_weight * cls_loss + reg_weight * reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "reg_loss": total_reg_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_cal_mae = 0.0
    n_batches = 0
    n_samples = 0

    for images, nutrition, ingr_labels, _ in tqdm(loader, desc="  Eval", leave=False):
        images = images.to(device)
        nutrition = nutrition.to(device)
        ingr_labels = ingr_labels.to(device)

        cls_logits, reg_output = model(images)
        cls_loss = cls_criterion(cls_logits, ingr_labels)
        reg_loss = reg_criterion(reg_output, nutrition)
        loss = cls_weight * cls_loss + reg_weight * reg_loss

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()

        # Calorie MAE (first column of regression output)
        total_cal_mae += torch.abs(reg_output[:, 0] - nutrition[:, 0]).sum().item()
        n_samples += images.size(0)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "reg_loss": total_reg_loss / max(n_batches, 1),
        "calorie_mae": total_cal_mae / max(n_samples, 1),
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Build label vocabulary
    metadata_dir = os.path.join(args.data_root, "metadata")
    label_vocab = build_label_vocab(metadata_dir)
    num_classes = len(label_vocab)
    print(f"Number of ingredient classes: {num_classes}")

    # Datasets & loaders
    train_ds = Nutrition5kDataset(args.data_root, split="train", label_vocab=label_vocab)
    test_ds = Nutrition5kDataset(args.data_root, split="test", label_vocab=label_vocab)
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = NutritionModel(num_classes=num_classes, pretrained=True).to(device)

    # Resume from checkpoint (load backbone + reg_head, rebuild cls_head for new vocab)
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        old_state = ckpt["model_state_dict"]
        old_num_classes = ckpt["num_classes"]

        # Load all weights except cls_head if num_classes changed
        if old_num_classes != num_classes:
            print(f"  num_classes changed ({old_num_classes} -> {num_classes}), "
                  f"reinitializing cls_head")
            filtered = {k: v for k, v in old_state.items()
                        if not k.startswith("cls_head.")}
            model.load_state_dict(filtered, strict=False)
        else:
            model.load_state_dict(old_state)

    if args.finetune_cls_only:
        # Freeze backbone and regression head, only train cls_head
        print("Freezing backbone and reg_head — only training cls_head")
        model.freeze_backbone()
        for param in model.reg_head.parameters():
            param.requires_grad = False
        trainable_params = list(model.cls_head.parameters())
    else:
        model.freeze_backbone()
        trainable_params = list(model.cls_head.parameters()) + list(model.reg_head.parameters())

    # Loss functions
    cls_criterion = nn.BCEWithLogitsLoss()
    reg_criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_cal_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after warmup (skip if only finetuning cls_head)
        if epoch == args.freeze_epochs + 1 and not args.finetune_cls_only:
            print(f"Epoch {epoch}: unfreezing backbone")
            model.unfreeze_backbone()
            # Add backbone params to optimizer with lower LR
            optimizer.add_param_group({
                "params": model.backbone.parameters(),
                "lr": args.backbone_lr,
            })

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, cls_criterion, reg_criterion,
            args.cls_weight, args.reg_weight, device,
        )
        val_metrics = evaluate(
            model, test_loader, cls_criterion, reg_criterion,
            args.cls_weight, args.reg_weight, device,
        )
        scheduler.step()

        print(f"  Train — loss: {train_metrics['loss']:.4f}  "
              f"cls: {train_metrics['cls_loss']:.4f}  reg: {train_metrics['reg_loss']:.4f}")
        print(f"  Val   — loss: {val_metrics['loss']:.4f}  "
              f"cls: {val_metrics['cls_loss']:.4f}  reg: {val_metrics['reg_loss']:.4f}  "
              f"cal_MAE: {val_metrics['calorie_mae']:.1f} kcal")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        # Save best model by calorie MAE
        if val_metrics["calorie_mae"] < best_cal_mae:
            best_cal_mae = val_metrics["calorie_mae"]
            save_path = ckpt_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes,
                "label_vocab": label_vocab,
                "nutrition_stats": train_ds.nutrition_stats(),
                "calorie_mae": best_cal_mae,
            }, save_path)
            print(f"  -> Saved best model (cal MAE: {best_cal_mae:.1f} kcal)")

    # Save last model & training history
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "label_vocab": label_vocab,
        "nutrition_stats": train_ds.nutrition_stats(),
    }, ckpt_dir / "last_model.pth")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best calorie MAE: {best_cal_mae:.1f} kcal")


if __name__ == "__main__":
    main()
