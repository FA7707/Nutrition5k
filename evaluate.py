"""Detailed evaluation — shows per-dish predicted vs actual results.

Usage:
    python evaluate.py --data_root data/nutrition5k --checkpoint checkpoints/best_model.pth
    python evaluate.py --data_root data/nutrition5k --checkpoint checkpoints/best_model.pth --save_csv results.csv
"""

import argparse
import csv

import torch
from torch.utils.data import DataLoader

from dataset import Nutrition5kDataset
from model import NutritionModel


def parse_args():
    p = argparse.ArgumentParser(description="Detailed evaluation with per-dish results")
    p.add_argument("--data_root", type=str, default="data/nutrition5k")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Sigmoid threshold for ingredient prediction")
    p.add_argument("--top_n", type=int, default=20,
                   help="Number of individual dish results to print")
    p.add_argument("--save_csv", type=str, default=None,
                   help="Save all results to CSV file")
    return p.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    label_vocab = ckpt["label_vocab"]
    idx_to_name = {v: k for k, v in label_vocab.items()}
    num_classes = ckpt["num_classes"]

    # Model
    model = NutritionModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dataset (test split)
    test_ds = Nutrition5kDataset(
        args.data_root, split="test", label_vocab=label_vocab
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Evaluating on {len(test_ds)} test dishes...\n")

    # Collect all results
    all_results = []
    total_cal_error = 0.0
    total_fat_error = 0.0
    total_carb_error = 0.0
    total_prot_error = 0.0

    with torch.no_grad():
        for images, nutrition, ingr_labels, dish_ids in test_loader:
            cls_logits, reg_output = model(images)
            probs = torch.sigmoid(cls_logits)

            for i in range(images.size(0)):
                # Actual ingredients
                actual_idxs = (ingr_labels[i] == 1).nonzero(as_tuple=True)[0]
                actual_ingr = [idx_to_name.get(j.item(), "?") for j in actual_idxs]

                # Predicted ingredients
                pred_mask = probs[i] >= args.threshold
                pred_idxs = pred_mask.nonzero(as_tuple=True)[0]
                pred_ingr = [idx_to_name.get(j.item(), "?") for j in pred_idxs]

                # Nutrition values
                actual_cal = nutrition[i][0].item()
                actual_fat = nutrition[i][1].item()
                actual_carb = nutrition[i][2].item()
                actual_prot = nutrition[i][3].item()

                pred_cal = reg_output[i][0].item()
                pred_fat = reg_output[i][1].item()
                pred_carb = reg_output[i][2].item()
                pred_prot = reg_output[i][3].item()

                cal_err = abs(pred_cal - actual_cal)
                fat_err = abs(pred_fat - actual_fat)
                carb_err = abs(pred_carb - actual_carb)
                prot_err = abs(pred_prot - actual_prot)

                total_cal_error += cal_err
                total_fat_error += fat_err
                total_carb_error += carb_err
                total_prot_error += prot_err

                all_results.append({
                    "dish_id": dish_ids[i],
                    "actual_ingredients": actual_ingr,
                    "predicted_ingredients": pred_ingr,
                    "actual_cal": actual_cal,
                    "pred_cal": pred_cal,
                    "cal_error": cal_err,
                    "actual_fat": actual_fat,
                    "pred_fat": pred_fat,
                    "fat_error": fat_err,
                    "actual_carb": actual_carb,
                    "pred_carb": pred_carb,
                    "carb_error": carb_err,
                    "actual_prot": actual_prot,
                    "pred_prot": pred_prot,
                    "prot_error": prot_err,
                })

    n = len(all_results)

    # --- Print individual dish results ---
    print("=" * 70)
    print(f"PER-DISH RESULTS (showing {min(args.top_n, n)} of {n})")
    print("=" * 70)

    for r in all_results[:args.top_n]:
        print(f"\nDish: {r['dish_id']}")
        print(f"  Ingredients (actual):    {', '.join(r['actual_ingredients']) or 'none'}")
        print(f"  Ingredients (predicted): {', '.join(r['predicted_ingredients']) or 'none'}")
        print(f"  {'':20s} {'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
        print(f"  {'Calories (kcal)':<20s} {r['actual_cal']:>10.0f} {r['pred_cal']:>10.0f} {r['cal_error']:>10.0f}")
        print(f"  {'Fat (g)':<20s} {r['actual_fat']:>10.1f} {r['pred_fat']:>10.1f} {r['fat_error']:>10.1f}")
        print(f"  {'Carbs (g)':<20s} {r['actual_carb']:>10.1f} {r['pred_carb']:>10.1f} {r['carb_error']:>10.1f}")
        print(f"  {'Protein (g)':<20s} {r['actual_prot']:>10.1f} {r['pred_prot']:>10.1f} {r['prot_error']:>10.1f}")

    # --- Overall summary ---
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Total test dishes: {n}")
    print(f"  Mean Absolute Error:")
    print(f"    Calories: {total_cal_error / n:>8.1f} kcal")
    print(f"    Fat:      {total_fat_error / n:>8.1f} g")
    print(f"    Carbs:    {total_carb_error / n:>8.1f} g")
    print(f"    Protein:  {total_prot_error / n:>8.1f} g")

    # Ingredient accuracy metrics
    total_correct = 0
    total_predicted = 0
    total_actual = 0
    for r in all_results:
        actual_set = set(r["actual_ingredients"])
        pred_set = set(r["predicted_ingredients"])
        total_correct += len(actual_set & pred_set)
        total_predicted += len(pred_set)
        total_actual += len(actual_set)

    precision = total_correct / max(total_predicted, 1)
    recall = total_correct / max(total_actual, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    print(f"\n  Ingredient Recognition:")
    print(f"    Precision: {precision:>8.1%}")
    print(f"    Recall:    {recall:>8.1%}")
    print(f"    F1 Score:  {f1:>8.1%}")

    # --- Save to CSV ---
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dish_id",
                "actual_ingredients", "predicted_ingredients",
                "actual_cal", "pred_cal", "cal_error",
                "actual_fat", "pred_fat", "fat_error",
                "actual_carb", "pred_carb", "carb_error",
                "actual_prot", "pred_prot", "prot_error",
            ])
            for r in all_results:
                writer.writerow([
                    r["dish_id"],
                    "; ".join(r["actual_ingredients"]),
                    "; ".join(r["predicted_ingredients"]),
                    f"{r['actual_cal']:.0f}", f"{r['pred_cal']:.0f}", f"{r['cal_error']:.0f}",
                    f"{r['actual_fat']:.1f}", f"{r['pred_fat']:.1f}", f"{r['fat_error']:.1f}",
                    f"{r['actual_carb']:.1f}", f"{r['pred_carb']:.1f}", f"{r['carb_error']:.1f}",
                    f"{r['actual_prot']:.1f}", f"{r['pred_prot']:.1f}", f"{r['prot_error']:.1f}",
                ])
        print(f"\n  Results saved to {args.save_csv}")


if __name__ == "__main__":
    main()
