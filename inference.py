"""Single-image inference for food recognition and nutrition estimation.

Usage:
    python inference.py --image path/to/food.jpg --checkpoint checkpoints/best_model.pth
    python inference.py --image path/to/food.jpg --onnx model.onnx --label_vocab checkpoints/best_model.pth
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms


EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on a food image")
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--checkpoint", type=str, default=None, help="PyTorch checkpoint")
    p.add_argument("--onnx", type=str, default=None, help="ONNX model path")
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of top predicted ingredients to show")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Sigmoid threshold for ingredient prediction")
    return p.parse_args()


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return EVAL_TRANSFORM(img).unsqueeze(0)  # [1, 3, 224, 224]


def infer_pytorch(image_tensor: torch.Tensor, checkpoint_path: str, top_k: int, threshold: float):
    from model import NutritionModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    label_vocab = ckpt["label_vocab"]
    idx_to_name = {v: k for k, v in label_vocab.items()}

    model = NutritionModel(num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        cls_logits, reg_output = model(image_tensor)

    display_results(cls_logits, reg_output, idx_to_name, top_k, threshold)


def infer_onnx(image_tensor: torch.Tensor, onnx_path: str, label_vocab_path: str,
               top_k: int, threshold: float):
    import onnxruntime as ort

    ckpt = torch.load(label_vocab_path, map_location="cpu", weights_only=False)
    label_vocab = ckpt["label_vocab"]
    idx_to_name = {v: k for k, v in label_vocab.items()}

    session = ort.InferenceSession(onnx_path)
    outputs = session.run(None, {"input": image_tensor.numpy()})
    cls_logits = torch.tensor(outputs[0])
    reg_output = torch.tensor(outputs[1])

    display_results(cls_logits, reg_output, idx_to_name, top_k, threshold)


def display_results(cls_logits, reg_output, idx_to_name, top_k, threshold):
    # Nutrition estimates
    nutrition = reg_output[0]
    print("\n--- Nutrition Estimate ---")
    print(f"  Calories: {nutrition[0]:.0f} kcal")
    print(f"  Fat:      {nutrition[1]:.1f} g")
    print(f"  Carbs:    {nutrition[2]:.1f} g")
    print(f"  Protein:  {nutrition[3]:.1f} g")

    # Ingredient predictions
    probs = torch.sigmoid(cls_logits[0])
    top_vals, top_idxs = probs.topk(top_k)
    print(f"\n--- Top {top_k} Predicted Ingredients ---")
    for val, idx in zip(top_vals, top_idxs):
        name = idx_to_name.get(idx.item(), f"class_{idx.item()}")
        marker = "*" if val.item() >= threshold else " "
        print(f"  {marker} {name}: {val.item():.3f}")
    print(f"\n  (* = above {threshold} threshold)")


def main():
    args = parse_args()
    image_tensor = load_image(args.image)

    if args.onnx:
        vocab_path = args.checkpoint or args.onnx.replace(".onnx", ".pth")
        infer_onnx(image_tensor, args.onnx, vocab_path, args.top_k, args.threshold)
    elif args.checkpoint:
        infer_pytorch(image_tensor, args.checkpoint, args.top_k, args.threshold)
    else:
        print("Error: provide --checkpoint or --onnx")
        return


if __name__ == "__main__":
    main()
