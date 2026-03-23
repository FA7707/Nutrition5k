"""Export trained model to ONNX format for NPU inference (DirectML).

Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx
"""

import argparse

import torch

from model import NutritionModel


def parse_args():
    p = argparse.ArgumentParser(description="Export Nutrition5k model to ONNX")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--dynamic_batch", action="store_true",
                   help="Enable dynamic batch size axis")
    return p.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = ckpt["num_classes"]

    model = NutritionModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 224, 224)

    # Dynamic axes
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "cls_logits": {0: "batch_size"},
            "nutrition": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["cls_logits", "nutrition"],
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX model to {args.output}")
    print(f"  Opset: {args.opset}")
    print(f"  Num classes: {num_classes}")
    print(f"  Dynamic batch: {args.dynamic_batch}")

    # Validate with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(args.output)
        outputs = session.run(None, {"input": dummy.numpy()})
        print(f"  Validation passed — cls_logits shape: {outputs[0].shape}, "
              f"nutrition shape: {outputs[1].shape}")
    except ImportError:
        print("  onnxruntime not installed, skipping validation")


if __name__ == "__main__":
    main()
