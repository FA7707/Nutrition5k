"""Smoke test — verify model, forward pass, and ONNX export work without real data.

Usage:
    python smoke_test.py
"""

import torch
from model import NutritionModel

NUM_CLASSES = 20  # fake ingredient count


def test_model_forward():
    print("1. Testing model forward pass...")
    model = NutritionModel(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)  # batch of 2
    with torch.no_grad():
        cls_logits, reg_output = model(dummy_input)

    assert cls_logits.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {cls_logits.shape}"
    assert reg_output.shape == (2, 4), f"Expected (2, 4), got {reg_output.shape}"
    print(f"   cls_logits shape: {cls_logits.shape}  ✓")
    print(f"   reg_output shape: {reg_output.shape}  ✓")
    print(f"   Sample nutrition prediction: cal={reg_output[0][0]:.1f}, "
          f"fat={reg_output[0][1]:.1f}, carb={reg_output[0][2]:.1f}, "
          f"protein={reg_output[0][3]:.1f}")


def test_freeze_unfreeze():
    print("\n2. Testing backbone freeze/unfreeze...")
    model = NutritionModel(num_classes=NUM_CLASSES, pretrained=False)

    model.freeze_backbone()
    frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    total = sum(1 for p in model.backbone.parameters())
    assert frozen == total, "Not all backbone params are frozen"
    print(f"   Frozen: {frozen}/{total} backbone params  ✓")

    model.unfreeze_backbone()
    trainable = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    assert trainable == total, "Not all backbone params are unfrozen"
    print(f"   Unfrozen: {trainable}/{total} backbone params  ✓")


def test_loss_backward():
    print("\n3. Testing loss computation & backward pass...")
    model = NutritionModel(num_classes=NUM_CLASSES, pretrained=False)
    model.train()

    images = torch.randn(4, 3, 224, 224)
    fake_nutrition = torch.rand(4, 4) * 500  # random cal/fat/carb/protein
    fake_labels = (torch.rand(4, NUM_CLASSES) > 0.7).float()  # random multi-hot

    cls_logits, reg_output = model(images)
    cls_loss = torch.nn.BCEWithLogitsLoss()(cls_logits, fake_labels)
    reg_loss = torch.nn.MSELoss()(reg_output, fake_nutrition)
    loss = cls_loss + 0.01 * reg_loss

    loss.backward()
    print(f"   cls_loss: {cls_loss.item():.4f}  ✓")
    print(f"   reg_loss: {reg_loss.item():.4f}  ✓")
    print(f"   total_loss: {loss.item():.4f}  ✓")
    print("   Gradients computed successfully  ✓")


def test_onnx_export():
    print("\n4. Testing ONNX export...")
    model = NutritionModel(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["cls_logits", "nutrition"],
        )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"   Exported to ONNX ({size_mb:.1f} MB)  ✓")

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            outputs = session.run(None, {"input": dummy.numpy()})
            print(f"   ONNX inference — cls: {outputs[0].shape}, nutrition: {outputs[1].shape}  ✓")
        except ImportError:
            print("   onnxruntime not installed, skipping ONNX inference validation")
    finally:
        os.unlink(onnx_path)


def test_save_load_checkpoint():
    print("\n5. Testing checkpoint save/load...")
    model = NutritionModel(num_classes=NUM_CLASSES, pretrained=False)

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name

    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "label_vocab": {f"ingredient_{i}": i for i in range(NUM_CLASSES)},
            "nutrition_stats": {
                "calories": (400.0, 200.0),
                "fat": (20.0, 10.0),
                "carbs": (50.0, 25.0),
                "protein": (30.0, 15.0),
            },
        }, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2 = NutritionModel(num_classes=loaded["num_classes"], pretrained=False)
        model2.load_state_dict(loaded["model_state_dict"])
        model2.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(dummy)
            out2 = model2(dummy)
        assert torch.allclose(out1[0], out2[0]), "Cls outputs don't match"
        assert torch.allclose(out1[1], out2[1]), "Reg outputs don't match"
        print(f"   Save/load round-trip matches  ✓")
    finally:
        os.unlink(ckpt_path)


if __name__ == "__main__":
    print("=" * 50)
    print("Nutrition5k Smoke Test (no dataset needed)")
    print("=" * 50)

    test_model_forward()
    test_freeze_unfreeze()
    test_loss_backward()
    test_onnx_export()
    test_save_load_checkpoint()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
