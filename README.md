# Nutrition5k Food Image Classifier & Calorie Estimator

Multi-task deep learning model that identifies dishes/ingredients and estimates nutritional content (calories, fat, carbs, protein) from food images, built on the [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k) (CVPR 2021).

## Architecture

- **Backbone**: ResNet-50 (ImageNet pretrained)
- **Classification head**: Dish/ingredient identification
- **Regression head**: Calorie, fat, carb, protein estimation
- **Export**: ONNX for NPU inference (DirectML compatible)

## Dataset Setup

Download the Nutrition5k dataset and place it so the structure looks like:

```
data/
  nutrition5k/
    dish_ids/
      splits/
        rgb_train_ids.txt
        rgb_test_ids.txt
    metadata/
      dish_metadata_cafe1.csv
      dish_metadata_cafe2.csv
      ingredients_metadata.csv
    imagery/
      realsense_overhead/
        <dish_id>/
          rgb.png
          depth_color.png
```

## Usage

### Train
```bash
python train.py --data_root data/nutrition5k --epochs 50 --batch_size 32
```

### Inference
```bash
python inference.py --image path/to/food.jpg --checkpoint checkpoints/best_model.pth
```

### ONNX Export
```bash
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx
```

## References

- [Nutrition5k Dataset](https://github.com/google-research-datasets/Nutrition5k)
- Thames et al., "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food", CVPR 2021
