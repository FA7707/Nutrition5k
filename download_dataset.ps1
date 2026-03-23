# Download Nutrition5k dataset files
# Run this script from the Nutrition5k project folder in PowerShell

# Metadata (CSV files)
gsutil cp "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv" ./metadata/
gsutil cp "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv" ./metadata/
gsutil cp "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/ingredients_metadata.csv" ./metadata/

# Train/test split IDs
gsutil cp "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt" ./dish_ids/splits/
gsutil cp "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt" ./dish_ids/splits/

# Overhead RGB + depth images (~3 GB)
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" .
