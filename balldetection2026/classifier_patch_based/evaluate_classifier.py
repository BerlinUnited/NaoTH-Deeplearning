import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, required=True, help="Set BOTTOM or TOP")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained .keras model")
    args = parser.parse_args()

    data_dir = f"data/{args.camera}/patches/val"
    
    print(f"Lade Modell: {args.model}")
    model = tf.keras.models.load_model(args.model)

    print(f"Lade Testdaten aus: {data_dir}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=42,
        image_size=(16, 16),
        color_mode="grayscale",
        batch_size=32,
        labels="inferred",
        label_mode="categorical",
        class_names=["noball", "ball"],
        shuffle=False 
    )

    print("Sammle wahre Labels...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true_indices = np.argmax(y_true, axis=1)

    predictions = model.predict(test_ds)
    y_pred_indices = np.argmax(predictions, axis=1)

    class_names = test_ds.class_names
    
    report_dict = classification_report(y_true_indices, y_pred_indices, target_names=class_names, zero_division=0, output_dict=True)
    
    for key, value in report_dict.items():
        if isinstance(value, dict):
            print(f"--- {key.upper()} ---")
            for metric_name, metric_val in value.items():
                print(f"{metric_name}: {metric_val:.4f}")
            print()
        else:
            print(f"{key.upper()}: {value:.4f}\n")

    error_dir = f"data/{args.camera}/errors"
    if os.path.exists(error_dir):
        shutil.rmtree(error_dir)
    
    os.makedirs(f"{error_dir}/false_positives")
    os.makedirs(f"{error_dir}/false_negatives")

    file_paths = test_ds.file_paths
    error_count = 0

    for i in range(len(y_true_indices)):
        if y_pred_indices[i] != y_true_indices[i]:
            error_count += 1
            original_path = file_paths[i]
            filename = os.path.basename(original_path)
            
            if y_pred_indices[i] == 1:
                shutil.copy(original_path, f"{error_dir}/false_positives/{filename}")
            else: 
                shutil.copy(original_path, f"{error_dir}/false_negatives/{filename}")

    print(f"Fertig! {error_count} Fehlerbilder wurden nach '{error_dir}' kopiert.")